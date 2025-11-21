import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ 設定: エネルギービンと閾値
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# 1. エネルギービンの定義 (keV)
#    形式: (最小, 最大, ラベル名)
ENERGY_BINS = [
    (1.0, 100.0,    '001keV_100keV'),
    (100.0, 200.0,  '100keV_200keV'),
    (200.0, 500.0,  '200keV_500keV'),
    (500.0, 100000.0, '500keV_over')
]

# 2. 熱的/非熱的の分離しきい値
#    粒子エネルギー > ALPHA * (その場の温度 kB Te) なら「Non-Thermal」
ALPHA_THRESHOLD = 10.0 

# グリッド設定 (シミュレーションに合わせて調整)
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
DELX = 1.0

# 座標範囲
X_MIN, X_MAX = 0.0, NX * DELX
Y_MIN, Y_MAX = 0.0, NY * DELX

# =======================================================
# 計算エンジン
# =======================================================
def calculate_detailed_intensity_maps(particle_data):
    """
    粒子データを以下の2軸で分類し、それぞれのIntensityマップを作成する辞書を返す
      軸1: Thermal / NonThermal (局所温度依存)
      軸2: Energy Bin (1-100, 100-200...)
    """
    # --- 1. 準備: 座標とエネルギー計算 ---
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    vx = particle_data[:, 2]
    vy = particle_data[:, 3]
    vz = particle_data[:, 4]

    print("  -> Calculating energies...")
    v_sq = vx**2 + vy**2 + vz**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J

    # --- 2. 局所温度・密度マップの作成 ---
    print("  -> Generating local temperature map...")
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)

    # グリッドごとの粒子数(Density Proxy)
    H_count, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges])
    Density_map = H_count 

    # グリッドごとの総エネルギー -> 温度計算用
    H_sum_E, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges], weights=E_kin_keV)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Mean_E_map = H_sum_E / H_count
        Mean_E_map[H_count == 0] = 0.0
    
    # 局所温度 Te ≈ (2/3)<E>
    T_local_map = (2.0 / 3.0) * Mean_E_map

    # --- 3. 粒子の振り分け (ベクトル化) ---
    print("  -> Classifying particles...")
    
    # 粒子ごとのグリッド座標 (ix, iy)
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # 粒子位置での温度を取得
    T_particle = T_local_map[iy, ix]
    
    # タイプ判定: Thermal vs NonThermal
    threshold_E = ALPHA_THRESHOLD * T_particle
    # 温度が定義できない(粒子が少ない)場所は便宜上全部NonThermal扱い等を防ぐため条件追加
    is_nonthermal = (E_kin_keV > threshold_E) & (T_particle > 1e-6)
    is_thermal    = ~is_nonthermal

    # --- 4. マップの集計 ---
    print("  -> Aggregating intensity maps...")
    
    # 制動放射の重み (∝ sqrt(E))
    Emission_weight = np.sqrt(E_kin_keV)
    
    # 結果を格納する辞書: maps[Type][BinLabel]
    maps = {
        'Thermal': {},
        'NonThermal': {}
    }

    # すべてのビンに対してマップを初期化
    for _, _, label in ENERGY_BINS:
        maps['Thermal'][label] = np.zeros((NY, NX))
        maps['NonThermal'][label] = np.zeros((NY, NX))

    # ループでビンごとに処理 (メモリ節約のためビンループ)
    for e_min, e_max, label in ENERGY_BINS:
        # エネルギー帯域マスク
        in_bin = (E_kin_keV >= e_min) & (E_kin_keV < e_max)
        
        # (A) Thermal かつ このビン
        mask_T = is_thermal & in_bin
        if np.any(mask_T):
            # ソース項 (Sum sqrt(E)) を計算
            source_map = np.zeros((NY, NX))
            np.add.at(source_map, (iy[mask_T], ix[mask_T]), Emission_weight[mask_T])
            # Intensity = Density * Source
            maps['Thermal'][label] = Density_map * source_map
            
        # (B) NonThermal かつ このビン
        mask_NT = is_nonthermal & in_bin
        if np.any(mask_NT):
            source_map = np.zeros((NY, NX))
            np.add.at(source_map, (iy[mask_NT], ix[mask_NT]), Emission_weight[mask_NT])
            maps['NonThermal'][label] = Density_map * source_map
            
    return maps

# =======================================================
# 保存関数 (TXTのみ)
# =======================================================
def save_category_txt(maps_dict, timestep, output_base):
    """
    辞書に入ったマップをTXTファイルとして保存する (画像プロットなし)
    """
    for particle_type in ['Thermal', 'NonThermal']:
        type_dir = os.path.join(output_base, particle_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for e_min, e_max, bin_label in ENERGY_BINS:
            
            intensity_map = maps_dict[particle_type][bin_label]
            
            # --- TXT保存 ---
            # フォルダ階層: output/Thermal/100-200keV/intensity_...
            bin_dir = os.path.join(type_dir, bin_label)
            os.makedirs(bin_dir, exist_ok=True)
            
            txt_name = f'intensity_{particle_type}_{bin_label}_{timestep}.txt'
            txt_path = os.path.join(bin_dir, txt_name)
            
            header = (f'Bremsstrahlung Intensity Map\n'
                      f'Type: {particle_type}\n'
                      f'Energy Bin: {bin_label} ({e_min}-{e_max} keV)\n'
                      f'Timestep: {timestep}\n'
                      f'Shape: ({NY}, {NX})')
            
            # データが空(ゼロ)でも保存する（プロッターがエラーにならないように）
            np.savetxt(txt_path, intensity_map, header=header, fmt='%.6g')
            print(f"    Saved TXT: {particle_type} - {bin_label}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python detailed_brems_txt_only.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # ディレクトリ設定
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_detailed_intensity')
    
    print(f"--- Detailed Bremsstrahlung Extraction (TXT Only) ---")
    print(f"Output: {OUTPUT_DIR}")
    
    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        print(f"\n=== Processing TS: {ts} ===")
        
        fpath = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_e.dat')
        if not os.path.exists(fpath):
            print("  File not found.")
            continue
            
        try:
            data = np.loadtxt(fpath)
            if data.size == 0: continue
            if data.ndim == 1: data = data.reshape(1, -1)
        except: continue
        
        # 計算
        result_maps = calculate_detailed_intensity_maps(data)
        
        # 保存 (TXTのみ)
        save_category_txt(result_maps, ts, OUTPUT_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()