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
    (500.0, 1000.0,  '500keV_1000keV'),
    (1000.0, 2000.0,  '1000keV_2000keV'),
    (2000.0, 5000.0,  '2000keV_5000keV'),
    (5000.0, 10000.0,  '5000keV_10000keV'),
    (10000.0, 20000.0,  '10000keV_20000keV'),
    (20000.0, 50000.0,  '20000keV_50000keV'),
    (50000.0, 100000.0, '50000keV_over')
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
    改良版: 2段階温度決定法
    高エネルギー粒子に引きずられない「背景温度(Background Temperature)」を算出し、
    それを用いて熱的/非熱的を厳密に分離する。
    """
    # --- 1. 準備 ---
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

    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)

    # 粒子ごとのグリッド座標 (ix, iy) を先に計算
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # ===========================================================
    # ★ STEP 1: 仮の温度 (T_raw) を計算
    # ===========================================================
    print("  -> Step 1: Estimating raw temperature...")
    # 全粒子で密度とエネルギー和を計算
    H_count_all, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges])
    H_sum_E_all, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges], weights=E_kin_keV)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Mean_E_raw = H_sum_E_all / H_count_all
        Mean_E_raw[H_count_all == 0] = 0.0
    
    T_raw_map = (2.0 / 3.0) * Mean_E_raw

    # ===========================================================
    # ★ STEP 2: 背景温度 (T_bg) の再計算（ここが重要！）
    # ===========================================================
    print("  -> Step 2: Refining background temperature (removing hot tails)...")
    
    # 各粒子の場所の「仮温度」を取得
    T_particle_raw = T_raw_map[iy, ix]
    
    # 「仮温度」の 5倍 以下の粒子だけを「Core (背景プラズマ)」とみなす
    # ※ ここで厳しい基準(5.0)で高エネルギー粒子を完全に排除して温度を計算し直す
    mask_core = (E_kin_keV <= 5.0 * T_particle_raw)
    
    # Core粒子だけで統計を取り直す
    H_count_core, _, _ = np.histogram2d(Y_pos[mask_core], X_pos[mask_core], bins=[y_edges, x_edges])
    H_sum_E_core, _, _ = np.histogram2d(Y_pos[mask_core], X_pos[mask_core], bins=[y_edges, x_edges], weights=E_kin_keV[mask_core])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Mean_E_bg = H_sum_E_core / H_count_core
        # 粒子がいなくなってしまった場所は、仕方ないのでRawの値を使うか0にする
        # ここではRawの値をフォールバックとして使う
        Mean_E_bg[H_count_core == 0] = Mean_E_raw[H_count_core == 0]
        
    T_bg_map = (2.0 / 3.0) * Mean_E_bg

    # ===========================================================
    # ★ STEP 3: 本番の振り分け (Thermal vs NonThermal)
    # ===========================================================
    print("  -> Classifying particles using refined Temperature...")
    
    # 洗練された背景温度を取得
    T_particle_final = T_bg_map[iy, ix]
    
    # 閾値判定: 背景温度の 10倍 以上なら NonThermal
    # これにより、背景が冷たい場所にある高エネルギー粒子が際立つ
    threshold_final = ALPHA_THRESHOLD * T_particle_final
    
    is_nonthermal = (E_kin_keV > threshold_final) & (T_particle_final > 1e-6)
    is_thermal    = ~is_nonthermal

    # --- 4. マップの集計 (以前と同じ) ---
    print("  -> Aggregating intensity maps...")
    
    # ※密度マップは「全粒子」のものを使うのが一般的だが、
    #   熱的成分のマップには「全密度」を使うか「熱的粒子の密度」を使うか議論がある。
    #   ここでは「その成分自身の密度×エネルギー」となるよう、成分ごとに集計し直すのが最も正確。
    
    Emission_weight = np.sqrt(E_kin_keV)
    
    maps = {
        'Thermal': {},
        'NonThermal': {}
    }

    for _, _, label in ENERGY_BINS:
        maps['Thermal'][label] = np.zeros((NY, NX))
        maps['NonThermal'][label] = np.zeros((NY, NX))

    for e_min, e_max, label in ENERGY_BINS:
        # エネルギー帯域
        in_bin = (E_kin_keV >= e_min) & (E_kin_keV < e_max)
        
        # (A) Thermal
        mask_T = is_thermal & in_bin
        if np.any(mask_T):
            # ここでは単純化のため「そのビンにいる粒子の数密度 × その粒子の放射強度」を直接足しこむ
            # Intensity ~ Density_local * sqrt(E) 
            # グリッドごとの合計値として計算
            intens_map = np.zeros((NY, NX))
            # 重み = sqrt(E)。これを足し合わせれば、(ΣN) * <sqrt(E)> に近い物理量になる
            np.add.at(intens_map, (iy[mask_T], ix[mask_T]), Emission_weight[mask_T])
            maps['Thermal'][label] = intens_map
            
        # (B) NonThermal
        mask_NT = is_nonthermal & in_bin
        if np.any(mask_NT):
            intens_map = np.zeros((NY, NX))
            np.add.at(intens_map, (iy[mask_NT], ix[mask_NT]), Emission_weight[mask_NT])
            maps['NonThermal'][label] = intens_map
            
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