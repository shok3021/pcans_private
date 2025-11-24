import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ 設定: エネルギービンと閾値
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# 1. エネルギービンの定義 (keV)
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

# 2. マクスウェル分布フィッティング設定
# -------------------------------------------------------
# ★ FIT_CUTOFF_RATIO:
# 温度を決めるためのフィッティング時に、暫定温度の何倍以上を「外れ値」として捨てるか。
# マクスウェル分布のコアを厳密に捉えるには 3.0 ~ 4.0 程度が適切です。
FIT_CUTOFF_RATIO = 3.0 

# ★ MAX_ITER:
# 温度収束計算の反復回数。通常5〜10回で十分収束します。
MAX_ITER = 10

# ★ ALPHA_THRESHOLD:
# 最終的な分類（Thermal vs Non-Thermal）のしきい値。
# 収束した「真の背景温度」に対して、何倍以上を非熱的とするか。
# 通常、熱的分布は数倍の温度で指数関数的に減少するため、10倍あれば確実に非熱的です。
ALPHA_THRESHOLD = 10.0 
# -------------------------------------------------------

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
DELX = 1.0

# 座標範囲
X_MIN, X_MAX = 0.0, NX * DELX
Y_MIN, Y_MAX = 0.0, NY * DELX

# =======================================================
# 計算エンジン (反復フィッティング版)
# =======================================================
def calculate_detailed_intensity_maps(particle_data):
    """
    改良版: 反復的マクスウェル・フィッティング (Iterative Maxwellian Fitting)
    
    グリッドごとに局所的な速度分布関数を確認し、
    高エネルギー粒子の影響を排除した「真の熱的温度 (T_core)」が収束するまで計算を繰り返す。
    """
    # --- 1. 粒子情報の展開 ---
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

    # 粒子ごとのグリッド座標 (ix, iy)
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # ===========================================================
    # ★ 反復的温度収束プロセス (Iterative Core Temperature Fitting)
    # ===========================================================
    print(f"  -> Iterative fitting of Maxwellian core (Max Iter: {MAX_ITER})...")

    # 初期状態: 全粒子を「熱的フィッティング候補」とする
    mask_fitting_candidate = np.ones(len(E_kin_keV), dtype=bool)
    
    # 温度マップの初期化 (適当な値を入れておく、初回ループで上書きされる)
    T_core_map = np.zeros((NY, NX))

    for i in range(MAX_ITER):
        # 現在の候補粒子だけで統計をとる
        # 重み付きヒストグラムを使うことで、forループを使わずにグリッド計算
        
        # 候補粒子の抽出
        curr_X = X_pos[mask_fitting_candidate]
        curr_Y = Y_pos[mask_fitting_candidate]
        curr_E = E_kin_keV[mask_fitting_candidate]
        
        if len(curr_E) == 0:
            print(f"     Iter {i+1}: No particles left for fitting (unlikely). Break.")
            break

        # 密度(カウント)とエネルギー密度(ΣE)を計算
        H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges])
        H_sum_E, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges], weights=curr_E)
        
        # 温度計算: T = (2/3) * <E>
        with np.errstate(divide='ignore', invalid='ignore'):
            Mean_E = H_sum_E / H_count
            Mean_E[H_count == 0] = 0.0
        
        T_current_iter = (2.0 / 3.0) * Mean_E
        
        # NaNケア (粒子がない場所は0にする)
        T_current_iter = np.nan_to_num(T_current_iter, nan=0.0)
        
        # 保存
        T_core_map = T_current_iter.copy()
        
        # --- 次の反復のための選別 ---
        # 各粒子の場所の温度を取得
        # 注: ここでは全粒子に対して温度を割り当てる（候補外になった粒子も復活の可能性を持たせるため、全数チェックが望ましいが、
        #     収束を早めるため、通常は「現在より厳しい基準」で絞っていく）
        
        T_particle = T_core_map[iy, ix]
        
        # 新しいマスクを作成: エネルギーが "FIT_CUTOFF_RATIO * T" 以下のものだけを次の「熱的コア」とする
        # 例: 3倍の温度より熱いものは「フィッティング用の計算」からは除外する（※最終分類ではない）
        # 温度が0(粒子なし)の場所は、全ての粒子を含める(マスクTrue)か除外するかだが、便宜上除外しないように閾値を工夫
        
        # T_particleが0の場所での除算エラーを防ぐ
        safe_T = T_particle.copy()
        safe_T[safe_T < 1e-9] = 1.0 # ダミー値 (どうせエネルギー判定で落ちないようにするか、別途処理)
        
        ratio = E_kin_keV / safe_T
        
        # T=0の場所は粒子が一つのみ、あるいはコールドなので、すべてFittingに使う
        ratio[T_particle < 1e-9] = 0.0 
        
        # 更新: コア分布に近いものだけを残す
        mask_fitting_candidate = (ratio <= FIT_CUTOFF_RATIO)
        
        # ログ出力（進捗確認用）
        n_candidates = np.sum(mask_fitting_candidate)
        print(f"     Iter {i+1}/{MAX_ITER}: Particles in thermal core fit = {n_candidates} / {len(E_kin_keV)}")
        
        # 変化が少なければ早期終了してもよいが、念のため指定回数回す

    # ===========================================================
    # ★ 最終分類 (Final Classification)
    # ===========================================================
    print("  -> Final classification using converged core temperature...")
    
    # 最終的に収束した温度 T_core_map を使う
    T_final = T_core_map[iy, ix]
    
    # しきい値判定
    # ここではユーザー指定の ALPHA_THRESHOLD (例: 10倍) を使う
    # ※ T_final は「高エネルギー粒子を完全に排除して計算された純粋な背景温度」なので、
    #    これの10倍を超える粒子は、マクスウェル分布の確率的にほぼあり得ない＝確実に非熱的であると言える。
    
    threshold_energy = ALPHA_THRESHOLD * T_final
    
    # 温度が極端に低い(計算不能)場所のケア
    valid_T = (T_final > 1e-9)
    
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold_energy[valid_T])
    
    # マクスウェル分布に従っているもの（Thermal）
    is_thermal = ~is_nonthermal

    # ===========================================================
    # ★ マップ集計
    # ===========================================================
    print("  -> Aggregating intensity maps...")
    
    Emission_weight = np.sqrt(E_kin_keV)
    
    maps = {
        'Thermal': {},
        'NonThermal': {}
    }

    # マップ初期化
    for _, _, label in ENERGY_BINS:
        maps['Thermal'][label] = np.zeros((NY, NX))
        maps['NonThermal'][label] = np.zeros((NY, NX))

    # ビンごとの集計
    for e_min, e_max, label in ENERGY_BINS:
        # エネルギー帯域フィルタ
        in_bin = (E_kin_keV >= e_min) & (E_kin_keV < e_max)
        
        # (A) Thermal Map
        mask_T = is_thermal & in_bin
        if np.any(mask_T):
            intens_map = np.zeros((NY, NX))
            np.add.at(intens_map, (iy[mask_T], ix[mask_T]), Emission_weight[mask_T])
            maps['Thermal'][label] = intens_map
            
        # (B) NonThermal Map
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
    辞書に入ったマップをTXTファイルとして保存する
    """
    for particle_type in ['Thermal', 'NonThermal']:
        type_dir = os.path.join(output_base, particle_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for e_min, e_max, bin_label in ENERGY_BINS:
            intensity_map = maps_dict[particle_type][bin_label]
            
            bin_dir = os.path.join(type_dir, bin_label)
            os.makedirs(bin_dir, exist_ok=True)
            
            txt_name = f'intensity_{particle_type}_{bin_label}_{timestep}.txt'
            txt_path = os.path.join(bin_dir, txt_name)
            
            header = (f'Bremsstrahlung Intensity Map\n'
                      f'Type: {particle_type}\n'
                      f'Energy Bin: {bin_label} ({e_min}-{e_max} keV)\n'
                      f'Timestep: {timestep}\n'
                      f'Shape: ({NY}, {NX})')
            
            np.savetxt(txt_path, intensity_map, header=header, fmt='%.6g')
            print(f"    Saved TXT: {particle_type} - {bin_label}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python detailed_brems_iterative.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # ディレクトリ設定 (環境に合わせて調整してください)
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_maxwell_intensity')
    
    print(f"--- Detailed Bremsstrahlung Extraction (Iterative Maxwellian Fit) ---")
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
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue
        
        # 計算
        result_maps = calculate_detailed_intensity_maps(data)
        
        # 保存
        save_category_txt(result_maps, ts, OUTPUT_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()