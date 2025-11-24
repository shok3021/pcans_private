import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# 設定
# =======================================================
GLOBAL_NX_GRID_POINTS = 321  
GLOBAL_NY_GRID_POINTS = 640
GLOBAL_NX_PHYS = GLOBAL_NX_GRID_POINTS - 1 
GLOBAL_NY_PHYS = GLOBAL_NY_GRID_POINTS - 1 
DELX = 1.0 

X_MIN = 0.0           
X_MAX = GLOBAL_NX_PHYS * DELX 
Y_MIN = 0.0           
Y_MAX = GLOBAL_NY_PHYS * DELX 

# パラメータファイルのパス
PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat')

# 定数: 電子の静止質量エネルギー (eV)
# mc^2 (J) / e (C) = eV
REST_MASS_E_EV = (m_e * c**2) / elementary_charge

# =======================================================
# ヘルパー関数: パラメータ読み込み
# =======================================================
def load_mi_from_param(param_filepath):
    """
    init_param.dat から Mi (イオン質量比 = mi/me) を読み込む
    """
    mi_val = 1.0 # デフォルト
    try:
        if not os.path.exists(param_filepath):
            print(f"警告: {param_filepath} が見つかりません。Mi=100.0 (仮) として計算します。")
            return 100.0

        with open(param_filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('Mi, Me'):
                    parts = line.split()
                    mi_val = float(parts[3])
                    print(f"  -> パラメータファイルより Mi = {mi_val} を取得しました。")
                    return mi_val
    except Exception as e:
        print(f"警告: パラメータ読み込みエラー ({e})。Mi=100.0 として計算します。")
        return 100.0
    return mi_val

# =======================================================
# 計算エンジン (相対論的エネルギー版)
# =======================================================
def calculate_moments_relativistic(particle_data):
    """
    粒子の生データから密度、流体速度、および
    相対論的運動エネルギーに基づいた温度 (T = 2/3 * <Ek>) を計算する。
    
    Returns:
        density: 粒子数密度
        vx, vy, vz: 平均速度 (c単位)
        T_norm: 規格化された温度 (単位: mc^2)。後でeVに変換する。
    """
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    # 1. データ展開
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    vx = particle_data[:, 2]
    vy = particle_data[:, 3]
    vz = particle_data[:, 4]
    
    # 2. 相対論的運動エネルギー (正規化単位: mc^2) の計算
    # 参照コードと同じロジック: E_kin = (gamma - 1.0) * mc^2
    # ここでは mc^2 を掛けずに (gamma - 1.0) だけを計算し、後で質量を掛ける
    v_sq = vx**2 + vy**2 + vz**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12) # 光速を超えないようにクリップ
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_norm = gamma - 1.0 # 単位: mc^2

    # 3. グリッドの定義
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    # 4. 粒子座標のフィルタリングとインデックス計算
    # numpy.histogram2d を使うため、マスク処理は最小限で良いが、
    # 範囲外の粒子を除外しておく
    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)
    
    curr_X = X_pos[mask]
    curr_Y = Y_pos[mask]
    curr_vx = vx[mask]
    curr_vy = vy[mask]
    curr_vz = vz[mask]
    curr_E  = E_kin_norm[mask]

    # 5. グリッドへの集計 (ヒストグラム)
    # (A) 粒子数密度 (Density)
    H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins])
    
    # (B) 速度の和 (流体速度用)
    H_sum_vx, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_vx)
    H_sum_vy, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_vy)
    H_sum_vz, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_vz)
    
    # (C) 運動エネルギーの和 (温度用)
    H_sum_E, _, _  = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_E)

    # 6. 平均値の計算
    # ゼロ除算を防ぐマスク処理
    with np.errstate(divide='ignore', invalid='ignore'):
        density_safe = H_count.copy()
        density_safe[density_safe == 0] = 1.0
        
        av_vx = H_sum_vx / density_safe
        av_vy = H_sum_vy / density_safe
        av_vz = H_sum_vz / density_safe
        
        Mean_E = H_sum_E / density_safe

    # 粒子がいない場所を0に戻す
    mask_zero = (H_count == 0)
    av_vx[mask_zero] = 0.0
    av_vy[mask_zero] = 0.0
    av_vz[mask_zero] = 0.0
    Mean_E[mask_zero] = 0.0
    
    # 7. 温度計算
    # 参照コードに従い、T = (2/3) * <E_kin> とする
    # ※ これは「運動エネルギーから換算した等価温度」です
    T_norm = (2.0 / 3.0) * Mean_E

    return H_count, av_vx, av_vy, av_vz, T_norm

def load_text_data(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            if data.size == 0: return np.array([])
            data = data.reshape(1, -1)
        if data.size == 0: return np.array([])
        return data
    except:
        return None

def save_data_to_txt(data_2d, label, timestep, species, out_dir, filename):
    output_file = os.path.join(out_dir, f'data_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"  Saved: {species} {filename}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python psd_extractor_relativistic.py [start] [end] [step]")
        sys.exit(1)

    start_step = int(sys.argv[1])
    end_step   = int(sys.argv[2])
    step_size  = int(sys.argv[3])

    # ディレクトリ設定
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.') 
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Miの取得
    MI_RATIO = load_mi_from_param(PARAM_FILE_PATH)
    print(f"--- Ion Mass Ratio set to: {MI_RATIO} ---")
    print(f"--- Electron Rest Energy: {REST_MASS_E_EV:.2f} eV ---")

    species_list = [('e', 'electron'), ('i', 'ion')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        timestep = f"{current_step:06d}" 
        print(f"\n=== Processing TS: {timestep} ===")

        for suffix, species_label in species_list:
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)

            particle_data = load_text_data(filepath)
            if particle_data is None or particle_data.size == 0:
                print(f"  Skipping {species_label} (no data)")
                # データがない場合はゼロ配列を作って保存（エラー回避のため）
                # 必要に応じてスキップでも可
                continue

            print(f"  -> Calculating relativistic moments for {species_label}...")
            
            # 1. 相対論的モーメント計算
            # T_norm は "mc^2" 単位の温度
            density, av_vx, av_vy, av_vz, T_norm = calculate_moments_relativistic(particle_data)

            # 2. eVへの変換
            # T(eV) = T_norm * (その粒子のmc^2 in eV)
            if species_label == 'electron':
                Temperature_eV = T_norm * REST_MASS_E_EV
            elif species_label == 'ion':
                Temperature_eV = T_norm * (REST_MASS_E_EV * MI_RATIO)
            
            # 統計情報の表示（デバッグ用）
            max_T = np.max(Temperature_eV)
            mean_T = np.mean(Temperature_eV[density > 0])
            print(f"     Max T: {max_T:.2f} eV, Mean T (where den>0): {mean_T:.2f} eV")

            # 3. 保存
            save_data_to_txt(density, 'Density', timestep, species_label, OUTPUT_DIR, 'density_count')
            save_data_to_txt(av_vx, 'Vx', timestep, species_label, OUTPUT_DIR, 'Vx')
            save_data_to_txt(av_vy, 'Vy', timestep, species_label, OUTPUT_DIR, 'Vy')
            save_data_to_txt(av_vz, 'Vz', timestep, species_label, OUTPUT_DIR, 'Vz')
            
            # ★ eV単位で保存
            save_data_to_txt(Temperature_eV, 'Temperature (eV)', timestep, species_label, OUTPUT_DIR, 'T')

    print("\nDone.")

if __name__ == "__main__":
    main()