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

# パラメータファイルのパス (環境に合わせて書き換えてください)
# ※ イオンの質量比(Mi)を取得するために必要です
PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat')

# 定数: 電子の静止質量エネルギー (eV)
# mc^2 / e  Running approx: 510998.95 eV
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
# 計算エンジン
# =======================================================
def calculate_moments_from_particle_list(particle_data):
    """
    粒子データから密度、流体速度、速度分散(温度の元)を計算する
    """
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    Vx_raw = particle_data[:, 2]
    Vy_raw = particle_data[:, 3]
    Vz_raw = particle_data[:, 4]

    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)

    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)

    ix_masked = ix[mask]
    iy_masked = iy[mask]
    vx_masked = Vx_raw[mask]
    vy_masked = Vy_raw[mask]
    vz_masked = Vz_raw[mask]
    N_masked = len(ix_masked)

    density = np.zeros((NY, NX))
    vx_sum = np.zeros((NY, NX))
    vy_sum = np.zeros((NY, NX))
    vz_sum = np.zeros((NY, NX))
    vx2_sum = np.zeros((NY, NX))
    vy2_sum = np.zeros((NY, NX))
    vz2_sum = np.zeros((NY, NX))

    if N_masked > 0:
        np.add.at(density, (iy_masked, ix_masked), 1)
        np.add.at(vx_sum, (iy_masked, ix_masked), vx_masked)
        np.add.at(vy_sum, (iy_masked, ix_masked), vy_masked)
        np.add.at(vz_sum, (iy_masked, ix_masked), vz_masked)
        np.add.at(vx2_sum, (iy_masked, ix_masked), vx_masked**2)
        np.add.at(vy2_sum, (iy_masked, ix_masked), vy_masked**2)
        np.add.at(vz2_sum, (iy_masked, ix_masked), vz_masked**2)

    density_safe = np.where(density > 0, density, 1.0)
    
    average_vx = vx_sum / density_safe
    average_vy = vy_sum / density_safe
    average_vz = vz_sum / density_safe
    
    mean_vx2 = vx2_sum / density_safe
    mean_vy2 = vy2_sum / density_safe
    mean_vz2 = vz2_sum / density_safe
    
    # ここでの Temperature_Variance は単位なし(c^2 規格化)の分散値
    Tx = mean_vx2 - average_vx**2
    Ty = mean_vy2 - average_vy**2
    Tz = mean_vz2 - average_vz**2
    
    Temperature_Variance = (Tx + Ty + Tz) / 3.0

    mask_zero = (density == 0)
    average_vx[mask_zero] = 0.0
    average_vy[mask_zero] = 0.0
    average_vz[mask_zero] = 0.0
    Temperature_Variance[mask_zero] = 0.0

    return density, average_vx, average_vy, average_vz, Temperature_Variance

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
        print("Usage: python psd_extractor_revised.py [start] [end] [step]")
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

    # Miの取得 (eV計算用)
    MI_RATIO = load_mi_from_param(PARAM_FILE_PATH)
    print(f"--- Ion Mass Ratio set to: {MI_RATIO} ---")
    print(f"--- Rest Mass Energy (e): {REST_MASS_E_EV/1000.0:.2f} keV ---")

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
                continue

            # 1. モーメント計算 (Temperature_Variance はまだ v/c の二乗単位)
            density, av_vx, av_vy, av_vz, T_variance = calculate_moments_from_particle_list(particle_data)

            # 2. eVへの変換
            # T(eV) = Variance * Mass * c^2
            # Simulation velocity is normalized to c, so Variance is dimensionless (beta^2).
            # T(eV) = Variance * (mc^2 in eV)
            if species_label == 'electron':
                Temperature_eV = T_variance * REST_MASS_E_EV
            elif species_label == 'ion':
                Temperature_eV = T_variance * REST_MASS_E_EV * MI_RATIO
            
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