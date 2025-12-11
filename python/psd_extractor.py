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
PARAM_FILE_PATH = os.path.join('/data/shok/dat/init_param.dat')

# 定数: 電子の静止質量エネルギー (eV)
# mc^2 (J) / e (C) = eV
REST_MASS_E_EV = (m_e * c**2) / elementary_charge

# =======================================================
# ヘルパー関数: パラメータ読み込み
# =======================================================
def load_mi_from_param(param_filepath):
    mi_val = 1.0
    try:
        if not os.path.exists(param_filepath):
            return 100.0
        with open(param_filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('Mi, Me'):
                    parts = line.split()
                    mi_val = float(parts[3])
                    return mi_val
    except:
        return 100.0
    return mi_val

# =======================================================
# 計算エンジン (修正版: 固有速度対応 & 熱エネルギー分離)
# =======================================================
def calculate_moments_relativistic(particle_data):
    """
    粒子の生データ(固有速度 u = gamma*v/c と仮定)から
    密度、流体速度、および温度(eV)を計算する。
    
    T = (2/3) * ( <E_total> - E_bulk )
    """
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    # 1. データ展開
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    
    # PICコードの出力は通常、固有速度 u = gamma * v/c (運動量 p/mc)
    ux = particle_data[:, 2]
    uy = particle_data[:, 3]
    uz = particle_data[:, 4]
    
    # 2. 粒子ごとのガンマ因子と全運動エネルギー (単位: mc^2)
    # gamma = sqrt(1 + u^2) を使用 (これがエラー回避のキモです)
    u_sq = ux**2 + uy**2 + uz**2
    gamma_particle = np.sqrt(1.0 + u_sq)
    E_kin_particle = gamma_particle - 1.0 # 全運動エネルギー

    # 3. グリッド定義
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    # 4. マスク処理
    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)
    
    curr_X = X_pos[mask]
    curr_Y = Y_pos[mask]
    curr_ux = ux[mask]
    curr_uy = uy[mask]
    curr_uz = uz[mask]
    curr_E  = E_kin_particle[mask]

    # 5. グリッドへの集計
    # (A) 粒子数密度
    H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins])
    
    # (B) 平均固有速度 (流体の運動量のようなもの)
    H_sum_ux, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_ux)
    H_sum_uy, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_uy)
    H_sum_uz, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_uz)
    
    # (C) 平均全エネルギー
    H_sum_E, _, _  = np.histogram2d(curr_Y, curr_X, bins=[y_bins, x_bins], weights=curr_E)

    # 6. 平均値計算
    with np.errstate(divide='ignore', invalid='ignore'):
        density_safe = H_count.copy()
        density_safe[density_safe == 0] = 1.0
        
        # 平均固有速度 <u_fluid>
        av_ux = H_sum_ux / density_safe
        av_uy = H_sum_uy / density_safe
        av_uz = H_sum_uz / density_safe
        
        # 平均全エネルギー <E_total>
        Mean_E_total = H_sum_E / density_safe

    # 7. 温度計算 (熱エネルギー = 全エネルギー - バルクエネルギー)
    # バルクのガンマ因子
    u_fluid_sq = av_ux**2 + av_uy**2 + av_uz**2
    gamma_fluid = np.sqrt(1.0 + u_fluid_sq)
    E_bulk = gamma_fluid - 1.0
    
    # 熱エネルギー (単位: mc^2)
    E_thermal = Mean_E_total - E_bulk
    
    # 数値誤差でマイナスになるのを防ぐ
    E_thermal = np.clip(E_thermal, 0.0, None)

    # 温度 T = (2/3) * E_thermal
    T_norm = (2.0 / 3.0) * E_thermal

    # 粒子がいない場所のクリーンアップ
    mask_zero = (H_count == 0)
    av_ux[mask_zero] = 0.0
    av_uy[mask_zero] = 0.0
    av_uz[mask_zero] = 0.0
    T_norm[mask_zero] = 0.0
    
    # 流体速度を物理速度 v/c に変換して返す (プロット用)
    # v = u / gamma
    av_vx = av_ux / gamma_fluid
    av_vy = av_uy / gamma_fluid
    av_vz = av_uz / gamma_fluid
    av_vx[mask_zero] = 0.0
    av_vy[mask_zero] = 0.0
    av_vz[mask_zero] = 0.0

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

    data_dir = os.path.join('/data/shok/psd/')
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.') 
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                continue

            print(f"  -> Calculating relativistic T (bulk-subtracted) for {species_label}...")
            
            density, av_vx, av_vy, av_vz, T_norm = calculate_moments_relativistic(particle_data)

            # eVへの変換
            if species_label == 'electron':
                Temperature_eV = T_norm * REST_MASS_E_EV
            elif species_label == 'ion':
                Temperature_eV = T_norm * (REST_MASS_E_EV * MI_RATIO)
            
            max_T = np.max(Temperature_eV)
            mean_T = np.mean(Temperature_eV[density > 0])
            print(f"     Max T: {max_T:.2f} eV, Mean T: {mean_T:.2f} eV")

            save_data_to_txt(density, 'Density', timestep, species_label, OUTPUT_DIR, 'density_count')
            save_data_to_txt(av_vx, 'Vx', timestep, species_label, OUTPUT_DIR, 'Vx')
            save_data_to_txt(av_vy, 'Vy', timestep, species_label, OUTPUT_DIR, 'Vy')
            save_data_to_txt(av_vz, 'Vz', timestep, species_label, OUTPUT_DIR, 'Vz')
            save_data_to_txt(Temperature_eV, 'Temperature (eV)', timestep, species_label, OUTPUT_DIR, 'T')

    print("\nDone.")

if __name__ == "__main__":
    main()