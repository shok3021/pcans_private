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
FIT_CUTOFF_RATIO = 3.0 
MAX_ITER = 10
ALPHA_THRESHOLD = 10.0 

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
X_MIN, X_MAX = 0.0, NX * 1.0
Y_MIN, Y_MAX = 0.0, NY * 1.0

# =======================================================
# 計算エンジン (エネルギー密度版)
# =======================================================
def calculate_energy_density_maps(particle_data):
    """
    反復的マクスウェルフィッティングを行い、
    Thermal/Non-Thermalに分離した「エネルギー密度マップ」を計算する。
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

    # 粒子ごとのグリッド座標
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # ===========================================================
    # ★ 反復的温度収束プロセス (分離ロジックは変えない)
    # ===========================================================
    print(f"  -> Iterative fitting of Maxwellian core (Max Iter: {MAX_ITER})...")

    mask_fitting_candidate = np.ones(len(E_kin_keV), dtype=bool)
    T_core_map = np.zeros((NY, NX))

    for i in range(MAX_ITER):
        curr_X = X_pos[mask_fitting_candidate]
        curr_Y = Y_pos[mask_fitting_candidate]
        curr_E = E_kin_keV[mask_fitting_candidate]
        
        if len(curr_E) == 0: break

        H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges])
        H_sum_E, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges], weights=curr_E)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Mean_E = H_sum_E / H_count
            Mean_E[H_count == 0] = 0.0
        
        T_current_iter = (2.0 / 3.0) * Mean_E
        T_current_iter = np.nan_to_num(T_current_iter, nan=0.0)
        T_core_map = T_current_iter.copy()
        
        T_particle = T_core_map[iy, ix]
        safe_T = T_particle.copy()
        safe_T[safe_T < 1e-9] = 1.0
        
        ratio = E_kin_keV / safe_T
        ratio[T_particle < 1e-9] = 0.0 
        
        mask_fitting_candidate = (ratio <= FIT_CUTOFF_RATIO)
        
        n_candidates = np.sum(mask_fitting_candidate)
        # print(f"     Iter {i+1}: {n_candidates} particles")

    # ===========================================================
    # ★ 最終分類
    # ===========================================================
    print("  -> Final classification...")
    T_final = T_core_map[iy, ix]
    threshold_energy = ALPHA_THRESHOLD * T_final
    valid_T = (T_final > 1e-9)
    
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold_energy[valid_T])
    is_thermal = ~is_nonthermal

    # ===========================================================
    # ★ マップ集計 (ここを変更)
    # ===========================================================
    print("  -> Aggregating ENERGY DENSITY maps...")
    
    # 【重要】制動放射強度(sqrt E)ではなく、エネルギーそのもの(E)を足す
    # これにより「エネルギー密度 (Pressure相当)」のマップになる
    Weight = E_kin_keV
    
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
        in_bin = (E_kin_keV >= e_min) & (E_kin_keV < e_max)
        
        # (A) Thermal Energy Density
        mask_T = is_thermal & in_bin
        if np.any(mask_T):
            dens_map = np.zeros((NY, NX))
            np.add.at(dens_map, (iy[mask_T], ix[mask_T]), Weight[mask_T])
            maps['Thermal'][label] = dens_map
            
        # (B) NonThermal Energy Density
        mask_NT = is_nonthermal & in_bin
        if np.any(mask_NT):
            dens_map = np.zeros((NY, NX))
            np.add.at(dens_map, (iy[mask_NT], ix[mask_NT]), Weight[mask_NT])
            maps['NonThermal'][label] = dens_map
            
    return maps

# =======================================================
# 保存関数
# =======================================================
def save_category_txt(maps_dict, timestep, output_base):
    for particle_type in ['Thermal', 'NonThermal']:
        type_dir = os.path.join(output_base, particle_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for e_min, e_max, bin_label in ENERGY_BINS:
            intensity_map = maps_dict[particle_type][bin_label]
            
            bin_dir = os.path.join(type_dir, bin_label)
            os.makedirs(bin_dir, exist_ok=True)
            
            # ファイル名を energy_density に変更
            txt_name = f'energy_density_{particle_type}_{bin_label}_{timestep}.txt'
            txt_path = os.path.join(bin_dir, txt_name)
            
            header = (f'Electron Kinetic Energy Density Map\n'
                      f'Type: {particle_type}\n'
                      f'Energy Bin: {bin_label} ({e_min}-{e_max} keV)\n'
                      f'Timestep: {timestep}\n'
                      f'Unit: Total keV per grid cell')
            
            np.savetxt(txt_path, intensity_map, header=header, fmt='%.6g')
            print(f"    Saved TXT: {particle_type} - {bin_label}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python calc_energy_density.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    # 出力フォルダ名を変更
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'electron_energy_density')
    
    print(f"--- Electron Energy Density Mapping ---")
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
        result_maps = calculate_energy_density_maps(data)
        
        # 保存
        save_category_txt(result_maps, ts, OUTPUT_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()