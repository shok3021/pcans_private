import numpy as np
import os
import sys

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

ENERGY_THRESHOLD_COEFF = 5.0

def calculate_moments_and_energy(particle_data):
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    Vx_raw = particle_data[:, 2]
    Vy_raw = particle_data[:, 3]
    Vz_raw = particle_data[:, 4]

    # --- 1. グリッド割り当て ---
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)
    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)
    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)

    ix_m = ix[mask]
    iy_m = iy[mask]
    vx_m = Vx_raw[mask]
    vy_m = Vy_raw[mask]
    vz_m = Vz_raw[mask]

    # --- 2. 基礎モーメント計算 ---
    density = np.zeros((NY, NX))
    vx_sum = np.zeros((NY, NX))
    vy_sum = np.zeros((NY, NX))
    vz_sum = np.zeros((NY, NX))
    v2_sum = np.zeros((NY, NX)) 

    if len(ix_m) > 0:
        np.add.at(density, (iy_m, ix_m), 1)
        np.add.at(vx_sum, (iy_m, ix_m), vx_m)
        np.add.at(vy_sum, (iy_m, ix_m), vy_m)
        np.add.at(vz_sum, (iy_m, ix_m), vz_m)
        v_sq = vx_m**2 + vy_m**2 + vz_m**2
        np.add.at(v2_sum, (iy_m, ix_m), v_sq)

    density_safe = np.where(density > 0, density, 1.0)
    ux = vx_sum / density_safe
    uy = vy_sum / density_safe
    uz = vz_sum / density_safe
    
    mean_v2 = v2_sum / density_safe
    u2 = ux**2 + uy**2 + uz**2
    T_scalar = (mean_v2 - u2) / 3.0 
    
    zero_mask = (density == 0)
    ux[zero_mask] = 0; uy[zero_mask] = 0; uz[zero_mask] = 0; T_scalar[zero_mask] = 0

    # --- 3. エネルギー分離計算 ---
    p_ux = ux[iy_m, ix_m]
    p_uy = uy[iy_m, ix_m]
    p_uz = uz[iy_m, ix_m]
    p_T  = T_scalar[iy_m, ix_m]

    # 粒子の運動エネルギー (流体枠)
    e_p = 0.5 * ((vx_m - p_ux)**2 + (vy_m - p_uy)**2 + (vz_m - p_uz)**2)

    # 閾値判定
    e_thresh = ENERGY_THRESHOLD_COEFF * (1.5 * p_T)
    is_non_thermal = (e_p > e_thresh) & (p_T > 1e-9)
    is_thermal     = ~is_non_thermal

    # --- 【修正点】 エネルギーだけでなく、個数(密度)も分離して集計する ---
    E_thermal_grid = np.zeros((NY, NX))
    E_nonthermal_grid = np.zeros((NY, NX))
    
    N_thermal_grid = np.zeros((NY, NX))     # 追加: 熱的粒子の数
    N_nonthermal_grid = np.zeros((NY, NX))  # 追加: 非熱的粒子の数

    # 熱的成分
    np.add.at(E_thermal_grid, (iy_m[is_thermal], ix_m[is_thermal]), e_p[is_thermal])
    np.add.at(N_thermal_grid, (iy_m[is_thermal], ix_m[is_thermal]), 1.0) # 数をカウント
    
    # 非熱的成分
    np.add.at(E_nonthermal_grid, (iy_m[is_non_thermal], ix_m[is_non_thermal]), e_p[is_non_thermal])
    np.add.at(N_nonthermal_grid, (iy_m[is_non_thermal], ix_m[is_non_thermal]), 1.0) # 数をカウント

    # 4つの配列を返す
    return E_thermal_grid, N_thermal_grid, E_nonthermal_grid, N_nonthermal_grid


def load_text_data(filepath):
    if not os.path.exists(filepath): return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1: data = data.reshape(1, -1)
        return data if data.size > 0 else np.array([])
    except: return None

def save_data(data, species, out_dir, step, label):
    fname = os.path.join(out_dir, f'data_{step}_{species}_{label}.txt')
    np.savetxt(fname, data, fmt='%.6e', delimiter=',')
    print(f"  -> Saved: {fname}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python psd_energy_extractor.py [start] [end] [step]")
        sys.exit(1)

    start, end, step_size = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_energy_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_species = [('e', 'electron')] 

    for current_step in range(start, end + step_size, step_size):
        timestep = f"{current_step:06d}"
        print(f"--- Processing Step: {timestep} ---")

        for suffix, species_label in target_species:
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            pdata = load_text_data(filepath)
            if pdata is None: continue

            # 計算
            Eth, Nth, Enth, Nnth = calculate_moments_and_energy(pdata)

            # 保存 (密度データも保存する)
            save_data(Eth,  species_label, OUTPUT_DIR, timestep, 'Energy_Thermal')
            save_data(Nth,  species_label, OUTPUT_DIR, timestep, 'Density_Thermal') # NEW
            save_data(Enth, species_label, OUTPUT_DIR, timestep, 'Energy_NonThermal')
            save_data(Nnth, species_label, OUTPUT_DIR, timestep, 'Density_NonThermal') # NEW

if __name__ == "__main__":
    main()