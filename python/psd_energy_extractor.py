import numpy as np
import os
import sys

# =======================================================
# 設定 (Fortran const モジュール準拠)
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

# ★★★ エネルギー分離の閾値係数 ★★★
# 粒子のエネルギーが局所熱エネルギー(1.5 * T)の何倍を超えたら「非熱的」とみなすか
# 例: 3.0 〜 5.0 が一般的
ENERGY_THRESHOLD_COEFF = 5.0

def calculate_moments_and_energy(particle_data):
    """
    粒子データから密度、流体速度、温度を計算し、
    さらにそれを用いて熱的/非熱的エネルギー密度を分離計算する。
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

    # --- 2. 基礎モーメント計算 (密度、流体速度、温度) ---
    density = np.zeros((NY, NX))
    vx_sum = np.zeros((NY, NX))
    vy_sum = np.zeros((NY, NX))
    vz_sum = np.zeros((NY, NX))
    v2_sum = np.zeros((NY, NX)) # 全速度の二乗和

    if len(ix_m) > 0:
        np.add.at(density, (iy_m, ix_m), 1)
        np.add.at(vx_sum, (iy_m, ix_m), vx_m)
        np.add.at(vy_sum, (iy_m, ix_m), vy_m)
        np.add.at(vz_sum, (iy_m, ix_m), vz_m)
        # 温度計算用 (v^2)
        v_sq = vx_m**2 + vy_m**2 + vz_m**2
        np.add.at(v2_sum, (iy_m, ix_m), v_sq)

    # 流体量計算
    density_safe = np.where(density > 0, density, 1.0)
    ux = vx_sum / density_safe
    uy = vy_sum / density_safe
    uz = vz_sum / density_safe
    
    mean_v2 = v2_sum / density_safe
    u2 = ux**2 + uy**2 + uz**2
    
    # 温度 T (速度分散) = <v^2> - <u>^2
    # ここでの T は v^2 の次元 (エネルギー/質量)
    T_scalar = (mean_v2 - u2) / 3.0 # 3次元等方と仮定した1自由度あたりの温度
    
    # 密度0の場所をクリア
    zero_mask = (density == 0)
    ux[zero_mask] = 0; uy[zero_mask] = 0; uz[zero_mask] = 0; T_scalar[zero_mask] = 0

    # --- 3. エネルギー分離計算 (Thermal vs Non-Thermal) ---
    # ベクトル化のために、各粒子に対応する流体速度と温度を配列化
    p_ux = ux[iy_m, ix_m]
    p_uy = uy[iy_m, ix_m]
    p_uz = uz[iy_m, ix_m]
    p_T  = T_scalar[iy_m, ix_m]

    # 粒子の流体枠での運動エネルギー (質量=1として計算 => v^2/2)
    # e_p = 0.5 * (v - u)^2
    e_p = 0.5 * ((vx_m - p_ux)**2 + (vy_m - p_uy)**2 + (vz_m - p_uz)**2)

    # 閾値エネルギー (局所温度に基づく)
    # 熱エネルギーの平均は (3/2)T。その ENERGY_THRESHOLD_COEFF 倍を閾値とする
    e_thresh = ENERGY_THRESHOLD_COEFF * (1.5 * p_T)

    # 判定 (温度が極端に低い、または密度が低い場所はすべてThermal扱いにするなどの安全策)
    is_non_thermal = (e_p > e_thresh) & (p_T > 1e-9)
    is_thermal     = ~is_non_thermal

    # 集計用配列
    E_thermal_grid = np.zeros((NY, NX))
    E_nonthermal_grid = np.zeros((NY, NX))

    # 熱的エネルギー密度積算
    np.add.at(E_thermal_grid, (iy_m[is_thermal], ix_m[is_thermal]), e_p[is_thermal])
    
    # 非熱的エネルギー密度積算
    np.add.at(E_nonthermal_grid, (iy_m[is_non_thermal], ix_m[is_non_thermal]), e_p[is_non_thermal])

    return density, E_thermal_grid, E_nonthermal_grid


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
    
    # 入力ディレクトリ設定
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_energy_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 電子のみを対象とする場合 ('e', 'electron')
    target_species = [('e', 'electron')] 

    for current_step in range(start, end + step_size, step_size):
        timestep = f"{current_step:06d}"
        print(f"--- Processing Step: {timestep} ---")

        for suffix, species_label in target_species:
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            pdata = load_text_data(filepath)
            if pdata is None: continue

            # 計算実行
            dens, Eth, Enth = calculate_moments_and_energy(pdata)

            # 保存
            save_data(Eth, species_label, OUTPUT_DIR, timestep, 'Energy_Thermal')
            save_data(Enth, species_label, OUTPUT_DIR, timestep, 'Energy_NonThermal')
            
            # 割合なども計算したければここで計算可能
            # ratio = Enth / (Eth + Enth + 1e-20)
            # save_data(ratio, species_label, OUTPUT_DIR, timestep, 'Energy_Ratio')

if __name__ == "__main__":
    main()