import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Scipy互換性
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# =======================================================
# ★ ユーザー設定
# =======================================================
INPUT_DATA_DIR = os.path.join(os.path.abspath('.'), 'photon_intensity')
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data')
OUTPUT_PLOT_DIR = os.path.join(os.path.abspath('.'), 'photon_plots')
PARAM_FILE_PATH = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat' 

TARGET_BINS = [
    '001keV_100keV',
    '100keV_200keV',
    '200keV_500keV',
    '500keV_1000keV',
    '1000keV_2000keV',
    '2000keV_5000keV',
    '5000keV_10000keV',
    '10000keV_20000keV',
    '20000keV_50000keV',
    '50000keV_over'
]

# ビンごとのVMAX設定
BIN_VMAX_MAP = {
    '001keV_100keV':    1.0e4,
    '100keV_200keV':    1.0e3,
    '200keV_500keV':    1.0e2,
    '500keV_1000keV':   1.0e2,
    '1000keV_2000keV':  1.0e1,
    '2000keV_5000keV':  1.0e1,
    '5000keV_10000keV': 1.0e1,
    '10000keV_20000keV': 1.0e1,
    '20000keV_50000keV': 1.0e1,
    '50000keV_over':     1.0e1
}

# グリッド設定
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# 固定スケール設定
FIXED_SCALE = False 
FIXED_VMIN = 0.0
FIXED_VMAX = 1.0e-2

# =======================================================
# 1. データ読み込みヘルパー
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    init_param.dat からパラメータを読み込む
    参照コード(plot_energy_density.py)のロジックに合わせて修正
    """
    # デフォルト値
    params = {'C_LIGHT': 1.0, 'FPI': 1.0, 'DT': 0.1, 'FGI': 0.1, 'MI': 1.0, 'QI': 1.0}
    
    if not os.path.exists(param_filepath):
        print(f"Warning: Parameter file not found at {param_filepath}. Using defaults.")
        return params

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                s = line.strip()
                parts = s.split()
                if not parts: continue
                
                # 参照コードと同じ判定ロジックを使用
                if s.startswith('dx, dt, c'):
                    params['DT'] = float(parts[5])
                    params['C_LIGHT'] = float(parts[6])
                elif s.startswith('Mi, Me'):
                    params['MI'] = float(parts[3])
                elif s.startswith('Qi, Qe'):
                    params['QI'] = float(parts[3])
                elif s.startswith('Fpe, Fge, Fpi Fgi'):
                    params['FPI'] = float(parts[7])
                    params['FGI'] = float(parts[8])
                    
    except Exception as e:
        print(f"Error loading parameters: {e}")
        
    return params

def load_photon_map(timestep, bin_label):
    target_dir = os.path.join(INPUT_DATA_DIR, bin_label)
    filename = f'photon_intensity_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    
    if not os.path.exists(filepath):
        return None
        
    try:
        data = np.loadtxt(filepath)
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            if data.T.shape == (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
                return data.T
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                return data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_field_data(timestep):
    bx_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    by_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    if not os.path.exists(bx_path): return None, None
    try:
        bx = np.loadtxt(bx_path, delimiter=',')[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        by = np.loadtxt(by_path, delimiter=',')[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        return bx, by
    except: return None, None

# =======================================================
# 2. プロット描画
# =======================================================
def plot_timestep(timestep_str, params):
    c_light = params['C_LIGHT']
    fpi = params['FPI']
    dt = params['DT']
    fgi = params['FGI']
    
    d_i = c_light / fpi if fpi != 0 else 1.0
    
    # 参照コードと同じ計算方法
    omega_t = fgi * float(timestep_str) * dt
    time_label = fr"$\Omega_{{ci}}t = {omega_t:.2f}$"

    x = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS) / d_i
    y = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / d_i
    X, Y = np.meshgrid(x, y)

    Bx, By = load_field_data(timestep_str)
    Psi = None
    if Bx is not None:
        Psi = cumtrapz(By, dx=1.0, axis=1, initial=0)

    print(f"Processing TS: {timestep_str} (Omega_t = {omega_t:.2f})") # 値確認用

    for bin_label in TARGET_BINS:
        data = load_photon_map(timestep_str, bin_label)
        
        if data is None:
            continue
            
        max_val = np.max(data)
        if max_val == 0:
            # print(f"  [Skip] {bin_label}: Zero intensity.")
            continue

        # VMAX設定
        vmin = FIXED_VMIN if FIXED_SCALE else 0.0
        specific_vmax = BIN_VMAX_MAP.get(bin_label)

        if specific_vmax is not None:
            vmax = specific_vmax
        elif FIXED_SCALE:
            vmax = FIXED_VMAX
        else:
            vmax = max_val * 1.05

        fig, ax = plt.subplots(figsize=(7, 10))
        
        levels = np.linspace(vmin, vmax, 100)
        im = ax.contourf(X, Y, data, levels=levels, cmap='inferno')
        
        if Psi is not None:
            ax.contour(X, Y, Psi, levels=25, colors='white', linewidths=0.5, alpha=0.5)
        
        ax.set_title(f"X-ray Intensity: {bin_label}\n{time_label}", fontsize=14)
        ax.set_xlabel(r'$x / d_i$', fontsize=12)
        ax.set_ylabel(r'$y / d_i$', fontsize=12)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Photon Flux (Arb. Unit)', fontsize=12)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        out_folder = os.path.join(OUTPUT_PLOT_DIR, bin_label)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f'photon_{bin_label}_{timestep_str}.png')
        
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  [OK]   {bin_label} -> Saved.")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_photon_maps.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # パラメータ読み込みと確認
    params = load_simulation_parameters(PARAM_FILE_PATH)
    print(f"Loaded Params: DT={params['DT']}, FGI={params['FGI']}, FPI={params['FPI']}")
    
    print(f"--- Plotting Photon Maps ---")
    print(f"Input: {INPUT_DATA_DIR}")
    print(f"Output: {OUTPUT_PLOT_DIR}")
    
    for t_int in range(start, end + step, step):
        ts_str = f"{t_int:06d}"
        plot_timestep(ts_str, params)

if __name__ == "__main__":
    main()