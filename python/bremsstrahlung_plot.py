import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =======================================================
# ★ ステップ1: init_param.dat 読み込み
# =======================================================
def load_simulation_parameters(param_filepath):
    params = {}
    print(f"パラメータ読み込み: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "=>" in line:
                    parts = line.split("=>")
                    key_part = parts[0].strip()
                    value_part = parts[1].strip().replace('x', ' ')
                    values = value_part.split()
                    if not values: continue

                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0])
                        params['NY_GRID_POINTS'] = int(values[1])
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])
                        params['DT'] = float(values[1])
                        params['C_LIGHT'] = float(values[2])
                    elif key_part.startswith('Mi, Me'):
                        params['MI'] = float(values[0])
                    elif key_part.startswith('Qi, Qe'):
                        params['QI'] = float(values[0])
                    elif key_part.startswith('Fpe, Fge, Fpi Fgi'):
                        params['FPI'] = float(values[2])
                        params['FGI'] = float(values[3])
    except FileNotFoundError:
        print(f"エラー: ファイルなし: {param_filepath}")
        sys.exit(1)
        
    required = ['NX_GRID_POINTS', 'NY_GRID_POINTS', 'DELX', 'C_LIGHT', 'FPI', 'MI', 'QI', 'FGI']
    if not all(k in params for k in required):
        sys.exit(1)
        
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
    params['DI'] = params['C_LIGHT'] / params['FPI']
    params['B0'] = (params['FGI'] * params['MI'] * params['C_LIGHT']) / params['QI']
    return params

# =======================================================
# ★ ステップ2: ヘルパー関数
# =======================================================
def load_2d_field_data(timestep, component, field_dir, ny, nx):
    path = os.path.join(field_dir, f'data_{timestep}_{component}.txt')
    try:
        d = np.loadtxt(path, delimiter=',')
        if d.shape != (ny, nx): return None
        return d 
    except: return None

def create_coordinates(NX, NY, DELX, DI):
    x_phys = np.linspace(0.0, NX * DELX, NX) 
    y_phys = np.linspace(0.0, NY * DELX, NY)
    return np.meshgrid(x_phys / DI, y_phys / DI)

def load_xray_proxy_map(filepath, ny, nx):
    try:
        d = np.loadtxt(filepath)
        if d.shape != (ny, nx): return None
        print(f"-> マップ読み込み: {os.path.basename(filepath)}")
        return d
    except: return None

# =======================================================
# ★ ステップ3: プロット処理
# =======================================================
def plot_2d_map(timestep, energy_bin_label, params, field_dir, xray_dir, plot_dir):
    NX, NY = params['NX_PHYS'], params['NY_PHYS']
    
    map_path = os.path.join(xray_dir, energy_bin_label, f'xray_proxy_{timestep}_{energy_bin_label}.txt')
    Z_map = load_xray_proxy_map(map_path, NY, NX)
    if Z_map is None: return

    X_norm, Y_norm = create_coordinates(NX, NY, params['DELX'], params['DI'])

    Bx = load_2d_field_data(timestep, 'Bx', field_dir, NY, NX)
    By = load_2d_field_data(timestep, 'By', field_dir, NY, NX)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = 'hot'
    Z_plot = np.where(Z_map > 0, Z_map, np.nan)
    
    from matplotlib.colors import LogNorm
    try:
        min_val = np.nanmin(Z_plot) if np.nanmin(Z_plot) > 0.1 else 0.1
        max_val = np.nanmax(Z_plot)
        norm = LogNorm(vmin=min_val, vmax=max_val)
        cf = ax.contourf(X_norm, Y_norm, Z_plot, 
                         levels=np.logspace(np.log10(min_val), np.log10(max_val), 50), 
                         cmap=cmap, norm=norm, extend='max')
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'Intensity [a.u.]')
    except:
        cf = ax.contourf(X_norm, Y_norm, Z_map, 50, cmap=cmap)
        plt.colorbar(cf, ax=ax).set_label(f'Intensity [a.u.]')

    if Bx is not None and By is not None:
        Bx_n, By_n = Bx / params['B0'], By / params['B0']
        st = max(1, NX // 30)
        ax.streamplot(X_norm[::st, ::st], Y_norm[::st, ::st], 
                      Bx_n[::st, ::st], By_n[::st, ::st], 
                      color='white', linewidth=0.5, density=1.0, minlength=0.1)
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(f'Soft X-ray Intensity ({energy_bin_label}) at TS {timestep}')
    ax.tick_params(direction='in', top=True, right=True)

    out_path = os.path.join(plot_dir, energy_bin_label, f'intensity_{timestep}_{energy_bin_label}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"--- 保存完了: {out_path} ---")

# =======================================================
# ★ ステップ4: メイン実行
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [start] [end] [step]")
        sys.exit(1)
        
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.') 

    PARAM_FILE = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'
    FIELD_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
    XRAY_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_binned_txt')
    PLOT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots_binned')
    
    params = load_simulation_parameters(PARAM_FILE)

    # ★★★ 更新されたプロット対象ビン ★★★
    ENERGY_BINS_TO_PLOT = [
        '001keV_100keV',
        '100keV_200keV',
        '200keV_300keV',
        '300keV_400keV', # 追加
        '400keV_over'
    ]

    steps = []
    if len(sys.argv) == 4:
        steps = range(int(sys.argv[1]), int(sys.argv[2]) + int(sys.argv[3]), int(sys.argv[3]))
    else:
        steps = [int(x) for x in sys.argv[1:]]

    for ts in steps:
        print(f"\n--- TS {ts:06d} ---")
        for label in ENERGY_BINS_TO_PLOT:
            if os.path.exists(os.path.join(XRAY_DIR, label)):
                plot_2d_map(f"{ts:06d}", label, params, FIELD_DIR, XRAY_DIR, PLOT_DIR)