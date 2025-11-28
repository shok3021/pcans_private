import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys

# =======================================================
# 設定
# =======================================================
BASE_DIR = os.path.abspath('/Users/shohgookazaki/Documents/GitHub/pcans')

# 入出力ディレクトリ
# Extractorの出力先 (gca_bulk_ne_data) から読み込みます
INPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_data') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_FILE_PATH = os.path.join(BASE_DIR, 'em2d_mpi/md_mrx/dat/init_param.dat')

# プロット設定
# バルク値(Power Density)用のレンジ。値が大きすぎ/小さすぎたらここを調整してください。
FIXED_LIMIT = 0.5   
LIN_THRESH = 0.005  
BINNING_FACTOR = 4  
DELX = 1.0          

# =======================================================
# ヘルパー関数
# =======================================================
def load_simulation_parameters(param_filepath):
    C_LIGHT, FPI = None, None
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if 'dx,' in line and 'c' in line:
                    C_LIGHT = float(parts[6])
                elif 'Fpe,' in line and 'Fpi' in line:
                    FPI = float(parts[7])
    except Exception as e:
        print(f"Warning: Failed to load params: {e}")
    return C_LIGHT, FPI

# =======================================================
# メイン処理
# =======================================================
C_LIGHT_VAL, FPI_VAL = load_simulation_parameters(PARAM_FILE_PATH)

if C_LIGHT_VAL is not None and FPI_VAL is not None:
    DI = C_LIGHT_VAL / FPI_VAL
    print(f"Loaded Parameters: c={C_LIGHT_VAL}, wpi={FPI_VAL} -> d_i={DI:.4f}")
else:
    print("Warning: Using default parameters (d_i = 1.0)")
    DI = 1.0 

DX_PHYS_COARSE = (DELX / DI) * BINNING_FACTOR

def load_data(timestep, name):
    # Extractorの出力名 (total_xxxxx.txt 等) に合わせる
    path = os.path.join(INPUT_DIR, f'{name}_{timestep}.txt')
    try:
        return np.loadtxt(path)
    except:
        return None

def plot_heating_bwr(timestep):
    print(f"Plotting Timestep (Bulk, BWR): {timestep}")
    
    # データ読み込み
    total = load_data(timestep, 'total')
    e_par = load_data(timestep, 'epar')
    curv = load_data(timestep, 'curv')
    gradb = load_data(timestep, 'gradb')
    Bx = load_data(timestep, 'Bx')
    By = load_data(timestep, 'By')
    
    if total is None or Bx is None:
        print(f"  Data not found for step {timestep}")
        return

    ny, nx = total.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    
    plot_configs = [
        (total, r'Total Energy Gain Density ($\mathbf{J} \cdot \mathbf{E}$)', axs[0]),
        (e_par, r'Parallel Work Density ($E_\parallel J_\parallel$)', axs[1]),
        (curv, r'Curvature Drift Power Density', axs[2]),
        (gradb, r'Grad-B Drift Power Density', axs[3])
    ]
    
    # ★ 指定の levels 作成
    levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 200)

    for data, title, ax in plot_configs:
        ax.set_facecolor('#d3d3d3')
        
        # ★ 指定の norm 作成
        norm = colors.SymLogNorm(linthresh=LIN_THRESH, linscale=1.0, 
                                 vmin=-FIXED_LIMIT, vmax=FIXED_LIMIT, base=10)
        
        # ★★★ 指定通りのプロットコマンド (cmap='bwr') ★★★
        cf = ax.contourf(X, Y, data, levels=levels, cmap='bwr', norm=norm)
        
        stride = 2
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x/d_i$', fontsize=12)
        ax.set_ylabel('$y/d_i$', fontsize=12)
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95)
        cbar.set_label('Power Density [Arb. Units]', fontsize=10)

    fig.suptitle(rf"Bulk Electron Energy Gain Density (Not Normalized) - TS {timestep}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, f'gca_bulk_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visual_heating_bulk_bwr.py [start] [end] [step]")
    else:
        s, e, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+1, step):
            plot_heating_bwr(f"{ts:06d}")