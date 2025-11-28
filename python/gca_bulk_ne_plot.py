import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys

# =======================================================
# 設定
# =======================================================
BASE_DIR = os.path.abspath('/Users/shohgookazaki/Documents/GitHub/pcans')

INPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_ne_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_FILE_PATH = os.path.join(BASE_DIR, 'em2d_mpi/md_mrx/dat/init_param.dat')

# グリッド設定
DELX = 1.0
BINNING_FACTOR = 4

# フォントサイズ設定 (ポスター用)
FS_TITLE = 24       
FS_LABEL = 20       
FS_TICK = 18        
FS_CBAR_LABEL = 18  
FS_SUPTITLE = 28    

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

def load_data(timestep, variable_name):
    filename = f'{variable_name}_{timestep}.txt'
    path = os.path.join(INPUT_DIR, filename)
    try:
        return np.loadtxt(path)
    except:
        return None

def plot_poster_energization(timestep):
    print(f"Processing Poster Plot: {timestep}")
    
    total = load_data(timestep, 'total')
    e_par = load_data(timestep, 'epar')
    curv = load_data(timestep, 'curv')
    gradb = load_data(timestep, 'gradb')
    Bx = load_data(timestep, 'Bx')
    By = load_data(timestep, 'By')
    ne = load_data(timestep, 'ne')
    
    if total is None or ne is None or Bx is None:
        print(f"  Skipping {timestep}: Required data not found.")
        return

    # 密度正規化
    ne_safe = ne.copy()
    ne_safe[ne_safe < 1e-3] = np.nan 
    
    total_norm = total / ne_safe
    epar_norm = e_par / ne_safe
    curv_norm = curv / ne_safe
    gradb_norm = gradb / ne_safe

    ny, nx = total.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    axs = axes.flatten()
    
    # ★★★ ここで各パネルのレンジを指定 (Data, Title, Axis, Limit) ★★★
    plot_configs = [
        (total_norm, r'Total Energy Gain / Particle', axs[0], 0.005),
        (epar_norm, r'Parallel Energy Gain ($E_\parallel J_\parallel / n_e$)', axs[1], 0.005),
        (curv_norm, r'Curvature Drift Energy Gain / Particle', axs[2], 0.005),
        (gradb_norm, r'Grad-B Drift Energy Gain / Particle', axs[3], 0.005)
    ]
    
    # ループ内で limit を受け取り、動的に設定を作成
    for data, title, ax, limit in plot_configs:
        ax.set_facecolor('#d3d3d3')
        
        # レンジに応じた ticks の自動生成 [-Limit, -Limit/2, 0, Limit/2, Limit]
        target_ticks = [-limit, -limit/2, 0, limit/2, limit]
        
        # レンジに応じた levels 生成
        levels = np.linspace(-limit, limit, 200)
        
        # レンジに応じた SymLogNorm の閾値設定 (レンジの2%程度にする)
        # これをしないと、レンジが小さいGradBなどで色がベタ塗りになってしまいます
        this_linthresh = limit * 0.02
        
        norm = colors.SymLogNorm(linthresh=this_linthresh, linscale=1.0, 
                                 vmin=-limit, vmax=limit, base=10)
        
        # プロット
        cf = ax.contourf(X, Y, data, levels=levels, cmap='bwr', norm=norm)
        
        # 磁力線
        stride = 2
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.8, arrowsize=1.2, density=0.8, arrowstyle='-')
        
        # タイトル・軸ラベル
        ax.set_title(title, fontsize=FS_TITLE, pad=10)
        ax.set_xlabel('$x/d_i$', fontsize=FS_LABEL)
        ax.set_ylabel('$y/d_i$', fontsize=FS_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
        
        # カラーバー (指数表記にするために format='%.1e' を推奨しますが、自動でも可)
        # 値が小さいので指数表記 (scientific notation) を強制します
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95, ticks=target_ticks, format='%.1e')
        cbar.set_label('Energy Gain Rate [Arb. Units]', fontsize=FS_CBAR_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK) 

    fig.suptitle(rf"Electron Energy Gain Rate per Particle (TS {timestep})", fontsize=FS_SUPTITLE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_filename = f'gca_bulk_ne_{timestep}.png'
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved Poster Plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visual_gca_poster_multi_range.py [start] [end] [step]")
    else:
        s, e, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+1, step):
            plot_poster_energization(f"{ts:06d}")