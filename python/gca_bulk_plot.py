import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys

# =======================================================
# 設定
# =======================================================
BASE_DIR = os.path.abspath('/Users/shohgookazaki/Documents/GitHub/pcans')
INPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_data') # Extractorの出力先
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理定数 (軸表示用)
C_LIGHT = 100.0 # ダミー値(Extractorで吸収されているのでここではラベル計算のみに使用)
FPI = 100.0
DI = C_LIGHT / FPI
DELX = 1.0
BINNING_FACTOR = 4 # Extractorと合わせる
DX_PHYS_COARSE = (DELX / DI) * BINNING_FACTOR

# プロット設定 (提示コード準拠)
FIXED_LIMIT = 0.5   # レンジの最大最小
LIN_THRESH = 0.01   # 線形スケールの閾値

def load_data(timestep, name):
    path = os.path.join(INPUT_DIR, f'{name}_{timestep}.txt')
    try:
        return np.loadtxt(path)
    except:
        return None

def plot_heating(timestep):
    print(f"Plotting Timestep: {timestep}")
    
    # データ読み込み
    total = load_data(timestep, 'heat_total')
    e_par = load_data(timestep, 'heat_epar')
    curv = load_data(timestep, 'heat_curv')
    gradb = load_data(timestep, 'heat_gradb')
    Bx = load_data(timestep, 'field_Bx')
    By = load_data(timestep, 'field_By')
    
    if total is None or Bx is None:
        print("  Data not found.")
        return

    ny, nx = total.shape
    
    # 座標作成
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    # プロット準備
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    
    plot_configs = [
        (total, r'Total Heating $dU/dt$', axs[0]),
        (e_par, r'Parallel E-Field Work: $E_\parallel J_\parallel$', axs[1]),
        (curv, r'Curvature Drift Heating ($v_E \cdot \kappa$)', axs[2]),
        (gradb, r'Grad-B Drift Heating ($v_E \cdot \nabla B$)', axs[3])
    ]
    
    # SymLogNorm用のレベル設定
    fixed_levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 200)

    for data, title, ax in plot_configs:
        ax.set_facecolor('#d3d3d3') # 背景グレー
        
        # SymLogNorm
        norm = colors.SymLogNorm(linthresh=LIN_THRESH, linscale=1.0, 
                                 vmin=-FIXED_LIMIT, vmax=FIXED_LIMIT, base=10)
        
        # コンタープロット
        cf = ax.contourf(X, Y, data, levels=fixed_levels, cmap='gist_ncar', norm=norm, extend='both')
        
        # 流線 (Streamplot)
        # データが大きいので間引き
        stride = 2
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x/d_i$', fontsize=12)
        ax.set_ylabel('$y/d_i$', fontsize=12)
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95)
        cbar.set_label('Heating Rate', fontsize=10)

    fig.suptitle(rf"Electron Heating (Fixed Range $\pm${FIXED_LIMIT}) - TS {timestep}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, f'plot_heating_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visual_heating_accurate.py [start] [end] [step]")
        # デフォルト動作
        plot_heating("015000")
    else:
        s, e, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+1, step):
            plot_heating(f"{ts:06d}")