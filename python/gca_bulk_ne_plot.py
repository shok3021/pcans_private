import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys

# =======================================================
# 設定
# =======================================================
BASE_DIR = os.path.abspath('/home/shok/pcans')

# ★ パス設定変更 (指定通り) ★
INPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_ne_data')  # 読み込み元
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_ne_plot') # 出力先
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理定数・グリッド設定
C_LIGHT = 100.0
FPI = 100.0
DI = C_LIGHT / FPI
DELX = 1.0
BINNING_FACTOR = 4
DX_PHYS_COARSE = (DELX / DI) * BINNING_FACTOR

# プロットレンジ (密度で割った後の値: eV/s 相当)
# 加速の強さに応じて調整してください
FIXED_LIMIT = 0.05 
LIN_THRESH = 0.001

def load_data(timestep, variable_name):
    # ファイル名: variable_timestep.txt
    filename = f'{variable_name}_{timestep}.txt'
    path = os.path.join(INPUT_DIR, filename)
    try:
        return np.loadtxt(path)
    except Exception as e:
        # print(f"Warning: Could not load {path}")
        return None

def plot_normalized_heating(timestep):
    print(f"Processing Plot: {timestep}")
    
    # データの読み込み
    total = load_data(timestep, 'total')
    e_par = load_data(timestep, 'epar')
    curv = load_data(timestep, 'curv')
    gradb = load_data(timestep, 'gradb')
    Bx = load_data(timestep, 'Bx')
    By = load_data(timestep, 'By')
    ne = load_data(timestep, 'ne')
    
    if total is None or ne is None or Bx is None:
        print(f"  Skipping {timestep}: Required data not found in {INPUT_DIR}")
        return

    # ★ 密度による正規化 (Heating Rate per Particle) ★
    # 密度が極端に低い場所(バックグラウンド)のノイズ除去
    ne_safe = ne.copy()
    ne_safe[ne_safe < 0.1] = np.nan 
    
    total_norm = total / ne_safe
    epar_norm = e_par / ne_safe
    curv_norm = curv / ne_safe
    gradb_norm = gradb / ne_safe

    # 座標作成
    ny, nx = total.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    # プロット作成 (2x2パネル)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    
    plot_configs = [
        (total_norm, r'Total Heating / Particle ($dU/dt / n_e$)', axs[0]),
        (epar_norm, r'Parallel Accel / Particle ($E_\parallel J_\parallel / n_e$)', axs[1]),
        (curv_norm, r'Curvature Drift Heating / Particle', axs[2]),
        (gradb_norm, r'Grad-B Drift Heating / Particle', axs[3])
    ]
    
    # カラーバー設定 (固定レンジ)
    fixed_levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 200)

    for data, title, ax in plot_configs:
        ax.set_facecolor('#d3d3d3') # NaN部分をグレーに
        
        # SymLogNorm: ゼロ付近を線形、大きな値を対数表示
        norm = colors.SymLogNorm(linthresh=LIN_THRESH, linscale=1.0, 
                                 vmin=-FIXED_LIMIT, vmax=FIXED_LIMIT, base=10)
        
        # コンター描画
        cf = ax.contourf(X, Y, data, levels=fixed_levels, cmap='gist_ncar', norm=norm, extend='both')
        
        # 磁力線 (間引き表示)
        stride = 2
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x/d_i$', fontsize=12)
        ax.set_ylabel('$y/d_i$', fontsize=12)
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95)
        cbar.set_label('Heating Rate [Arbitrary Units]', fontsize=10)

    fig.suptitle(rf"GCA Heating Rate per Particle (Normalized by $n_e$) - TS {timestep}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # ★ 保存 (指定ディレクトリ) ★
    save_filename = f'gca_norm_plot_{timestep}.png'
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visual_gca_normalized.py [start] [end] [step]")
    else:
        s, e, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+1, step):
            plot_normalized_heating(f"{ts:06d}")