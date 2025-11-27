import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys

# =======================================================
# 設定
# =======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 計算スクリプトの出力先と合わせる
INPUT_DIR = os.path.join(SCRIPT_DIR, 'particle_gca_results') 
OUTPUT_PLOT_DIR = os.path.join(SCRIPT_DIR, 'particle_gca_plots')
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

DELX = 1.0 # シミュレーションのグリッドサイズ

def plot_gca_map(timestep):
    print(f"Plotting Timestep: {timestep}")
    
    # ファイル読み込み
    try:
        t1 = np.loadtxt(os.path.join(INPUT_DIR, f'gca_term1_par_{timestep}.txt'))
        t2 = np.loadtxt(os.path.join(INPUT_DIR, f'gca_term2_betatron_{timestep}.txt'))
        t3 = np.loadtxt(os.path.join(INPUT_DIR, f'gca_term3_fermi_{timestep}.txt'))
        total = np.loadtxt(os.path.join(INPUT_DIR, f'gca_total_{timestep}.txt'))
        Bx = np.loadtxt(os.path.join(INPUT_DIR, f'field_Bx_{timestep}.txt'))
        By = np.loadtxt(os.path.join(INPUT_DIR, f'field_By_{timestep}.txt'))
    except FileNotFoundError:
        print(f"  Data not found for {timestep}. Run calculation script first.")
        return

    ny, nx = t1.shape
    X, Y = np.meshgrid(np.arange(nx)*DELX, np.arange(ny)*DELX)

    # プロット設定
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()

    titles = [
        r'Total Particle Energization ($d\epsilon/dt$)',
        r'Parallel Acceleration ($q E_\parallel v_\parallel$)',
        r'Fermi Acceleration ($\gamma m v_\parallel^2 \mathbf{u}_E \cdot \mathbf{\kappa}$)',
        r'Betatron Acceleration ($\mu/\gamma \mathbf{u}_E \cdot \nabla B$)'
    ]
    data_list = [total, t1, t3, t2] # 順番: Total, Par, Fermi, Betatron

    # カラーバー設定 (SymLogNorm: ゼロ付近は線形、大きい値は対数)
    # 値の範囲はデータを見て調整してください
    limit = max(np.abs(np.percentile(total, 99)), 1e-5) 
    norm = colors.SymLogNorm(linthresh=limit*0.02, linscale=1.0, 
                             vmin=-limit, vmax=limit, base=10)
    FIXED_LIMIT = 20.0  

    fixed_levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 1000)
    for ax, data, title in zip(axs, data_list, titles):
        ax.set_facecolor('#d3d3d3') # データがない場所はグレー
        
        # コンタープロット
        cf = ax.contourf(X, Y, data, levels=fixed_levels, cmap='gist_ncar', norm=norm)
        
        # 磁力線 (間引きして表示)
        stride = 20
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.5, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        
        # カラーバー
        cbar = plt.colorbar(cf, ax=ax, shrink=0.9)
        cbar.set_label('Energy Gain Density', fontsize=10)

    fig.suptitle(f"Particle-based Acceleration Analysis (TS: {timestep})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_PLOT_DIR, f'gca_plot_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # テスト用
        print("Usage: python vis_particle_gca.py [timestep_start] [timestep_end] [step]")
        plot_gca_map("015000")
    else:
        s = int(sys.argv[1])
        e = int(sys.argv[2])
        st = int(sys.argv[3])
        for ts in range(s, e+st, st):
            plot_gca_map(f"{ts:06d}")