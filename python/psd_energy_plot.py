import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 設定
# =======================================================
try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except: SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data')
ENERGY_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_energy_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'mean_energy_plots') # 出力フォルダ名変更
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_FILE = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

def load_sim_params(filepath):
    # (省略: 前と同じ)
    try:
        vals = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if 'Fpi' in line and 'Fgi' in line:
                    vals['Fpi'] = float(parts[7]); vals['Fgi'] = float(parts[8])
                if 'dx,' in line and 'c' in line:
                    vals['c'] = float(parts[6]); vals['dt'] = float(parts[5])
                if 'Va,' in line:
                    vals['Va'] = float(parts[7])
        return vals
    except:
        return {'c': 1.0, 'Va': 0.1, 'Fpi': 1.0, 'Fgi': 1.0, 'dt': 1.0}

def load_data(path, shape=(GLOBAL_NY_PHYS, GLOBAL_NX_PHYS)):
    try:
        d = np.loadtxt(path, delimiter=',')
        return d if d.shape == shape else np.zeros(shape)
    except: return np.zeros(shape)

def plot_energy_map(ax, X, Y, Z, Bx, By, title, label, cmap='plasma', use_log=False):
    Psi = cumtrapz(By, dx=1.0, axis=1, initial=0)
    
    if use_log:
        # 平均エネルギーは0になりにくいが念のため
        Z_safe = np.where(Z > 1e-8, Z, 1e-8)
        norm = colors.LogNorm(vmin=Z_safe.min(), vmax=Z_safe.max())
        cf = ax.contourf(X, Y, Z_safe, 100, cmap=cmap, norm=norm)
    else:
        # 見やすくするために上限をクリップするのもあり
        # vmax = np.percentile(Z, 99) 
        cf = ax.contourf(X, Y, Z, 100, cmap=cmap)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(label)
    ax.contour(X, Y, Psi, levels=20, colors='white', linewidths=0.5, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r'$x/d_i$')
    ax.set_ylabel(r'$y/d_i$')
    ax.set_aspect('equal')

def process_step(timestep, params):
    print(f"Visualizing Mean Energy Step {timestep}...")
    
    B0 = (params['Fgi'] * 1.0 * params['c']) / 1.0
    di = params['c'] / params['Fpi']
    
    # --- データ読み込み ---
    Bx_raw = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt'))
    By_raw = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt'))
    
    # エネルギー密度と粒子数密度を読み込む
    Eth_dens  = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Energy_Thermal.txt'))
    Nth_dens  = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Density_Thermal.txt'))
    
    Enth_dens = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Energy_NonThermal.txt'))
    Nnth_dens = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Density_NonThermal.txt'))

    # --- 【修正点】 平均エネルギー (Mean Energy) の計算 ---
    # Mean Energy = Total Energy / Number of Particles
    # ゼロ除算を防ぐ
    
    # 1. Thermal Mean Energy (≒ Temperature)
    mean_Eth = np.zeros_like(Eth_dens)
    mask_th = Nth_dens > 0
    mean_Eth[mask_th] = Eth_dens[mask_th] / Nth_dens[mask_th]
    
    # 2. Non-Thermal Mean Energy (tailの平均エネルギー)
    mean_Enth = np.zeros_like(Enth_dens)
    mask_nth = Nnth_dens > 0
    mean_Enth[mask_nth] = Enth_dens[mask_nth] / Nnth_dens[mask_nth]
    
    # 座標系
    x = np.linspace(0, GLOBAL_NX_PHYS * DELX, GLOBAL_NX_PHYS) / di
    y = np.linspace(0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / di
    X, Y = np.meshgrid(x, y)

    # --- プロット作成 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
    
    # Thermal Mean Energy
    plot_energy_map(axes[0], X, Y, mean_Eth, Bx_raw, By_raw, 
                    f"Mean Thermal Energy (Step {timestep})", 
                    r"Mean Energy per Particle ($T_{bulk}$ equivalent)", 
                    cmap='inferno', use_log=False)

    # Non-Thermal Mean Energy
    # ここでセパトリクスが明るくなれば、粒子個々のエネルギーが高いことを意味する
    plot_energy_map(axes[1], X, Y, mean_Enth, Bx_raw, By_raw, 
                    f"Mean Non-Thermal Energy (Step {timestep})", 
                    r"Mean Energy per Non-Thermal Particle", 
                    cmap='magma', use_log=False) 

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, f'mean_energy_{timestep}.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"-> Saved: {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python visual_energy.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    params = load_sim_params(PARAM_FILE)
    
    for t in range(start, end + step, step):
        process_step(f"{t:06d}", params)