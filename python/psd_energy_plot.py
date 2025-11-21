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

# パス設定
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data')        # 電磁場データ
ENERGY_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_energy_data') # 今回抽出したエネルギーデータ
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'energy_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# パラメータファイル (規格化定数取得用)
PARAM_FILE = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'

GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# =======================================================
# 関数群
# =======================================================
def load_sim_params(filepath):
    # 簡易的なパラメータ読み込み
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
    # 磁束線計算
    Psi = cumtrapz(By, dx=1.0, axis=1, initial=0)
    
    if use_log:
        # ゼロや負の値をマスク
        Z_safe = np.where(Z > 1e-8, Z, 1e-8)
        norm = colors.LogNorm(vmin=Z_safe.min(), vmax=Z_safe.max())
        cf = ax.contourf(X, Y, Z_safe, 100, cmap=cmap, norm=norm)
    else:
        # リニアスケール
        vmax = np.max(Z)
        vmin = 0
        cf = ax.contourf(X, Y, Z, 100, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(label)
    
    # 磁力線 (等高線)
    ax.contour(X, Y, Psi, levels=20, colors='white', linewidths=0.5, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel(r'$x/d_i$')
    ax.set_ylabel(r'$y/d_i$')
    ax.set_aspect('equal')

# =======================================================
# メイン処理
# =======================================================
def process_step(timestep, params):
    print(f"Visualizing Step {timestep}...")
    
    # 規格化定数
    B0 = (params['Fgi'] * 1.0 * params['c']) / 1.0 # Qi=1, Mi=1 仮定
    VA = params['Va']
    di = params['c'] / params['Fpi']
    
    # --- データ読み込み ---
    # 電磁場 (背景の磁力線用)
    Bx_raw = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt'))
    By_raw = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt'))
    
    # エネルギーデータ
    Eth_raw  = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Energy_Thermal.txt'))
    Enth_raw = load_data(os.path.join(ENERGY_DATA_DIR, f'data_{timestep}_electron_Energy_NonThermal.txt'))

    # 規格化 (必要に応じて)
    # エネルギー密度は n * v^2 に比例。ここでは VA^2 で割って無次元化するか、そのまま表示する。
    # 今回は比較のため生の値(シミュレーション単位)を表示し、ラベルで注釈を入れる。
    Eth = Eth_raw
    Enth = Enth_raw
    
    # 座標系
    x = np.linspace(0, GLOBAL_NX_PHYS * DELX, GLOBAL_NX_PHYS) / di
    y = np.linspace(0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / di
    X, Y = np.meshgrid(x, y)

    # --- プロット作成 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
    
    # 1. Thermal Energy Density
    plot_energy_map(axes[0], X, Y, Eth, Bx_raw, By_raw, 
                    f"Thermal Electron Energy Density (Step {timestep})", 
                    r"Energy Density (Sim Units $\propto n T_{bulk}$)", 
                    cmap='inferno', use_log=False)

    # 2. Non-Thermal Energy Density
    # 非熱的成分は局所的に強いことが多いので、必要なら use_log=True にしてください
    plot_energy_map(axes[1], X, Y, Enth, Bx_raw, By_raw, 
                    f"Non-Thermal Electron Energy Density (Step {timestep})", 
                    r"Energy Density (Sim Units $\propto n \mathcal{E}_{tail}$)", 
                    cmap='magma', use_log=False) # tailは見えにくいことがあるのでLog推奨

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, f'energy_compare_{timestep}.png')
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