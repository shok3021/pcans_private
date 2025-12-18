import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.integrate import cumtrapz
from multiprocessing import Pool, cpu_count

# =======================================================
# 1. パラメータ読み込み (変更なし)
# =======================================================
def load_simulation_parameters(param_filepath):
    NX = NY = DELX = C_LIGHT = FPI = DT = FGI = VA0 = MI = QI = None
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                s = line.strip()
                if s.startswith('grid size, debye lngth'):
                    parts = s.split(); raw_nx = int(parts[5].replace('x', '')); raw_ny = int(parts[6])
                    NX, NY = raw_nx - 1, raw_ny - 1
                    if len(parts) > 7: DELX = float(parts[7])
                elif s.startswith('dx, dt, c'):
                    parts = s.split(); DT = float(parts[5]); C_LIGHT = float(parts[6])
                    if DELX is None: DELX = float(parts[4])
                elif s.startswith('Mi, Me'): MI = float(s.split()[3])
                elif s.startswith('Qi, Qe'): QI = float(s.split()[3])
                elif s.startswith('Fpe, Fge, Fpi Fgi'):
                    parts = s.split(); FPI, FGI = float(parts[7]), float(parts[8])
                elif s.startswith('Va, Vi, Ve'): VA0 = float(s.split()[7])
    except FileNotFoundError:
        print(f"Error: {param_filepath} not found."); sys.exit(1)
    return NX, NY, DELX, C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# =======================================================
# 2. 設定と定数
# =======================================================
PARAM_FILE_PATH = '/data/shok/dat/init_param.dat'
NX, NY, DELX, C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI, B0 = C_LIGHT / FPI, (FGI * MI * C_LIGHT) / QI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data')
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')

GLOBAL_PLOT_RANGES = {
    'Bx': (-1.0, 1.0), 'By': (-1.5, 1.5), 'Bz': (-0.5, 0.5),
    'Ex': (-0.3, 0.3), 'Ey': (-0.05, 0.05), 'Ez': (-0.1, 0.1),
    'Jx': (-0.5, 0.5), 'Jy': (-0.75, 0.75), 'Jz': (-1.75, 1.75),
    'ne': (0, 200), 'ni': (0, 200), 'Psi': (-5000, 5000), 
    'Te': (0, 1e5), 'Ti': (0, 2e5),
    'Vxe': (-2.0, 2.0), 'Vxi': (-1.0, 1.0), 'Vye': (-5.0, 5.0),
    'Vyi': (-2.0, 2.0), 'Vze': (-8.0, 8.0), 'Vzi': (-10.0, 10.0),
}

# =======================================================
# 3. 高速読み込みと計算
# =======================================================
def load_2d_data(filepath):
    try:
        # np.loadtxtより圧倒的に速いpandas読み込み
        return pd.read_csv(filepath, header=None, delimiter=',', engine='c').values
    except: return np.zeros((NY, NX))

def get_plot_range(Z):
    if Z.size == 0 or np.all(np.isnan(Z)): return -1e-6, 1e-6
    max_abs = np.nanmax(np.abs(Z))
    return (-max_abs, max_abs) if not np.isclose(max_abs, 0.0) else (-1e-6, 1e-6)

# =======================================================
# 4. 描画関数 (pcolormeshで高速化)
# =======================================================
def apply_colorbar(cf, ax, label, tag_key):
    vmin, vmax = cf.get_clim()
    fmt = '%.1e' if (vmax > 0 and abs(vmax) < 0.01) or abs(vmax) > 1000 else '%.2f'
    num_ticks = 6 if tag_key in ['ne', 'ni', 'Te', 'Ti', 'Psi'] else 7
    ticks = np.linspace(vmin, vmax, num_ticks)
    cbar = plt.colorbar(cf, ax=ax, format=fmt, ticks=ticks, pad=0.02)
    cbar.set_label(label)
    return cbar

# =======================================================
# 5. メインプロセス
# =======================================================
def process_timestep(timestep_int):
    ts = f"{timestep_int:06d}"
    omega_t_str = fr"$\Omega_{{ci}}t = {FGI * timestep_int * DT:.2f}$"
    
    # --- データ読み込み ---
    data_map = {
        'Bx': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Bx.txt')),
        'By': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_By.txt')),
        'Bz': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Bz.txt')),
        'Ex': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ex.txt')),
        'Ey': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ey.txt')),
        'Ez': load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ez.txt')),
        'Vxe_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vx.txt')),
        'Vye_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vy.txt')),
        'Vze_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vz.txt')),
        'Vxi_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vx.txt')),
        'Vyi_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vy.txt')),
        'Vzi_r': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vz.txt')),
        'ne': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_density_count.txt')),
        'ni': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_density_count.txt')),
        'Te': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_T.txt')),
        'Ti': load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_T.txt')),
    }

    # --- 規格化 & 計算 ---
    X, Y = np.meshgrid(np.linspace(-NX*DELX/2, NX*DELX/2, NX)/DI, np.linspace(0, NY*DELX, NY)/DI)
    Psi = cumtrapz(data_map['By'], dx=DELX, axis=1, initial=0)
    
    n_proxy = (data_map['ne'] + data_map['ni']) / (2.0 * max(1.0, np.mean((data_map['ne']+data_map['ni'])[data_map['ne']+data_map['ni']>0.1]/2.0)))
    Jx, Jy, Jz = n_proxy*(data_map['Vxi_r']-data_map['Vxe_r']), n_proxy*(data_map['Vyi_r']-data_map['Vye_r']), n_proxy*(data_map['Vzi_r']-data_map['Vze_r'])
    
    # プロット用データセット
    plot_items = [
        ('Bx', data_map['Bx']/B0, r'Magnetic Field ($B_x/B_0$)', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', data_map['By']/B0, r'Magnetic Field ($B_y/B_0$)', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', data_map['Bz']/B0, r'Magnetic Field ($B_z/B_0$)', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Psi', Psi, r'Magnetic Flux $\Psi$', r'$\Psi$', plt.cm.seismic),
        ('Ex', data_map['Ex']/B0, r'Electric Field ($E_x/B_0$)', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', data_map['Ey']/B0, r'Electric Field ($E_y/B_0$)', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', data_map['Ez']/B0, r'Electric Field ($E_z/B_0$)', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', data_map['ne'], 'Electron Density', r'$n_e$', plt.cm.viridis),
        ('ni', data_map['ni'], 'Ion Density', r'$n_i$', plt.cm.viridis),
        ('Te', data_map['Te'], 'Electron Temperature', r'$T_e$ (eV)', plt.cm.plasma),
        ('Ti', data_map['Ti'], 'Ion Temperature', r'$T_i$ (eV)', plt.cm.plasma),
        ('Jx', Jx, 'Current Density ($J_x$)', r'$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, 'Current Density ($J_y$)', r'$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, 'Current Density ($J_z$)', r'$J_z$', plt.cm.RdBu_r),
        ('Vxi', data_map['Vxi_r']/VA0, r'Ion Velocity ($V_{ix}/V_{A0}$)', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        ('Vyi', data_map['Vyi_r']/VA0, r'Ion Velocity ($V_{iy}/V_{A0}$)', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        ('Vzi', data_map['Vzi_r']/VA0, r'Ion Velocity ($V_{iz}/V_{A0}$)', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        ('Vxe', data_map['Vxe_r']/VA0, r'Electron Velocity ($V_{ex}/V_{A0}$)', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        ('Vye', data_map['Vye_r']/VA0, r'Electron Velocity ($V_{ey}/V_{A0}$)', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        ('Vze', data_map['Vze_r']/VA0, r'Electron Velocity ($V_{ez}/V_{A0}$)', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
    ]

    # --- A. 個別プロット保存 ---
    for key, data, title, label, cmap in plot_items:
        sub_dir = os.path.join(OUTPUT_DIR, key)
        os.makedirs(sub_dir, exist_ok=True)
        vmin, vmax = GLOBAL_PLOT_RANGES.get(key, get_plot_range(data))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        apply_colorbar(cf, ax, label, key)
        ax.contour(X, Y, Psi, levels=20, colors='gray', linewidths=0.5, alpha=0.8)
        ax.set_title(f"TS {ts}: {title}"); ax.set_xlabel('$x/d_i$'); ax.set_ylabel('$y/d_i$')
        ax.text(0.98, 0.98, omega_t_str, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.savefig(os.path.join(sub_dir, f'plot_{ts}_{key}.png'), dpi=200)
        plt.close(fig)

    # --- B. 統合パネル保存 (5x4) ---
    fig, axes = plt.subplots(5, 4, figsize=(20, 22), sharex=True, sharey=True)
    fig.suptitle(f"Timestep: {ts}  ({omega_t_str})", fontsize=30, fontweight='bold')
    for i, (key, data, title, label, cmap) in enumerate(plot_items):
        ax = axes.flatten()[i]
        vmin, vmax = GLOBAL_PLOT_RANGES.get(key, get_plot_range(data))
        cf = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title, fontsize=16)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02); cbar.ax.tick_params(labelsize=10)
        ax.contour(X, Y, Psi, levels=12, colors='gray', linewidths=0.5, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.join(OUTPUT_DIR, 'allcombined'), exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'allcombined', f'plot_combined_{ts}.png'), dpi=200)
    plt.close(fig)
    print(f"Done TS: {ts}")

# =======================================================
# 6. 並列実行エントリーポイント
# =======================================================
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif', 'mathtext.fontset': 'cm', 
                         'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18})
    
    if len(sys.argv) != 4:
        print("Usage: python script.py [start] [end] [interval]"); sys.exit(1)
        
    steps = range(int(sys.argv[1]), int(sys.argv[2]) + 1, int(sys.argv[3]))
    print(f"Parallel processing with {cpu_count()} cores...")
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_timestep, steps)