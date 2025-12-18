import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.integrate import cumtrapz
from multiprocessing import Pool, cpu_count

# =======================================================
# 1. パラメータ読み込み
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
# 3. 高速読み込みと補助関数
# =======================================================
def load_2d_data(filepath):
    try:
        return pd.read_csv(filepath, header=None, delimiter=',', engine='c').values
    except: return np.zeros((NY, NX))

def get_plot_range(Z):
    if Z.size == 0 or np.all(np.isnan(Z)): return -1e-6, 1e-6
    max_abs = np.nanmax(np.abs(Z))
    return (-max_abs, max_abs) if not np.isclose(max_abs, 0.0) else (-1e-6, 1e-6)

def apply_colorbar(cf, ax, label, tag_key, is_combined=False):
    vmin, vmax = cf.get_clim()
    fmt = '%.1e' if (vmax > 0 and abs(vmax) < 0.01) or abs(vmax) > 1000 else '%.2f'
    num_ticks = 5 if is_combined else (6 if tag_key in ['ne', 'ni', 'Te', 'Ti', 'Psi'] else 7)
    ticks = np.linspace(vmin, vmax, num_ticks)
    
    if is_combined:
        cbar = plt.colorbar(cf, ax=ax, format=fmt, ticks=ticks, shrink=0.9, aspect=30, pad=0.02)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar = plt.colorbar(cf, ax=ax, format=fmt, ticks=ticks, pad=0.02)
        cbar.set_label(label)
    return cbar

# =======================================================
# 4. メインプロセス
# =======================================================
def process_timestep(timestep_int):
    ts = f"{timestep_int:06d}"
    omega_t_val = FGI * timestep_int * DT
    omega_t_str = fr"$\Omega_{{ci}}t = {omega_t_val:.2f}$"
    
    # --- データ読み込み ---
    # 電磁場
    Bx_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Bx.txt'))
    By_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_By.txt'))
    Bz_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Bz.txt'))
    Ex_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ex.txt'))
    Ey_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ey.txt'))
    Ez_r = load_2d_data(os.path.join(FIELD_DATA_DIR, f'data_{ts}_Ez.txt'))
    # モーメント
    Vxe_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vx.txt'))
    Vye_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vy.txt'))
    Vze_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Vz.txt'))
    Vxi_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vx.txt'))
    Vyi_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vy.txt'))
    Vzi_r = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_Vz.txt'))
    ne = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_density_count.txt'))
    ni = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_density_count.txt'))
    Te = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_T.txt'))
    Ti = load_2d_data(os.path.join(MOMENT_DATA_DIR, f'data_{ts}_ion_T.txt'))

    # --- 規格化 & 計算 ---
    Bx, By, Bz, Ex, Ey, Ez = Bx_r/B0, By_r/B0, Bz_r/B0, Ex_r/B0, Ey_r/B0, Ez_r/B0
    Vxe, Vye, Vze, Vxi, Vyi, Vzi = Vxe_r/VA0, Vye_r/VA0, Vze_r/VA0, Vxi_r/VA0, Vyi_r/VA0, Vzi_r/VA0
    Psi = cumtrapz(By_r, dx=DELX, axis=1, initial=0)
    
    n_proxy = (ne + ni) / (2.0 * max(1.0, np.mean((ne+ni)[(ne+ni)>0.1]/2.0)))
    Jx, Jy, Jz = n_proxy*(Vxi_r-Vxe_r), n_proxy*(Vyi_r-Vye_r), n_proxy*(Vzi_r-Vze_r)
    
    Ez_non_ideal_e = (Ez_r + (Vxe_r * By_r - Vye_r * Bx_r)) / B0
    Ez_non_ideal_i = (Ez_r + (Vxi_r * By_r - Vyi_r * Bx_r)) / B0

    X, Y = np.meshgrid(np.linspace(-NX*DELX/2, NX*DELX/2, NX)/DI, np.linspace(0, NY*DELX, NY)/DI)

    # --- プロットアイテム定義 (統合パネルの順番通り) ---
    all_items = [
        ('Bx', Bx, r'Magnetic Field ($B_x/B_0$)', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, r'Magnetic Field ($B_y/B_0$)', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, r'Magnetic Field ($B_z/B_0$)', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Psi', Psi, r'Magnetic Flux $\Psi$', r'$\Psi$', plt.cm.seismic),
        ('Ex', Ex, r'Electric Field ($E_x/B_0$)', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, r'Electric Field ($E_y/B_0$)', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, r'Electric Field ($E_z/B_0$)', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', ne, 'Electron Density', r'$n_e$ (Counts)', plt.cm.viridis),
        ('Jx', Jx, 'Current Density ($J_x$)', r'$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, 'Current Density ($J_y$)', r'$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, 'Current Density ($J_z$)', r'$J_z$', plt.cm.RdBu_r),
        ('ni', ni, 'Ion Density', r'$n_i$ (Counts)', plt.cm.viridis),
        ('Vxi', Vxi, r'Ion Velocity ($V_{ix}/V_{A0}$)', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        ('Vyi', Vyi, r'Ion Velocity ($V_{iy}/V_{A0}$)', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        ('Vzi', Vzi, r'Ion Velocity ($V_{iz}/V_{A0}$)', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        ('Ti', Ti, 'Ion Temperature', r'$T_i$ (eV)', plt.cm.plasma),
        ('Vxe', Vxe, r'Electron Velocity ($V_{ex}/V_{A0}$)', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        ('Vye', Vye, r'Electron Velocity ($V_{ey}/V_{A0}$)', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        ('Vze', Vze, r'Electron Velocity ($V_{ez}/V_{A0}$)', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
        ('Te', Te, 'Electron Temperature', r'$T_e$ (eV)', plt.cm.plasma),
    ]
    
    # 個別プロット専用（非理想電場など）
    individual_only = [
        ('Ez_non_ideal_e', Ez_non_ideal_e, r'Non-Ideal $E_z$ (Electron)', r'$(E_z + (\mathbf{V}_e \times \mathbf{B})_z)/B_0$', plt.cm.jet),
        ('Ez_non_ideal_i', Ez_non_ideal_i, r'Non-Ideal $E_z$ (Ion)', r'$(E_z + (\mathbf{V}_i \times \mathbf{B})_z)/B_0$', plt.cm.jet),
    ]

    # --- A. 個別プロットの生成 ---
    for key, data, title, label, cmap in all_items + individual_only:
        sub_dir = os.path.join(OUTPUT_DIR, key.replace('/', '_'))
        os.makedirs(sub_dir, exist_ok=True)
        vmin, vmax = GLOBAL_PLOT_RANGES.get(key, get_plot_range(data))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        apply_colorbar(cf, ax, label, key)
        ax.contour(X, Y, Psi, levels=25, colors='gray', linewidths=0.5, alpha=0.8)
        
        ax.set_title(f"Timestep {ts}: {title}"); ax.set_xlabel('$x/d_i$'); ax.set_ylabel('$y/d_i$')
        ax.tick_params(direction='in', top=True, right=True)
        # サブタイトル（テキストボックス）
        ax.text(0.98, 0.98, omega_t_str, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7))
        
        plt.savefig(os.path.join(sub_dir, f'plot_{ts}_{key}.png'), dpi=200)
        plt.close(fig)

    # --- B. 統合パネルの生成 (5x4) ---
    # パネル用のタイトル（(a) Bx/B0 形式）をマッピング
    combined_titles = [
        r'(a) $B_x/B_0$', r'(b) $B_y/B_0$', r'(c) $B_z/B_0$', r'(d) $\Psi$',
        r'(e) $E_x/B_0$', r'(f) $E_y/B_0$', r'(g) $E_z/B_0$', r'(h) $n_e$',
        r'(i) $J_x$', r'(j) $J_y$', r'(k) $J_z$', r'(l) $n_i$',
        r'(m) $V_{ix}/V_{A0}$', r'(n) $V_{iy}/V_{A0}$', r'(o) $V_{iz}/V_{A0}$', r'(p) $T_i$',
        r'(q) $V_{ex}/V_{A0}$', r'(r) $V_{ey}/V_{A0}$', r'(s) $V_{ez}/V_{A0}$', r'(t) $T_e$'
    ]

    fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
    fig.suptitle(f"Timestep: {ts}  ({omega_t_str})", fontsize=30, fontweight='bold')
    
    for i, (key, data, _, label, cmap) in enumerate(all_items):
        ax = axes.flatten()[i]
        vmin, vmax = GLOBAL_PLOT_RANGES.get(key, get_plot_range(data))
        cf = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        ax.set_title(combined_titles[i], fontsize=18)
        ax.set_xlabel('$x/d_i$', fontsize=16); ax.set_ylabel('$y/d_i$', fontsize=16)
        ax.tick_params(direction='in', top=True, right=True, labelsize=14)
        
        stream_c = 'white' if cmap == plt.cm.seismic else 'gray'
        ax.contour(X, Y, Psi, levels=15, colors=stream_c, linewidths=0.5, alpha=0.8)
        apply_colorbar(cf, ax, label, key, is_combined=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=0.5)
    os.makedirs(os.path.join(OUTPUT_DIR, 'allcombined'), exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'allcombined', f'plot_combined_{ts}.png'), dpi=300)
    plt.close(fig)
    print(f"Finished Timestep: {ts}")

# =======================================================
# 5. 並列実行
# =======================================================
if __name__ == "__main__":
    plt.rcParams.update({
        'font.size': 20, 'font.family': 'serif', 'mathtext.fontset': 'cm', 
        'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
        'figure.titlesize': 28
    })
    
    if len(sys.argv) != 4:
        print("Usage: python script.py [start] [end] [interval]"); sys.exit(1)
        
    start_s, end_s, inter_s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    steps = range(start_s, end_s + 1, inter_s)
    
    print(f"Parallel processing with {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_timestep, steps)
    print("All tasks completed.")