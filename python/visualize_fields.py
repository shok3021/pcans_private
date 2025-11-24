import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ヘルパー関数
# =======================================================
def load_simulation_parameters(param_filepath):
    # (元のコードと同じ実装のため省略しますが、必ず元のコードの load_simulation_parameters をここに含めてください)
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = None, None, None, None, None, None, None
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if line.startswith('dx, dt, c'):
                    DT, C_LIGHT = float(parts[5]), float(parts[6])
                elif line.startswith('Mi, Me'):
                    MI = float(parts[3])
                elif line.startswith('Qi, Qe'):
                    QI = float(parts[3])
                elif line.startswith('Fpe, Fge, Fpi Fgi'):
                    FPI, FGI = float(parts[7]), float(parts[8])
                elif line.startswith('Va, Vi, Ve'):
                    VA0 = float(parts[7])
    except Exception as e:
        print(f"Error loading param: {e}"); sys.exit(1)
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# =======================================================
# 設定と定数
# =======================================================
try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except: SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'allcombined'), exist_ok=True)

# ユーザー環境のパスに合わせてください
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI
B0 = (FGI * MI * C_LIGHT) / QI

print(f"Norm Scales: di={DI:.2f}, B0={B0:.2f}, VA0={VA0:.2f}")

# ★★★ プロットレンジの更新 (eV対応) ★★★
# 値はシミュレーションパラメータによります。適宜調整してください。
GLOBAL_PLOT_RANGES = {
    'Bx': (-1.0, 1.0),
    'By': (-1.5, 1.5),
    'Bz': (-0.5, 0.5),
    'Ex': (-0.3, 0.3),
    'Ey': (-0.05, 0.05),
    'Ez': (-0.1, 0.1),
    'Jx': (-0.5, 0.5),
    'Jy': (-0.75, 0.75),
    'Jz': (-1.75, 1.75),
    'ne': (0, 500),
    'ni': (0, 500),
    'Psi': (-5000, 5000), 
    
    # ★ eV単位の設定 (例: 電子は数keVまで、イオンも同様と仮定)
    # 必要に応じて (0, 100) や (0, 5000) などに変更してください
    'Te': (0, 2000),   
    'Ti': (0, 2000),
    
    'Vxe': (-1.0, 1.0),
    'Vxi': (-0.5, 0.5),
    'Vye': (-2.5, 2.5),
    'Vyi': (-1.0, 1.0),
    'Vze': (-4.0, 4.0),
    'Vzi': (-2.0, 2.0),
}

GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# =======================================================
# データ読み込み・処理
# =======================================================
def load_2d_field_data(timestep, component):
    # (元のコードと同じ)
    path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_{component}.txt')
    try: return np.loadtxt(path, delimiter=',')
    except: return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_2d_moment_data(timestep, species, component):
    # (元のコードと同じ)
    path = os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_{species}_{component}.txt')
    try: return np.loadtxt(path, delimiter=',')
    except: return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def calculate_current_density(Bx, By, Ex, Ey, Ez, J_list, B0):
    # (元のコードと同じ)
    n_proxy = (J_list['density_count_e'] + J_list['density_count_i']) / 2.0
    avg_N0 = np.mean(n_proxy[n_proxy > 0.1]) if np.any(n_proxy > 0.1) else 1.0
    n_norm = n_proxy / avg_N0
    Jx = n_norm * (J_list['Vx_i'] - J_list['Vx_e'])
    Jy = n_norm * (J_list['Vy_i'] - J_list['Vy_e'])
    Jz = n_norm * (J_list['Vz_i'] - J_list['Vz_e'])
    return Jx, Jy, Jz, Ez

def calculate_magnetic_flux(Bx, By, DELX):
    return cumtrapz(By, dx=DELX, axis=1, initial=0)

def create_coordinates(NX, NY):
    x = np.linspace(-NX*DELX/2, NX*DELX/2, NX) / DI
    y = np.linspace(0, NY*DELX, NY) / DI
    return np.meshgrid(x, y)

def get_plot_range(Z, tag=None):
    if Z.size == 0: return -1, 1
    m = np.nanmax(np.abs(Z))
    if m == 0: return -1e-6, 1e-6
    return -m, m

# =======================================================
# プロット関数
# =======================================================
def plot_single_panel(ax, X, Y, Z, Bx, By, title, label, omega_t_str, cmap='RdBu_r', vmin=None, vmax=None, tag_key=None):
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    if np.isclose(vmin, vmax): vmin -= 1e-6; vmax += 1e-6
        
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap) 
    
    # 数値フォーマットの調整: eVなどの大きな値に対応
    fmt = '%.1e' if (np.abs(vmax) > 10000 or (vmax > 0 and vmax < 0.01)) else '%.2f'
    # eVの場合は整数で見たい場合もあるため適宜 '%.0f' などにしても良い
    
    ticks = np.linspace(vmin, vmax, 6)
    cbar = plt.colorbar(cf, ax=ax, format=fmt, ticks=ticks)
    cbar.set_label(label)

    try:
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
        ax.contour(X, Y, Psi_local, levels=25, colors='gray', linewidths=0.5, alpha=0.8)
    except: pass
    
    ax.text(0.98, 0.98, omega_t_str, transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.set_title(title); ax.set_xlabel('$x/d_i$'); ax.set_ylabel('$y/d_i$')

def plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap, vmin, vmax, tag_key, stream_color='gray'):
    if np.isclose(vmin, vmax): vmin -= 1e-6; vmax += 1e-6
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

    fmt = '%.1e' if (np.abs(vmax) > 10000 or (vmax > 0 and vmax < 0.01)) else '%.2f'
    ticks = np.linspace(vmin, vmax, 5)
    cbar = plt.colorbar(cf, ax=ax, format=fmt, shrink=0.9, aspect=30, pad=0.02, ticks=ticks)
    cbar.set_label(label, fontsize=7); cbar.ax.tick_params(labelsize=6)
    
    try:
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
        ax.contour(X, Y, Psi_local, levels=15, colors=stream_color, linewidths=0.5, alpha=0.8)
    except: pass
                  
    ax.set_title(title, fontsize=10); ax.set_xlabel('$x/d_i$', fontsize=8); ax.set_ylabel('$y/d_i$', fontsize=8)
    ax.tick_params(labelsize=7)

# =======================================================
# Main Processing
# =======================================================
def process_timestep(timestep):
    print(f"\n--- Processing TS: {timestep} ---")
    
    # 読み込み
    Bx_raw = load_2d_field_data(timestep, 'Bx')
    By_raw = load_2d_field_data(timestep, 'By')
    Bz_raw = load_2d_field_data(timestep, 'Bz')
    Ex_raw = load_2d_field_data(timestep, 'Ex')
    Ey_raw = load_2d_field_data(timestep, 'Ey')
    Ez_raw = load_2d_field_data(timestep, 'Ez')
    Vxe_raw = load_2d_moment_data(timestep, 'electron', 'Vx')
    Vye_raw = load_2d_moment_data(timestep, 'electron', 'Vy')
    Vze_raw = load_2d_moment_data(timestep, 'electron', 'Vz')
    Vxi_raw = load_2d_moment_data(timestep, 'ion', 'Vx')
    Vyi_raw = load_2d_moment_data(timestep, 'ion', 'Vy')
    Vzi_raw = load_2d_moment_data(timestep, 'ion', 'Vz')
    ne = load_2d_moment_data(timestep, 'electron', 'density_count')
    ni = load_2d_moment_data(timestep, 'ion', 'density_count')

    # ★ 温度 (eV) の読み込み (既にeV単位なのでVA0規格化はしない)
    Te_eV = load_2d_moment_data(timestep, 'electron', 'T')
    Ti_eV = load_2d_moment_data(timestep, 'ion', 'T')

    # 規格化
    Bx = Bx_raw / B0; By = By_raw / B0; Bz = Bz_raw / B0
    Ex = Ex_raw / B0; Ey = Ey_raw / B0; Ez = Ez_raw / B0
    Vxe = Vxe_raw / VA0; Vye = Vye_raw / VA0; Vze = Vze_raw / VA0
    Vxi = Vxi_raw / VA0; Vyi = Vyi_raw / VA0; Vzi = Vzi_raw / VA0
    Psi = calculate_magnetic_flux(Bx_raw, By_raw, DELX)

    # 電流などの計算
    J_data = {'density_count_e': ne, 'density_count_i': ni,
              'Vx_e': Vxe_raw, 'Vx_i': Vxi_raw, 'Vy_e': Vye_raw, 'Vy_i': Vyi_raw, 'Vz_e': Vze_raw, 'Vz_i': Vzi_raw}
    Jx, Jy, Jz, _ = calculate_current_density(Bx_raw, By_raw, Ex_raw, Ey_raw, Ez_raw, J_data, B0)
    
    X, Y = create_coordinates(GLOBAL_NX_PHYS, GLOBAL_NY_PHYS)
    omega_t_str = fr"$\Omega_{{ci}}t = {FGI * float(timestep) * DT:.2f}$"

    # プロットリスト
    # ★ ラベルを eV に変更
    plot_components = [
        ('Bx', Bx, r'Magnetic Field ($B_x/B_0$)', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, r'Magnetic Field ($B_y/B_0$)', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, r'Magnetic Field ($B_z/B_0$)', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Ex', Ex, r'Electric Field ($E_x/B_0$)', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, r'Electric Field ($E_y/B_0$)', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, r'Electric Field ($E_z/B_0$)', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', ne, 'Electron Density', r'$n_e$', plt.cm.viridis),
        ('ni', ni, 'Ion Density', r'$n_i$', plt.cm.viridis),
        
        # ★ ここを修正
        ('Te', Te_eV, 'Electron Temperature (eV)', r'$T_e$ (eV)', plt.cm.plasma),
        ('Ti', Ti_eV, 'Ion Temperature (eV)', r'$T_i$ (eV)', plt.cm.plasma),
        
        ('Psi', Psi, r'Magnetic Flux $\Psi$', r'$\Psi$', plt.cm.seismic),
        ('Jx', Jx, 'Current Density (Jx)', r'$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, 'Current Density (Jy)', r'$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, 'Current Density (Jz)', r'$J_z$', plt.cm.RdBu_r),
        ('Vxi', Vxi, r'Ion Velocity ($V_{ix}/V_{A0}$)', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        ('Vyi', Vyi, r'Ion Velocity ($V_{iy}/V_{A0}$)', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        ('Vzi', Vzi, r'Ion Velocity ($V_{iz}/V_{A0}$)', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        ('Vxe', Vxe, r'Electron Velocity ($V_{ex}/V_{A0}$)', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        ('Vye', Vye, r'Electron Velocity ($V_{ey}/V_{A0}$)', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        ('Vze', Vze, r'Electron Velocity ($V_{ez}/V_{A0}$)', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
    ]

    # 個別プロット
    for tag_key, Z, title, label, cmap in plot_components:
        SUB_DIR = os.path.join(OUTPUT_DIR, tag_key)
        os.makedirs(SUB_DIR, exist_ok=True)
        vmin, vmax = GLOBAL_PLOT_RANGES.get(tag_key, get_plot_range(Z))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_panel(ax, X, Y, Z, Bx, By, f"{title}", label, omega_t_str, cmap, vmin, vmax, tag_key)
        plt.savefig(os.path.join(SUB_DIR, f'plot_{timestep}_{tag_key}.png'), dpi=150)
        plt.close(fig)

    # 統合プロット
    fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
    ax_list = axes.flatten()
    fig.suptitle(f"Timestep: {timestep}  ({omega_t_str})", fontsize=16)

    # 順番を定義 (Te, Ti を含む)
    combined_keys = [
        'Bx', 'By', 'Bz', 'Psi', 
        'Ex', 'Ey', 'Ez', 'ne', 
        'Jx', 'Jy', 'Jz', 'ni', 
        'Vxi', 'Vyi', 'Vzi', 'Ti', 
        'Vxe', 'Vye', 'Vze', 'Te'
    ]
    
    # 辞書化してアクセス
    plot_dict = {item[0]: item[1:] for item in plot_components} # key: (Z, title, label, cmap)

    for i, key in enumerate(combined_keys):
        if i >= len(ax_list): break
        Z, title, label, cmap = plot_dict[key]
        vmin, vmax = GLOBAL_PLOT_RANGES.get(key, get_plot_range(Z))
        plot_combined(ax_list[i], X, Y, Z, Bx, By, title, label, cmap, vmin, vmax, key, 
                      stream_color='white' if cmap==plt.cm.seismic else 'gray')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, 'allcombined', f'plot_combined_{timestep}.png'), dpi=200)
    plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visual_fields.py [start] [end] [step]"); sys.exit(1)
    
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for s in range(start, end + 1, step):
        process_timestep(f"{s:06d}")