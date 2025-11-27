import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# CONFIGURATION
# =======================================================
# リビニング設定
BINNING_FACTOR = 4

# 前処理スムージング
PRE_SMOOTH_SIGMA = 2.0 

# ★検証用: ベクトル場へのスムージング★
# これを 0.0 にすると「ぐちゃぐちゃ」な状態に戻ります。
# 1.0 にすると、提案手法による改善効果が確認できます。
VECTOR_SMOOTH_SIGMA = 1.0

# 表示設定
LOW_B_CUTOFF = 0.02
VMAX_PERCENTILE = 95.0

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'heating_plots_pressureetc')
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# =======================================================
# HELPER FUNCTIONS
# =======================================================
def load_simulation_parameters(param_filepath):
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
                elif line.startswith('Fpe, Fge'):
                    FPI, FGI = float(parts[7]), float(parts[8])
                elif line.startswith('Va, Vi'):
                    VA0 = float(parts[7])
    except: pass
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
if None in [C_LIGHT, FPI, DT, FGI, VA0, MI, QI]:
    DT, FGI, VA0, DI, B0 = 0.02, 0.04, 0.1, 100.0, 1.0
else:
    DI = C_LIGHT / FPI
    B0 = (FGI * MI * C_LIGHT) / QI

GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0
DX_PHYS = DELX / DI 
DX_PHYS_COARSE = DX_PHYS * BINNING_FACTOR

def load_smooth_data(timestep, subdir, prefix, suffix):
    filename = f'{prefix}_{timestep}_{suffix}.txt'
    filepath = os.path.join(subdir, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return gaussian_filter(data, sigma=PRE_SMOOTH_SIGMA)
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def rebin(arr, factor):
    if factor <= 1: return arr
    ny, nx = arr.shape
    new_ny, new_nx = ny // factor, nx // factor
    arr_trimmed = arr[:new_ny*factor, :new_nx*factor]
    return arr_trimmed.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))

def gradient_2d(f, dx):
    grad_y, grad_x = np.gradient(f, dx, edge_order=2)
    return grad_x, grad_y

def calculate_check_components(timestep):
    # Load Fields
    Bx = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bx'), BINNING_FACTOR) / B0
    By = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'By'), BINNING_FACTOR) / B0
    Bz = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bz'), BINNING_FACTOR) / B0
    Ex = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ex'), BINNING_FACTOR) / B0
    Ey = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ey'), BINNING_FACTOR) / B0
    Ez = rebin(load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ez'), BINNING_FACTOR) / B0
    
    # Load Moments
    ne = rebin(load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_density_count'), BINNING_FACTOR)
    ni = rebin(load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_density_count'), BINNING_FACTOR)
    Vxe = rebin(load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vx'), BINNING_FACTOR) / VA0
    Vye = rebin(load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vy'), BINNING_FACTOR) / VA0
    Vze = rebin(load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vz'), BINNING_FACTOR) / VA0

    # Tensor
    def load_tensor_rebin(ts):
        comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
        tensor = {}
        path_check = os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Txx.txt')
        if os.path.exists(path_check):
            for c in comps:
                raw = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', f'electron_T{c}')
                tensor[c] = rebin(raw / (VA0**2), BINNING_FACTOR)
        else:
            raw_T = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', 'electron_T')
            T_iso = rebin(raw_T / (VA0**2), BINNING_FACTOR)
            tensor = {c: (np.zeros_like(T_iso) if 'x' in c and 'y' in c else T_iso) for c in comps}
        return tensor
    Te_tensor = load_tensor_rebin(timestep)

    # --- Calculations ---
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    valid_mask = B_mag > LOW_B_CUTOFF
    B_safe = B_mag.copy()
    B_safe[~valid_mask] = 1.0 
    bx, by, bz = Bx/B_safe, By/B_safe, Bz/B_safe

    # Pressure Components
    N_tot = (ne + ni) / 2.0
    avg_N = np.mean(N_tot[N_tot > 0.1]) if np.any(N_tot > 0.1) else 1.0
    n_proxy = N_tot / avg_N
    
    Pxx, Pyy, Pzz = n_proxy*Te_tensor['xx'], n_proxy*Te_tensor['yy'], n_proxy*Te_tensor['zz']
    Pxy, Pyz, Pxz = n_proxy*Te_tensor['xy'], n_proxy*Te_tensor['yz'], n_proxy*Te_tensor['xz']
    p_par = (bx**2*Pxx + by**2*Pyy + bz**2*Pzz + 2*bx*by*Pxy + 2*by*bz*Pyz + 2*bx*bz*Pxz)
    trace_P = Pxx + Pyy + Pzz
    p_perp = (trace_P - p_par) / 2.0

    # Kinematic Energy (rho * u_par^2)
    ue_par = Vxe*bx + Vye*by + Vze*bz
    dynamic_pressure = n_proxy * ue_par**2

    # --- Vector Terms Calculation & Smoothing ---
    
    # 1. ExB Drift Vector
    uE_x = (Ey*Bz - Ez*By) / (B_safe**2)
    uE_y = (Ez*Bx - Ex*Bz) / (B_safe**2)
    uE_z = (Ex*By - Ey*Bx) / (B_safe**2)

    # 2. Curvature Vector
    dbx_dx, dbx_dy = gradient_2d(bx, DX_PHYS_COARSE)
    dby_dx, dby_dy = gradient_2d(by, DX_PHYS_COARSE)
    dbz_dx, dbz_dy = gradient_2d(bz, DX_PHYS_COARSE)
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy

    # ★ SMOOTHING VECTORS BEFORE DOT PRODUCT ★
    if VECTOR_SMOOTH_SIGMA > 0:
        uE_x = gaussian_filter(uE_x, sigma=VECTOR_SMOOTH_SIGMA)
        uE_y = gaussian_filter(uE_y, sigma=VECTOR_SMOOTH_SIGMA)
        uE_z = gaussian_filter(uE_z, sigma=VECTOR_SMOOTH_SIGMA)
        
        kappa_x = gaussian_filter(kappa_x, sigma=VECTOR_SMOOTH_SIGMA)
        kappa_y = gaussian_filter(kappa_y, sigma=VECTOR_SMOOTH_SIGMA)
        kappa_z = gaussian_filter(kappa_z, sigma=VECTOR_SMOOTH_SIGMA)

    # 3. Dot Product (ve . kappa)
    vek_term = uE_x*kappa_x + uE_y*kappa_y + uE_z*kappa_z

    # Apply mask
    p_par[~valid_mask] = np.nan
    p_perp[~valid_mask] = np.nan
    dynamic_pressure[~valid_mask] = np.nan
    vek_term[~valid_mask] = np.nan

    return p_par, p_perp, dynamic_pressure, vek_term, Bx, By

def plot_check_vek(timestep):
    print(f"--- Checking ve K with Smoothing (sigma={VECTOR_SMOOTH_SIGMA}) - TS {timestep} ---")
    p_par, p_perp, dyn_pres, vek, Bx, By = calculate_check_components(timestep)
    
    ny, nx = Bx.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axs = axes.flatten()
    
    # Plot definitions
    plots = [
        (p_par,      r'Parallel Pressure $P_\parallel$', 'plasma'),
        (p_perp,     r'Perpendicular Pressure $P_\perp$', 'plasma'),
        (dyn_pres,   r'Dynamic Pressure $n m u_\parallel^2$', 'viridis'),
        (vek,        rf'Smoothed Geometric Drive $\mathbf{{v}}_E \cdot \boldsymbol{{\kappa}}$ ($\sigma=${VECTOR_SMOOTH_SIGMA})', 'RdBu_r') 
    ]
    
    for i, (data, title, cmap) in enumerate(plots):
        ax = axs[i]
        ax.set_facecolor('#d3d3d3')
        
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            if 'Pressure' in title:
                vmax = np.percentile(np.abs(valid_data), VMAX_PERCENTILE)
                vmin = 0.0
            else:
                # ve k は正負あるので対称に
                max_val = np.percentile(np.abs(valid_data), VMAX_PERCENTILE)
                vmax = max_val
                vmin = -max_val
        else:
            vmax, vmin = 1.0, 0.0
            
        cf = ax.contourf(X, Y, data, levels=100, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        
        stride = max(1, nx // 20)
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black' if 'Pressure' in title else 'black', 
                      linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        plt.colorbar(cf, ax=ax, shrink=0.9)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'check_vek_{timestep}.png')
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python check_vek.py [start] [end] [interval]")
    else:
        s, e, i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        curr = s
        while curr <= e:
            try:
                plot_check_vek(f"{curr:06d}")
            except Exception as ex:
                print(f"Error: {ex}")
            curr += i