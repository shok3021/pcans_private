import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# CONFIGURATION
# =======================================================
# ノイズ源特定のため、あえて少し解像度を残すか、前回と同じ条件で比較するか
# ここでは前回と同じ条件で「何が悪いか」を見ます
BINNING_FACTOR = 4

# 前処理スムージング (これを切るともっと酷くなるか確認できますが、まずは標準で)
PRE_SMOOTH_SIGMA = 2.0 

# 低磁場カットオフ
LOW_B_CUTOFF = 0.02

# 表示する値の最大値調整 (パーセンタイル)
# ノイズのスパイクを無視して全体像を見るために低めに設定
VMAX_PERCENTILE = 95.0

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'heating_plots_kappaetc')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# パラメータファイルへのパス (適宜変更してください)
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# =======================================================
# HELPER FUNCTIONS
# =======================================================
def load_simulation_parameters(param_filepath):
    # (前回と同じパラメータ読み込み)
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

# =======================================================
# DIAGNOSTIC CALCULATION
# =======================================================
def calculate_diagnostics(timestep):
    # Load Basic Fields
    _Bx = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bx') / B0
    _By = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'By') / B0
    _Bz = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bz') / B0
    _Ex = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ex') / B0
    _Ey = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ey') / B0
    _Ez = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ez') / B0
    
    # Rebin
    Bx = rebin(_Bx, BINNING_FACTOR)
    By = rebin(_By, BINNING_FACTOR)
    Bz = rebin(_Bz, BINNING_FACTOR)
    Ex = rebin(_Ex, BINNING_FACTOR)
    Ey = rebin(_Ey, BINNING_FACTOR)
    Ez = rebin(_Ez, BINNING_FACTOR)
    
    # Magnitudes & Unit Vectors
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    valid_mask = B_mag > LOW_B_CUTOFF
    B_safe = B_mag.copy()
    B_safe[~valid_mask] = 1.0 
    
    bx, by, bz = Bx/B_safe, By/B_safe, Bz/B_safe
    
    # 1. Curvature Vector kappa = (b.grad)b
    dbx_dx, dbx_dy = gradient_2d(bx, DX_PHYS_COARSE)
    dby_dx, dby_dy = gradient_2d(by, DX_PHYS_COARSE)
    dbz_dx, dbz_dy = gradient_2d(bz, DX_PHYS_COARSE)
    
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy
    kappa_mag = np.sqrt(kappa_x**2 + kappa_y**2 + kappa_z**2)
    
    # 2. Grad B Vector
    gradB_x, gradB_y = gradient_2d(B_mag, DX_PHYS_COARSE)
    gradB_mag = np.sqrt(gradB_x**2 + gradB_y**2)
    
    # 3. ExB Drift Velocity
    # vE = (E x B) / B^2
    ExB_x = Ey*Bz - Ez*By
    ExB_y = Ez*Bx - Ex*Bz
    ExB_z = Ex*By - Ey*Bx
    
    vE_x = ExB_x / (B_safe**2)
    vE_y = ExB_y / (B_safe**2)
    vE_z = ExB_z / (B_safe**2)
    vE_mag = np.sqrt(vE_x**2 + vE_y**2 + vE_z**2)
    
    # Masking
    kappa_mag[~valid_mask] = np.nan
    gradB_mag[~valid_mask] = np.nan
    vE_mag[~valid_mask] = np.nan
    E_mag[~valid_mask] = np.nan
    
    return kappa_mag, gradB_mag, vE_mag, E_mag, Bx, By

# =======================================================
# PLOTTING
# =======================================================
def plot_debug(timestep):
    print(f"--- Debugging timestep: {timestep} ---")
    
    kappa_mag, gradB_mag, vE_mag, E_mag, Bx, By = calculate_diagnostics(timestep)
    
    ny, nx = Bx.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axs = axes.flatten()
    
    # Plot definitions
    # (Data, Title, Colormap)
    plots = [
        (kappa_mag, r'Curvature Magnitude $|\kappa|$ (Geometry)', 'magma'),
        (gradB_mag, r'Grad-B Magnitude $|\nabla B|$ (Geometry)', 'magma'),
        (vE_mag,    r'ExB Drift Speed $|v_E|$ (Driver)', 'viridis'),
        (E_mag,     r'Raw Electric Field $|E|$ (Noise Source?)', 'inferno')
    ]
    
    omega_t = float(timestep) * DT * FGI
    
    for i, (data, title, cmap) in enumerate(plots):
        ax = axs[i]
        ax.set_facecolor('#d3d3d3')
        
        # Determine color scale (ignore NaNs)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmax = np.percentile(valid_data, VMAX_PERCENTILE)
            vmin = 0.0 # Magnitudes are positive
        else:
            vmax, vmin = 1.0, 0.0
            
        cf = ax.contourf(X, Y, data, levels=100, cmap=cmap, vmin=vmin, vmax=vmax, extend='max')
        
        # Add magnetic field lines
        stride = max(1, nx // 20)
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='white', linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        plt.colorbar(cf, ax=ax, shrink=0.9).set_label('Magnitude')
    
    fig.suptitle(rf"Noise Diagnostics - Timestep {timestep} ($\Omega_{{ci}}t={omega_t:.1f}$)" + "\n" +
                 f"Smoothing: $\sigma=${PRE_SMOOTH_SIGMA}, Binning: {BINNING_FACTOR}x{BINNING_FACTOR}", fontsize=16)
    
    save_path = os.path.join(OUTPUT_DIR, f'debug_terms_{timestep}.png')
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python debug_visual.py [start] [end] [interval]")
    else:
        s, e, i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        curr = s
        while curr <= e:
            try:
                plot_debug(f"{curr:06d}")
            except Exception as ex:
                print(f"Error: {ex}")
            curr += i