import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors

# =======================================================
# CONFIGURATION
# =======================================================

# 1. 空間平均 (Coarse Graining)
BINNING_FACTOR = 4

# 2. 入力データへのスムージング (前処理)
PRE_SMOOTH_SIGMA = 2.0 

# 3. ベクトル場へのスムージング (中間処理)
VECTOR_SMOOTH_SIGMA = 1.0

# 4. 計算後の加熱項へのスムージング (後処理)
POST_SMOOTH_SIGMA = 1.0

# 表示設定
LOW_B_CUTOFF = 0.02
# VMAX_PERCENTILE は固定レンジにするため使用しません

# パス設定
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'heating_plots_final')
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

def calculate_gca_heating(timestep):
    # 1. Load & Pre-Smooth (Fields)
    _Bx = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bx')
    _By = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'By')
    _Bz = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bz')
    _Ex = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ex')
    _Ey = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ey')
    _Ez = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ez')
    
    _ne = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_density_count')
    _ni = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_density_count')
    
    _Vxe = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vx') / VA0
    _Vye = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vy') / VA0
    _Vze = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vz') / VA0
    _Vxi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vx') / VA0
    _Vyi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vy') / VA0
    _Vzi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vz') / VA0

    # 2. Rebin
    Bx = rebin(_Bx, BINNING_FACTOR)
    By = rebin(_By, BINNING_FACTOR)
    Bz = rebin(_Bz, BINNING_FACTOR)
    Ex = rebin(_Ex, BINNING_FACTOR)
    Ey = rebin(_Ey, BINNING_FACTOR)
    Ez = rebin(_Ez, BINNING_FACTOR)
    
    ne = rebin(_ne, BINNING_FACTOR)
    ni = rebin(_ni, BINNING_FACTOR)
    
    Vxe = rebin(_Vxe, BINNING_FACTOR)
    Vye = rebin(_Vye, BINNING_FACTOR)
    Vze = rebin(_Vze, BINNING_FACTOR)
    Vxi = rebin(_Vxi, BINNING_FACTOR)
    Vyi = rebin(_Vyi, BINNING_FACTOR)
    Vzi = rebin(_Vzi, BINNING_FACTOR)

    # --- Density Proxy ---
    N_tot = (ne + ni) / 2.0
    avg_N = np.mean(N_tot[N_tot > 0.1]) if np.any(N_tot > 0.1) else 1.0
    n_proxy = N_tot / avg_N
    
    # --- Temperature Tensor Loading & Scaling ---
    def load_tensor_raw(ts):
        comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
        tensor = {}
        for c in comps:
            path_check = os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_T{c}.txt')
            if os.path.exists(path_check):
                raw = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', f'electron_T{c}')
                tensor[c] = rebin(raw, BINNING_FACTOR)
            else:
                raw_T = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', 'electron_T')
                val = rebin(raw_T, BINNING_FACTOR)
                tensor = {k: (np.zeros_like(val) if 'x' in k and 'y' in k else val) for k in comps}
                break
        return tensor

    Te_tensor_raw = load_tensor_raw(timestep)
    
    # ★★★ 自動スケーリング計算 (B=1, beta=0.8基準) ★★★
    TARGET_BETA = 0.8
    # B=1.0 (Normalized) のときのターゲット圧力
    target_P_avg = TARGET_BETA * 0.5 
    
    trace_T_raw = Te_tensor_raw['xx'] + Te_tensor_raw['yy'] + Te_tensor_raw['zz']
    mean_Trace_T = np.nanmean(trace_T_raw)
    current_P_proxy = 1.0 * (mean_Trace_T / 3.0) 
    
    if current_P_proxy > 1e-12:
        SCALE_FACTOR = target_P_avg / current_P_proxy
    else:
        SCALE_FACTOR = 1.0

    print(f"  [AUTO-SCALING] Target P (Fixed): {target_P_avg:.4e}, Raw P_proxy: {current_P_proxy:.4e}")
    print(f"  [AUTO-SCALING] Applied Scale Factor: {SCALE_FACTOR:.4e}")
    
    Te_tensor = {k: v * SCALE_FACTOR for k, v in Te_tensor_raw.items()}
    # ★★★★★★★★★★★★★★★★★★★★★★★★★
    
    # --- Vector Calculations ---
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    valid_mask = B_mag > LOW_B_CUTOFF
    B_safe = B_mag.copy()
    B_safe[~valid_mask] = 1.0 
    
    bx, by, bz = Bx/B_safe, By/B_safe, Bz/B_safe
    
    Jx = n_proxy * (Vxi - Vxe)
    Jy = n_proxy * (Vyi - Vye)
    Jz = n_proxy * (Vzi - Vze)
    
    E_par = Ex*bx + Ey*by + Ez*bz
    J_par = Jx*bx + Jy*by + Jz*bz
    ue_par = Vxe*bx + Vye*by + Vze*bz
    
    # Drift & Curvature
    uE_x = (Ey*Bz - Ez*By) / (B_safe**2)
    uE_y = (Ez*Bx - Ex*Bz) / (B_safe**2)
    uE_z = (Ex*By - Ey*Bx) / (B_safe**2)
    
    dbx_dx, dbx_dy = gradient_2d(bx, DX_PHYS_COARSE)
    dby_dx, dby_dy = gradient_2d(by, DX_PHYS_COARSE)
    dbz_dx, dbz_dy = gradient_2d(bz, DX_PHYS_COARSE)
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy
    
    gradB_x, gradB_y = gradient_2d(B_mag, DX_PHYS_COARSE)
    
    # Smoothing
    uE_x = gaussian_filter(uE_x, sigma=VECTOR_SMOOTH_SIGMA)
    uE_y = gaussian_filter(uE_y, sigma=VECTOR_SMOOTH_SIGMA)
    uE_z = gaussian_filter(uE_z, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_x = gaussian_filter(kappa_x, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_y = gaussian_filter(kappa_y, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_z = gaussian_filter(kappa_z, sigma=VECTOR_SMOOTH_SIGMA)
    gradB_x = gaussian_filter(gradB_x, sigma=VECTOR_SMOOTH_SIGMA)
    gradB_y = gaussian_filter(gradB_y, sigma=VECTOR_SMOOTH_SIGMA)
    
    # Heating Calculation
    Pxx, Pyy, Pzz = Te_tensor['xx'], Te_tensor['yy'], Te_tensor['zz']
    Pxy, Pyz, Pxz = Te_tensor['xy'], Te_tensor['yz'], Te_tensor['xz']
    
    p_par = (bx**2*Pxx + by**2*Pyy + bz**2*Pzz + 2*bx*by*Pxy + 2*by*bz*Pyz + 2*bx*bz*Pxz)
    trace_P = Pxx + Pyy + Pzz
    p_perp = (trace_P - p_par) / 2.0
    
    term_E_par = E_par * J_par
    
    curvature_drive = uE_x*kappa_x + uE_y*kappa_y + uE_z*kappa_z
    term_Curvature = (p_par + n_proxy * ue_par**2) * curvature_drive
    
    gradB_drive = uE_x * gradB_x + uE_y * gradB_y
    term_GradB = (p_perp / B_safe) * gradB_drive
    
    term_Total = term_E_par + term_Curvature + term_GradB

    # Post-Smoothing
    term_Total = gaussian_filter(term_Total, sigma=POST_SMOOTH_SIGMA)
    term_E_par = gaussian_filter(term_E_par, sigma=POST_SMOOTH_SIGMA)
    term_Curvature = gaussian_filter(term_Curvature, sigma=POST_SMOOTH_SIGMA)
    term_GradB = gaussian_filter(term_GradB, sigma=POST_SMOOTH_SIGMA)
    
    term_Total[~valid_mask] = np.nan
    term_E_par[~valid_mask] = np.nan
    term_Curvature[~valid_mask] = np.nan
    term_GradB[~valid_mask] = np.nan
    
    # Checks
    abs_JE = np.abs(term_E_par)
    abs_Curv = np.abs(term_Curvature)
    mean_JE = np.nanmean(abs_JE)
    mean_Curv = np.nanmean(abs_Curv)
    
    print(f"  [CHECK] J.E: {mean_JE:.4e}, Curv: {mean_Curv:.4e}")
    
    return term_Total, term_E_par, term_Curvature, term_GradB, Bx, By

def plot_heating_final(timestep):
    print(f"\n--- Processing timestep: {timestep} (Fixed Range -0.5 to 0.5) ---")
    
    total, e_par, curve, grad_b, Bx, By = calculate_gca_heating(timestep)
    
    ny, nx = Bx.shape
    x_phys = np.linspace(-nx * DX_PHYS_COARSE / 2.0, nx * DX_PHYS_COARSE / 2.0, nx)
    y_phys = np.linspace(0.0, ny * DX_PHYS_COARSE, ny)
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    
    plot_configs = [
        (total, r'Total Heating $dU/dt$', axs[0]),
        (e_par, r'Parallel E-Field Work: $E_\parallel J_\parallel$', axs[1]),
        (curve, r'Curvature Drift Heating ($v_E \cdot \kappa$)', axs[2]),
        (grad_b, r'Grad-B Drift Heating ($v_E \cdot \nabla B$)', axs[3])
    ]
    
    # ★★★ 固定レンジ設定 (-1.0 to 1.0) ★★★
    FIXED_LIMIT = 1.0   
    LIN_THRESH = 0.01

    # 【重要】levels を -1.0 ～ 1.0 の範囲で100等分した配列にする
    # これにより、データがどれだけ大きくても、描画とカラーバーは ±1.0 で固定されます
    fixed_levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 1000)

    for data, title, ax in plot_configs:
        ax.set_facecolor('#d3d3d3')
        
        # SymLogNormの設定（ここはそのまま）
        norm = colors.SymLogNorm(linthresh=LIN_THRESH, linscale=1.0, 
                                 vmin=-FIXED_LIMIT, vmax=FIXED_LIMIT, base=10)
        
        # 【修正箇所】 data はそのまま（clipしない）。levels に fixed_levels を渡す。
        # extend='both' にすることで、範囲外の値（>1.0 や <-1.0）には
        # カラーバーの両端の色（濃い赤/濃い青）が適用されます。
        cf = ax.contourf(X, Y, data, levels=fixed_levels, cmap='seismic', norm=norm)
                
        stride_x = max(1, nx // 30)
        stride_y = max(1, ny // 30)
        ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                      Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                      color='black', linewidth=0.5, density=1.0, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x/d_i$', fontsize=12)
        ax.set_ylabel('$y/d_i$', fontsize=12)
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95)
        cbar.set_label('Heating Rate', fontsize=10)

    fig.suptitle(rf"Electron Heating (Fixed Range $\pm$0.5) - TS {timestep}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, f'heating_final_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python final_visual.py [start] [end] [interval]")
    else:
        s, e, i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        curr = s
        while curr <= e:
            try:
                plot_heating_final(f"{curr:06d}")
            except Exception as ex:
                print(f"Error: {ex}")
            curr += i