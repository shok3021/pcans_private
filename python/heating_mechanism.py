import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# CONFIGURATION
# =======================================================
# 1. 入力データへのスムージング (前処理)
# ここを強く(3.0)して、微分前の粒子ノイズを徹底的に潰します
PRE_SMOOTH_SIGMA = 3.0

# 2. 計算後の加熱項へのスムージング (後処理)
# 微分によって新たに生まれたノイズを均します
POST_SMOOTH_SIGMA = 1.5

# 低磁場領域のカットオフ (B < 0.02 B0 は表示しない)
LOW_B_CUTOFF = 0.02

# カラーバーのコントラスト調整 (90%くらいまで下げて、中心構造を際立たせる)
VMAX_PERCENTILE = 90.0

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'heating_plots_clean')
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
                stripped = line.strip()
                parts = stripped.split()
                if stripped.startswith('dx, dt, c'):
                    DT, C_LIGHT = float(parts[5]), float(parts[6])
                elif stripped.startswith('Mi, Me'):
                    MI = float(parts[3])
                elif stripped.startswith('Qi, Qe'):
                    QI = float(parts[3])
                elif stripped.startswith('Fpe, Fge, Fpi Fgi'):
                    FPI, FGI = float(parts[7]), float(parts[8])
                elif stripped.startswith('Va, Vi, Ve'):
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

# 前処理スムージング付きロード関数
def load_smooth_data(timestep, subdir, prefix, suffix):
    filename = f'{prefix}_{timestep}_{suffix}.txt'
    filepath = os.path.join(subdir, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        # 強い前処理スムージング
        return gaussian_filter(data, sigma=PRE_SMOOTH_SIGMA)
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def gradient_2d(f, dx):
    grad_y, grad_x = np.gradient(f, dx, edge_order=2)
    return grad_x, grad_y

def calculate_gca_heating(timestep):
    # 1. Load & Pre-Smooth
    Bx = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bx') / B0
    By = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'By') / B0
    Bz = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Bz') / B0
    Ex = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ex') / B0
    Ey = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ey') / B0
    Ez = load_smooth_data(timestep, FIELD_DATA_DIR, 'data', 'Ez') / B0
    
    ne = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_density_count')
    ni = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_density_count')
    
    Vxe = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vx') / VA0
    Vye = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vy') / VA0
    Vze = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'electron_Vz') / VA0
    Vxi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vx') / VA0
    Vyi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vy') / VA0
    Vzi = load_smooth_data(timestep, MOMENT_DATA_DIR, 'data', 'ion_Vz') / VA0

    def load_tensor(ts):
        comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
        tensor = {}
        path_check = os.path.join(MOMENT_DATA_DIR, f'data_{ts}_electron_Txx.txt')
        if os.path.exists(path_check):
            for c in comps:
                raw = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', f'electron_T{c}')
                tensor[c] = raw / (VA0**2)
        else:
            raw_T = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', 'electron_T')
            T_iso = raw_T / (VA0**2)
            tensor = {c: (np.zeros_like(T_iso) if 'x' in c and 'y' in c else T_iso) for c in comps}
        return tensor
    Te_tensor = load_tensor(timestep)
    
    # --- Calculation ---
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    valid_mask = B_mag > LOW_B_CUTOFF
    B_safe = B_mag.copy()
    B_safe[~valid_mask] = 1.0 
    
    bx, by, bz = Bx/B_safe, By/B_safe, Bz/B_safe
    
    N_tot = (ne + ni) / 2.0
    avg_N = np.mean(N_tot[N_tot > 0.1]) if np.any(N_tot > 0.1) else 1.0
    n_proxy = N_tot / avg_N
    
    Jx = n_proxy * (Vxi - Vxe)
    Jy = n_proxy * (Vyi - Vye)
    Jz = n_proxy * (Vzi - Vze)
    
    # Parallel
    E_par = Ex*bx + Ey*by + Ez*bz
    J_par = Jx*bx + Jy*by + Jz*bz
    ue_par = Vxe*bx + Vye*by + Vze*bz
    
    # Drifts
    uE_x = (Ey*Bz - Ez*By) / (B_safe**2)
    uE_y = (Ez*Bx - Ex*Bz) / (B_safe**2)
    uE_z = (Ex*By - Ey*Bx) / (B_safe**2)
    
    # Curvature (Vector Gradient)
    dbx_dx, dbx_dy = gradient_2d(bx, DX_PHYS)
    dby_dx, dby_dy = gradient_2d(by, DX_PHYS)
    dbz_dx, dbz_dy = gradient_2d(bz, DX_PHYS)
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy
    
    # Pressure
    Pxx, Pyy, Pzz = n_proxy*Te_tensor['xx'], n_proxy*Te_tensor['yy'], n_proxy*Te_tensor['zz']
    Pxy, Pyz, Pxz = n_proxy*Te_tensor['xy'], n_proxy*Te_tensor['yz'], n_proxy*Te_tensor['xz']
    p_par = (bx**2*Pxx + by**2*Pyy + bz**2*Pzz + 2*bx*by*Pxy + 2*by*bz*Pyz + 2*bx*bz*Pxz)
    trace_P = Pxx + Pyy + Pzz
    p_perp = (trace_P - p_par) / 2.0
    
    # Heating Terms
    term_E_par = E_par * J_par
    
    curvature_drive = uE_x*kappa_x + uE_y*kappa_y + uE_z*kappa_z
    term_Curvature = (p_par + n_proxy * ue_par**2) * curvature_drive
    
    gradB_x, gradB_y = gradient_2d(B_mag, DX_PHYS)
    gradB_drive = uE_x * gradB_x + uE_y * gradB_y
    term_GradB = (p_perp / B_safe) * gradB_drive
    
    term_Total = term_E_par + term_Curvature + term_GradB

    # --- 2. Post-Smoothing (計算結果をさらに均す) ---
    # マスク適用前にスムージングを行うことで、マスク境界のアーティファクトを防ぐ
    term_Total = gaussian_filter(term_Total, sigma=POST_SMOOTH_SIGMA)
    term_E_par = gaussian_filter(term_E_par, sigma=POST_SMOOTH_SIGMA)
    term_Curvature = gaussian_filter(term_Curvature, sigma=POST_SMOOTH_SIGMA)
    term_GradB = gaussian_filter(term_GradB, sigma=POST_SMOOTH_SIGMA)
    
    # --- Masking (Display purposes) ---
    # マスク領域は NaN にして、プロット時に色を抜く
    term_Total[~valid_mask] = np.nan
    term_E_par[~valid_mask] = np.nan
    term_Curvature[~valid_mask] = np.nan
    term_GradB[~valid_mask] = np.nan
    
    return term_Total, term_E_par, term_Curvature, term_GradB, Bx, By

# =======================================================
# PLOTTING
# =======================================================
def plot_heating_clean(timestep):
    print(f"\n--- Processing timestep: {timestep} ---")
    
    total, e_par, curve, grad_b, Bx, By = calculate_gca_heating(timestep)
    
    x_phys = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS) / DI
    y_phys = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / DI
    X, Y = np.meshgrid(x_phys, y_phys)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    
    plot_configs = [
        (total, r'Total Heating $dU/dt$', axs[0]),
        (e_par, r'Parallel E-Field Work: $E_\parallel J_\parallel$', axs[1]),
        (curve, r'Curvature Drift Heating', axs[2]),
        (grad_b, r'Grad-B Drift Heating', axs[3])
    ]
    
    omega_t = float(timestep) * DT * FGI
    
    for data, title, ax in plot_configs:
        # 背景色をグレーに設定（マスク領域用）
        ax.set_facecolor('#d3d3d3')
        
        # スケール設定 (NaNを除外して計算)
        valid_data = data[~np.isnan(data)]
        abs_data = np.abs(valid_data)
        
        if len(valid_data) > 0:
            # 90パーセンタイルで厳しめにカット（ノイズを無視）
            max_val = np.percentile(abs_data, VMAX_PERCENTILE)
        else:
            max_val = 1.0
        
        # 完全にゼロに近い場合の保護
        if max_val < 1e-12: max_val = 1e-12
            
        vmin, vmax = -max_val, max_val
        
        cf = ax.contourf(X, Y, data, levels=100, cmap='seismic', vmin=vmin, vmax=vmax, extend='both')
        
        stride_x = max(1, Bx.shape[1] // 20)
        stride_y = max(1, Bx.shape[0] // 20)
        ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                      Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                      color='black', linewidth=0.5, density=0.8, arrowstyle='-')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x/d_i$', fontsize=12)
        ax.set_ylabel('$y/d_i$', fontsize=12)
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.95, format='%.1e')
        cbar.set_label('Heating Rate (Arb. Units)', fontsize=10)

    fig.suptitle(rf"Electron Heating (Double-Smoothed) - Timestep {timestep} ($\Omega_{{ci}}t={omega_t:.1f}$)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, f'heating_clean_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visual_heating_doublesmooth.py [start] [end] [interval]")
    else:
        s, e, i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        curr = s
        while curr <= e:
            try:
                plot_heating_clean(f"{curr:06d}")
            except Exception as ex:
                print(f"Error on {curr}: {ex}")
            curr += i