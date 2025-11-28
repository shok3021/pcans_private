import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# 設定 (提示されたコードの設定を再現)
# =======================================================
BINNING_FACTOR = 4         # 4x4の平均化
PRE_SMOOTH_SIGMA = 2.0     # 前処理スムージング
VECTOR_SMOOTH_SIGMA = 1.0  # ベクトル場スムージング
POST_SMOOTH_SIGMA = 1.0    # 結果スムージング
LOW_B_CUTOFF = 0.02
SIMULATION_BETA_BG = 0.125 # 背景プラズマベータ

# パス設定 (環境に合わせて変更してください)
BASE_DIR = os.path.abspath('/home/shok/pcans') 
FIELD_DATA_DIR = os.path.join(BASE_DIR, 'python/extracted_data')
MOMENT_DATA_DIR = os.path.join(BASE_DIR, 'python/extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/gca_bulk_data')
PARAM_FILE_PATH = os.path.join(BASE_DIR, 'em2d_mpi/md_mrx/dat/init_param.dat')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# グリッド設定 (高解像度)
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# =======================================================
# ヘルパー関数
# =======================================================
def load_simulation_parameters(param_filepath):
    """パラメータファイルから規格化定数を取得"""
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = None, None, None, None, None, None, None
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if 'dx,' in line: DT = float(parts[5]); C_LIGHT = float(parts[6])
                elif 'Mi,' in line: MI = float(parts[3])
                elif 'Qi,' in line: QI = float(parts[3])
                elif 'Fpe,' in line: FPI = float(parts[7]); FGI = float(parts[8])
                elif 'Va,' in line: VA0 = float(parts[7])
    except: pass
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# パラメータ読み込み
C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
if None in [C_LIGHT, FPI, DT, FGI, VA0, MI, QI]:
    print("Warning: Using default normalization parameters.")
    DT, FGI, VA0, DI, B0 = 0.02, 0.04, 0.1, 100.0, 1.0
else:
    DI = C_LIGHT / FPI
    B0 = (FGI * MI * C_LIGHT) / QI

# ビニング後のグリッド幅
DX_PHYS = DELX / DI
DX_PHYS_COARSE = DX_PHYS * BINNING_FACTOR

def load_smooth_data(timestep, subdir, prefix, suffix):
    """データの読み込みと前処理スムージング"""
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
    """ビニング (Coarse Graining)"""
    if factor <= 1: return arr
    ny, nx = arr.shape
    new_ny, new_nx = ny // factor, nx // factor
    arr_trimmed = arr[:new_ny*factor, :new_nx*factor]
    return arr_trimmed.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))

def gradient_2d(f, dx):
    """2次元勾配"""
    grad_y, grad_x = np.gradient(f, dx, edge_order=2)
    return grad_x, grad_y

def calculate_and_save_heating(timestep):
    print(f"--- Processing Timestep: {timestep} ---")

    # 1. データ読み込み (Pre-Smoothing込み)
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

    # 2. ビニング (Coarse Graining)
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

    # 密度プロキシ
    N_tot = (ne + ni) / 2.0
    avg_N = np.mean(N_tot[N_tot > 0.1]) if np.any(N_tot > 0.1) else 1.0
    n_proxy = N_tot / avg_N

    # 3. 圧力テンソルの読み込みと規格化
    def load_tensor_raw(ts):
        comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
        tensor = {}
        for c in comps:
            # ファイル名のパターンを確認してください (例: data_015000_electron_Txx.txt)
            raw = load_smooth_data(ts, MOMENT_DATA_DIR, 'data', f'electron_T{c}')
            tensor[c] = rebin(raw, BINNING_FACTOR)
        return tensor

    try:
        Te_tensor_raw = load_tensor_raw(timestep)
    except:
        print("  Error: Tensor data missing. Skipping.")
        return

    # ★★★ 物理的正規化 (ここが重要) ★★★
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    # 磁場強度 ~ 1.0 の領域を背景とする
    bg_mask = (B_mag > 0.9) & (B_mag < 1.1)
    if np.sum(bg_mask) < 10: bg_mask = np.ones_like(B_mag, dtype=bool)

    trace_T_raw = Te_tensor_raw['xx'] + Te_tensor_raw['yy'] + Te_tensor_raw['zz']
    P_raw_mean = np.nanmean(trace_T_raw[bg_mask]) / 3.0
    if P_raw_mean == 0 or np.isnan(P_raw_mean): P_raw_mean = 1.0

    # P_bg = beta * B^2 / 2
    P_target_bg = SIMULATION_BETA_BG * (1.0**2) / 2.0
    NORM_FACTOR = P_target_bg / P_raw_mean
    
    print(f"  [NORM] Raw P: {P_raw_mean:.3e} -> Target P: {P_target_bg:.3e} (Factor: {NORM_FACTOR:.3e})")
    
    # テンソルの正規化
    Te_tensor = {k: v * NORM_FACTOR for k, v in Te_tensor_raw.items()}

    # 4. GCA項の計算
    valid_mask = B_mag > LOW_B_CUTOFF
    B_safe = B_mag.copy()
    B_safe[~valid_mask] = 1.0 
    
    bx, by, bz = Bx/B_safe, By/B_safe, Bz/B_safe
    
    # 電流 J = n * (Vi - Ve)
    Jx = n_proxy * (Vxi - Vxe)
    Jy = n_proxy * (Vyi - Vye)
    Jz = n_proxy * (Vzi - Vze)
    
    E_par = Ex*bx + Ey*by + Ez*bz
    J_par = Jx*bx + Jy*by + Jz*bz
    ue_par = Vxe*bx + Vye*by + Vze*bz
    
    # 曲率と勾配 (Coarse Grid上での微分)
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
    
    # ベクトル量のスムージング (中間)
    uE_x = gaussian_filter(uE_x, sigma=VECTOR_SMOOTH_SIGMA)
    uE_y = gaussian_filter(uE_y, sigma=VECTOR_SMOOTH_SIGMA)
    uE_z = gaussian_filter(uE_z, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_x = gaussian_filter(kappa_x, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_y = gaussian_filter(kappa_y, sigma=VECTOR_SMOOTH_SIGMA)
    kappa_z = gaussian_filter(kappa_z, sigma=VECTOR_SMOOTH_SIGMA)
    gradB_x = gaussian_filter(gradB_x, sigma=VECTOR_SMOOTH_SIGMA)
    gradB_y = gaussian_filter(gradB_y, sigma=VECTOR_SMOOTH_SIGMA)

    # 加熱項の組み立て
    Pxx, Pyy, Pzz = Te_tensor['xx'], Te_tensor['yy'], Te_tensor['zz']
    Pxy, Pyz, Pxz = Te_tensor['xy'], Te_tensor['yz'], Te_tensor['xz']
    
    p_par = (bx**2*Pxx + by**2*Pyy + bz**2*Pzz + 2*bx*by*Pxy + 2*by*bz*Pyz + 2*bx*bz*Pxz)
    trace_P = Pxx + Pyy + Pzz
    p_perp = (trace_P - p_par) / 2.0

    # Term 1: E_par * J_par
    term_E_par = E_par * J_par
    
    # Term 2: Curvature (Fermi)
    curvature_drive = uE_x*kappa_x + uE_y*kappa_y + uE_z*kappa_z
    term_Curvature = (p_par + n_proxy * ue_par**2) * curvature_drive
    
    # Term 3: Grad-B (Betatron)
    gradB_drive = uE_x * gradB_x + uE_y * gradB_y
    term_GradB = (p_perp / B_safe) * gradB_drive
    
    term_Total = term_E_par + term_Curvature + term_GradB

    # 5. 結果のスムージング (事後) と保存
    term_Total = gaussian_filter(term_Total, sigma=POST_SMOOTH_SIGMA)
    term_E_par = gaussian_filter(term_E_par, sigma=POST_SMOOTH_SIGMA)
    term_Curvature = gaussian_filter(term_Curvature, sigma=POST_SMOOTH_SIGMA)
    term_GradB = gaussian_filter(term_GradB, sigma=POST_SMOOTH_SIGMA)
    
    # マスク適用 (ゼロ除算領域をNaNまたは0に)
    term_Total[~valid_mask] = 0.0
    
    # 保存 (テキスト形式, Coarse Gridサイズ)
    np.savetxt(os.path.join(OUTPUT_DIR, f'heat_total_{timestep}.txt'), term_Total, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'heat_epar_{timestep}.txt'), term_E_par, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'heat_curv_{timestep}.txt'), term_Curvature, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'heat_gradb_{timestep}.txt'), term_GradB, fmt='%.5e')
    
    # 磁場データも保存 (プロット用, Coarse Grid)
    np.savetxt(os.path.join(OUTPUT_DIR, f'field_Bx_{timestep}.txt'), Bx, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'field_By_{timestep}.txt'), By, fmt='%.5e')
    
    print(f"  Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_heating_accurate.py [start] [end] [step]")
        # デフォルト動作例
        calculate_and_save_heating("015000")
    else:
        s, e, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+1, step):
            calculate_and_save_heating(f"{ts:06d}")