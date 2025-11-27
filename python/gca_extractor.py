import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# 設定 (ユーザー環境に合わせてパスを変更してください)
# =======================================================
# パス設定
BASE_DIR = os.path.abspath('.') # 実行場所
FIELD_DIR = os.path.join(BASE_DIR, 'extracted_data') # 電磁場データの場所
PSD_DIR = os.path.join(BASE_DIR, 'psd') # 粒子データの場所
OUTPUT_DIR = os.path.join(BASE_DIR, 'particle_gca_results') # 出力先
os.makedirs(OUTPUT_DIR, exist_ok=True)

# グリッド設定
NX = 320
NY = 639
DELX = 1.0  # グリッド幅

# 物理定数 (正規化単位系を想定)
# 電子の場合 q = -1, m = 1 として計算します
CHARGE_E = -1.0
MASS_E = 1.0

# =======================================================
# ヘルパー関数
# =======================================================
def load_field(timestep, field_name):
    """電磁場データを読み込み (平滑化済みを想定)"""
    path = os.path.join(FIELD_DIR, f'data_{timestep}_{field_name}.txt')
    try:
        data = np.loadtxt(path, delimiter=',')
        # 必要に応じてスムージング (微分の安定化のため)
        return gaussian_filter(data, sigma=1.0)
    except Exception as e:
        print(f"Error loading {field_name}: {e}")
        return np.zeros((NY, NX))

def gradient_2d(f, dx):
    """2次元勾配 (y, x)"""
    grad_y, grad_x = np.gradient(f, dx, edge_order=2)
    return grad_x, grad_y

def calculate_particle_terms(timestep):
    print(f"--- Processing Timestep: {timestep} ---")

    # 1. 電磁場データの読み込み
    Bx = load_field(timestep, 'Bx')
    By = load_field(timestep, 'By')
    Bz = load_field(timestep, 'Bz')
    Ex = load_field(timestep, 'Ex')
    Ey = load_field(timestep, 'Ey')
    Ez = load_field(timestep, 'Ez')

    # 2. 場の量の事前計算 (Grid上)
    print("  Calculating Field Derivatives...")
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mag[B_mag < 1e-4] = 1.0 # ゼロ除算防止

    # 単位ベクトル b
    bx, by, bz = Bx/B_mag, By/B_mag, Bz/B_mag

    # 曲率 kappa = (b.nabla)b
    dbx_dx, dbx_dy = gradient_2d(bx, DELX)
    dby_dx, dby_dy = gradient_2d(by, DELX)
    dbz_dx, dbz_dy = gradient_2d(bz, DELX)
    
    # kappa_x = bx * dbx/dx + by * dbx/dy (2D仮定: d/dz=0)
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy

    # grad B
    gradB_x, gradB_y = gradient_2d(B_mag, DELX)

    # ExB Drift velocity: uE = (ExB) / B^2
    uE_x = (Ey*Bz - Ez*By) / (B_mag**2)
    uE_y = (Ez*Bx - Ex*Bz) / (B_mag**2)
    uE_z = (Ex*By - Ey*Bx) / (B_mag**2)

    # 3. 粒子データの読み込み
    # 形式: [x, y, ux, uy, uz]
    psd_file = os.path.join(PSD_DIR, f'{timestep}_0160-0320_psd_e.dat')
    if not os.path.exists(psd_file):
        print(f"  PSD file not found: {psd_file}")
        return

    print("  Loading Particle Data...")
    try:
        pdata = np.loadtxt(psd_file)
    except:
        print("  Failed to load particle data.")
        return

    if pdata.shape[0] == 0: return

    # 4. 粒子ループ計算 (Vectorized for speed)
    print(f"  Computing GCA terms for {pdata.shape[0]} particles...")
    
    px = pdata[:, 0]
    py = pdata[:, 1]
    p_ux = pdata[:, 2]
    p_uy = pdata[:, 3]
    p_uz = pdata[:, 4]

    # グリッドインデックスへのマッピング (最近傍)
    # 範囲外の粒子を除外
    valid_mask = (px >= 0) & (px < NX*DELX) & (py >= 0) & (py < NY*DELX)
    
    px = px[valid_mask]
    py = py[valid_mask]
    p_ux = p_ux[valid_mask]
    p_uy = p_uy[valid_mask]
    p_uz = p_uz[valid_mask]

    ix = (px / DELX).astype(int)
    iy = (py / DELX).astype(int)
    # インデックスのクリップ
    ix = np.clip(ix, 0, NX-1)
    iy = np.clip(iy, 0, NY-1)

    # --- 粒子位置での場の値を取得 ---
    # NumpyのAdvanced Indexingを使用
    p_bx = bx[iy, ix]
    p_by = by[iy, ix]
    p_bz = bz[iy, ix]
    p_Ex = Ex[iy, ix]
    p_Ey = Ey[iy, ix]
    p_Ez = Ez[iy, ix]
    p_Bmag = B_mag[iy, ix]
    
    p_kappa_x = kappa_x[iy, ix]
    p_kappa_y = kappa_y[iy, ix]
    p_kappa_z = kappa_z[iy, ix]
    
    p_gradB_x = gradB_x[iy, ix]
    p_gradB_y = gradB_y[iy, ix]
    
    p_uE_x = uE_x[iy, ix]
    p_uE_y = uE_y[iy, ix]
    p_uE_z = uE_z[iy, ix]

    # --- 粒子運動変数の計算 ---
    # Gamma factor
    gamma = np.sqrt(1.0 + p_ux**2 + p_uy**2 + p_uz**2)
    
    # Velocity v = u / gamma
    p_vx = p_ux / gamma
    p_vy = p_uy / gamma
    p_vz = p_uz / gamma

    # Parallel velocity: v_par = v . b
    v_par = p_vx*p_bx + p_vy*p_by + p_vz*p_bz
    
    # Perpendicular velocity squared: v_perp^2 = v^2 - v_par^2
    v_sq = p_vx**2 + p_vy**2 + p_vz**2
    v_perp_sq = v_sq - v_par**2
    v_perp_sq = np.maximum(v_perp_sq, 0.0) # 数値誤差対策

    # Parallel E-field
    E_par = p_Ex*p_bx + p_Ey*p_by + p_Ez*p_bz

    # =======================================================
    # ★★★ 式(5) に基づく項別計算 ★★★
    # =======================================================
    
    # Term 1: Parallel Electric Field Acceleration
    # W_par = q * E_par * v_par
    term1 = CHARGE_E * E_par * v_par

    # Term 2: Betatron Acceleration (Grad-B drift direction)
    # Eq: (mu / gamma) * (uE . gradB)
    # mu = m * gamma^2 * v_perp^2 / (2B)
    # -> mu / gamma = m * gamma * v_perp^2 / (2B)
    coeff_betatron = (MASS_E * gamma * v_perp_sq) / (2.0 * p_Bmag)
    drive_betatron = p_uE_x * p_gradB_x + p_uE_y * p_gradB_y
    term2 = coeff_betatron * drive_betatron

    # Term 3: Fermi Acceleration (Curvature drift direction)
    # Eq: gamma * m * v_par^2 * (uE . kappa)
    coeff_fermi = gamma * MASS_E * (v_par**2)
    drive_fermi = p_uE_x * p_kappa_x + p_uE_y * p_kappa_y + p_uE_z * p_kappa_z
    term3 = coeff_fermi * drive_fermi

    # Total
    term_total = term1 + term2 + term3

    # =======================================================
    # グリッドへの集計 (Deposit)
    # =======================================================
    # ここでは「その場所での加速の合計(Power Density)」を計算します
    print("  Binning results to grid...")

    grid_term1 = np.zeros((NY, NX))
    grid_term2 = np.zeros((NY, NX))
    grid_term3 = np.zeros((NY, NX))
    grid_total = np.zeros((NY, NX))
    grid_count = np.zeros((NY, NX)) # 粒子数（平均化用）

    # 高速化のためにhistogram2dを使用 (y, xの順に注意)
    # binsのエッジ定義
    x_edges = np.linspace(0, NX*DELX, NX+1)
    y_edges = np.linspace(0, NY*DELX, NY+1)

    # 重み付きヒストグラム作成
    H_t1, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=term1)
    H_t2, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=term2)
    H_t3, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=term3)
    H_tot, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=term_total)
    H_cnt, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges]) # 粒子数

    # 出力データの保存
    # 生の合計値（Power Density ~ J.E）か、粒子あたりの平均値か選べますが、
    # 加速の「強さ」を見るなら合計値（密度×加速）が一般的です。
    # ここでは見やすくするため H_tot (Power Density相当) を保存します。
    
    np.savetxt(os.path.join(OUTPUT_DIR, f'gca_term1_par_{timestep}.txt'), H_t1, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'gca_term2_betatron_{timestep}.txt'), H_t2, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'gca_term3_fermi_{timestep}.txt'), H_t3, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'gca_total_{timestep}.txt'), H_tot, fmt='%.5e')
    
    # 磁場データもプロット用にコピー保存
    np.savetxt(os.path.join(OUTPUT_DIR, f'field_Bx_{timestep}.txt'), Bx, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'field_By_{timestep}.txt'), By, fmt='%.5e')

    print(f"  Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_particle_gca.py [timestep_start] [timestep_end] [step]")
        # デフォルト動作（例）
        calculate_particle_terms("015000")
    else:
        s = int(sys.argv[1])
        e = int(sys.argv[2])
        st = int(sys.argv[3])
        for ts in range(s, e+st, st):
            calculate_particle_terms(f"{ts:06d}")