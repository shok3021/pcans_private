import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# 設定
# =======================================================
BASE_DIR = os.path.abspath('/home/shok/pcans') 
FIELD_DIR = os.path.join(BASE_DIR, 'python/extracted_data')
PSD_DIR = os.path.join(BASE_DIR, 'em2d_mpi/md_mrx/psd')
OUTPUT_DIR = os.path.join(BASE_DIR, 'python/fluid_gca_results') # 出力先を変更
os.makedirs(OUTPUT_DIR, exist_ok=True)

# グリッド設定
NX = 320
NY = 639
DELX = 1.0 

# 物理定数 (正規化単位系)
CHARGE_E = -1.0
MASS_E = 1.0

# =======================================================
# ヘルパー関数
# =======================================================
def load_field(timestep, field_name):
    path = os.path.join(FIELD_DIR, f'data_{timestep}_{field_name}.txt')
    try:
        data = np.loadtxt(path, delimiter=',')
        return gaussian_filter(data, sigma=1.0)
    except Exception as e:
        print(f"Error loading {field_name}: {e}")
        return np.zeros((NY, NX))

def gradient_2d(f, dx):
    grad_y, grad_x = np.gradient(f, dx, edge_order=2)
    return grad_x, grad_y

def calculate_fluid_terms(timestep):
    print(f"--- Processing Timestep (Fluid GCA): {timestep} ---")

    # 1. 電磁場データの読み込みと計算
    Bx = load_field(timestep, 'Bx')
    By = load_field(timestep, 'By')
    Bz = load_field(timestep, 'Bz')
    Ex = load_field(timestep, 'Ex')
    Ey = load_field(timestep, 'Ey')
    Ez = load_field(timestep, 'Ez')

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mag[B_mag < 1e-4] = 1.0

    # 単位ベクトル b
    bx, by, bz = Bx/B_mag, By/B_mag, Bz/B_mag

    # 曲率 kappa = (b.nabla)b
    dbx_dx, dbx_dy = gradient_2d(bx, DELX)
    dby_dx, dby_dy = gradient_2d(by, DELX)
    dbz_dx, dbz_dy = gradient_2d(bz, DELX)
    
    # 2次元シミュレーション(d/dz=0)を仮定
    kappa_x = bx * dbx_dx + by * dbx_dy
    kappa_y = bx * dby_dx + by * dby_dy
    kappa_z = bx * dbz_dx + by * dbz_dy

    # grad B
    gradB_x, gradB_y = gradient_2d(B_mag, DELX)

    # ExB Drift velocity: uE = (ExB) / B^2
    uE_x = (Ey*Bz - Ez*By) / (B_mag**2)
    uE_y = (Ez*Bx - Ex*Bz) / (B_mag**2)
    uE_z = (Ex*By - Ey*Bx) / (B_mag**2)

    # Parallel E-field: E_par
    E_par = Ex*bx + Ey*by + Ez*bz

    # 2. 粒子データから流体モーメントを作成
    psd_file = os.path.join(PSD_DIR, f'{timestep}_0160-0320_psd_e.dat')
    if not os.path.exists(psd_file):
        print(f"  PSD file not found: {psd_file}")
        return

    try:
        pdata = np.loadtxt(psd_file)
    except:
        return

    if pdata.shape[0] == 0: return

    px = pdata[:, 0]
    py = pdata[:, 1]
    p_ux = pdata[:, 2]
    p_uy = pdata[:, 3]
    p_uz = pdata[:, 4]

    # 有効範囲のフィルタリング
    mask = (px >= 0) & (px < NX*DELX) & (py >= 0) & (py < NY*DELX)
    px = px[mask]
    py = py[mask]
    p_ux = p_ux[mask]
    p_uy = p_uy[mask]
    p_uz = p_uz[mask]

    # グリッド位置の特定
    ix = (px / DELX).astype(int)
    iy = (py / DELX).astype(int)
    ix = np.clip(ix, 0, NX-1)
    iy = np.clip(iy, 0, NY-1)

    # --- 粒子の速度分解 ---
    # 粒子位置での磁場単位ベクトルを取得
    p_bx = bx[iy, ix]
    p_by = by[iy, ix]
    p_bz = bz[iy, ix]

    # Gamma factor
    gamma = np.sqrt(1.0 + p_ux**2 + p_uy**2 + p_uz**2)
    p_vx = p_ux / gamma
    p_vy = p_uy / gamma
    p_vz = p_uz / gamma

    # Parallel Velocity
    v_par = p_vx*p_bx + p_vy*p_by + p_vz*p_bz
    
    # Perpendicular Speed Squared
    v_sq = p_vx**2 + p_vy**2 + p_vz**2
    v_perp_sq = np.maximum(v_sq - v_par**2, 0.0)

    # --- モーメントの集計 (Deposit) ---
    print("  Depositing moments to grid...")
    
    # ヒストグラム用のエッジ
    x_edges = np.linspace(0, NX*DELX, NX+1)
    y_edges = np.linspace(0, NY*DELX, NY+1)

    # 重み (Weights) の定義
    # 1. Parallel Current Weight: J_par ~ q * v_par
    w_J_par = CHARGE_E * v_par

    # 2. Perpendicular Pressure Weight: p_perp ~ 0.5 * m * v_perp^2
    #    (注: 分布関数fに対する積分なので、粒子の総和を取れば密度込みの圧力になります)
    w_P_perp = 0.5 * MASS_E * v_perp_sq

    # 3. Parallel Stress Weight: (p_par + m*n*u_par^2) ~ m * v_par^2
    #    式(5)の第3項の係数は「全平行運動量フラックス」に相当します
    w_Stress_par = MASS_E * (v_par**2)

    # ヒストグラム作成 (合計値)
    # 単位体積あたりにするため、最後にセルの面積(体積)で割る必要がある場合がありますが、
    # ここでは「そのセル内の総量」として扱い、相対比較できるようにします。
    # 厳密な値を求めるなら /(DELX*DELX) などをしてください。
    
    grid_J_par, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=w_J_par)
    grid_P_perp, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=w_P_perp)
    grid_Stress_par, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges], weights=w_Stress_par)
    
    # 粒子数密度(参考用)
    grid_N, _, _ = np.histogram2d(py, px, bins=[y_edges, x_edges])

    # =======================================================
    # ★★★ 式(5) の項別計算 (Grid演算) ★★★
    # =======================================================
    print("  Calculating Eq(5) terms...")

    # Term 1: Parallel Acceleration
    # = E_par * J_par
    term1 = E_par * grid_J_par

    # Term 2: Betatron (Perpendicular) Heating
    # = (p_perp / B) * (uE . gradB)
    # ※ dB/dt の項はここでは省略 (データがないため)
    uE_dot_gradB = uE_x * gradB_x + uE_y * gradB_y
    term2 = (grid_P_perp / B_mag) * uE_dot_gradB

    # Term 3: Fermi Acceleration
    # = (p_par + m*n*u_par^2) * (uE . kappa)
    # = grid_Stress_par * (uE . kappa)
    uE_dot_kappa = uE_x * kappa_x + uE_y * kappa_y + uE_z * kappa_z
    term3 = grid_Stress_par * uE_dot_kappa

    # Total
    term_total = term1 + term2 + term3

    # =======================================================
    # 保存
    # =======================================================
    np.savetxt(os.path.join(OUTPUT_DIR, f'fluid_term1_par_{timestep}.txt'), term1, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'fluid_term2_betatron_{timestep}.txt'), term2, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'fluid_term3_fermi_{timestep}.txt'), term3, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'fluid_total_{timestep}.txt'), term_total, fmt='%.5e')
    
    # 参考用モーメント
    np.savetxt(os.path.join(OUTPUT_DIR, f'moment_J_par_{timestep}.txt'), grid_J_par, fmt='%.5e')
    np.savetxt(os.path.join(OUTPUT_DIR, f'moment_P_perp_{timestep}.txt'), grid_P_perp, fmt='%.5e')

    print(f"  Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # 使用例: 15000ステップのみ計算
    if len(sys.argv) < 2:
        calculate_fluid_terms("015000")
    else:
        # コマンドライン引数処理 (start end step)
        s = int(sys.argv[1])
        e = int(sys.argv[2])
        st = int(sys.argv[3])
        for ts in range(s, e+st, st):
            calculate_fluid_terms(f"{ts:06d}")