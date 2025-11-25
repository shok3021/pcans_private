import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# 1. 設定パラメータ (ここを変えれば全挙動が変わります)
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
X_MIN, X_MAX = 0.0, NX * 1.0
Y_MIN, Y_MAX = 0.0, NY * 1.0

# ★ 分離ロジックのパラメータ
FIT_CUTOFF_RATIO = 3.0  
MAX_ITER = 10           
ALPHA_THRESHOLD = 10.0  

# プロット設定
HIST_BINS = 100
PLOT_E_RANGE = (0.0, 200.0)

# =======================================================
# 2. 共通計算ロジック (メイン解析と完全一致)
# =======================================================
def get_particle_data(fpath):
    """ファイル読み込み & エネルギー計算"""
    try:
        data = np.loadtxt(fpath)
        if data.size == 0: return None, None, None
        if data.ndim == 1: data = data.reshape(1, -1)
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
        return None, None, None

    X_pos = data[:, 0]
    Y_pos = data[:, 1]
    # 速度 -> エネルギー変換
    v_sq = data[:, 2]**2 + data[:, 3]**2 + data[:, 4]**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J
    
    return X_pos, Y_pos, E_kin_keV

def separate_thermal_nonthermal(X_pos, Y_pos, E_kin_keV):
    """
    反復マクスウェルフィッティングによる分離
    """
    # グリッド準備
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # --- 反復計算 ---
    mask_fitting = np.ones(len(E_kin_keV), dtype=bool)
    T_core_map = np.zeros((NY, NX))

    for i in range(MAX_ITER):
        # 候補粒子のみ抽出
        curr_X = X_pos[mask_fitting]
        curr_Y = Y_pos[mask_fitting]
        curr_E = E_kin_keV[mask_fitting]
        
        if len(curr_E) == 0: break

        # グリッドごとの平均エネルギー算出
        H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges])
        H_sum_E, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges], weights=curr_E)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Mean_E = H_sum_E / H_count
            Mean_E[H_count == 0] = 0.0
        
        # 温度推定 T = (2/3)<E>
        T_iter = (2.0 / 3.0) * Mean_E
        T_iter = np.nan_to_num(T_iter, nan=0.0)
        T_core_map = T_iter.copy()
        
        # フィルタリング (T_coreの3倍以上を除外して再計算)
        T_particle = T_core_map[iy, ix]
        safe_T = T_particle.copy()
        safe_T[safe_T < 1e-9] = 1.0 # ゼロ除算回避
        
        ratio = E_kin_keV / safe_T
        ratio[T_particle < 1e-9] = 0.0
        
        mask_fitting = (ratio <= FIT_CUTOFF_RATIO)
    
    # --- 最終分類判定 ---
    T_final = T_core_map[iy, ix]
    
    # 閾値判定 (10倍温度)
    threshold = ALPHA_THRESHOLD * T_final
    valid_T = (T_final > 1e-9)
    
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold[valid_T])
    
    return ~is_nonthermal, is_nonthermal # Thermal, NonThermal

# =======================================================
# 3. 理論曲線
# =======================================================
def maxwellian_func(E, kT, N_total, dE):
    """ Eにおけるカウント数 (理論値) """
    if kT <= 0: return np.zeros_like(E)
    # f(E) = 2*sqrt(E/pi)*(1/kT)^1.5 * exp(-E/kT)
    f_E = 2.0 * np.sqrt(E/np.pi) * (1.0/kT)**1.5 * np.exp(-E/kT)
    return f_E * N_total * dE

# =======================================================
# 4. メイン処理
# =======================================================
def process_timestep(ts, fpath, output_dir):
    print(f"TS: {ts} loading...", end=" ")
    X, Y, E = get_particle_data(fpath)
    if X is None:
        print("Skipped.")
        return

    # 分離実行
    print("Separating...", end=" ")
    mask_T, mask_NT = separate_thermal_nonthermal(X, Y, E)
    
    E_thermal = E[mask_T]
    E_nonthermal = E[mask_NT]
    
    # 統計量 (全グリッド平均)
    N_T = len(E_thermal)
    Mean_T = np.mean(E_thermal) if N_T > 0 else 0
    kT_eff = (2.0/3.0) * Mean_T  # Thermal成分全体の実効温度
    
    print(f"Done. (N_T={N_T}, kT_eff={kT_eff:.2f} keV)")

    # --- プロット作成 ---
    plt.figure(figsize=(10, 7))
    
    # ビン設定
    bins = np.linspace(PLOT_E_RANGE[0], PLOT_E_RANGE[1], HIST_BINS)
    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # 1. Thermal (青)
    count_T, _ = np.histogram(E_thermal, bins=bins)
    err_T = np.sqrt(count_T) # ポアソン誤差 (エラーバー)
    
    # 0カウントはプロットしない(ログスケール対策)
    mask_plot_T = count_T > 0
    plt.errorbar(bin_centers[mask_plot_T], count_T[mask_plot_T], yerr=err_T[mask_plot_T], 
                 fmt='o', markersize=4, capsize=3, elinewidth=1,
                 color='blue', alpha=0.8, label=f'Thermal (Extracted)\nSum of all grids')

    # 2. Non-Thermal (赤)
    count_NT, _ = np.histogram(E_nonthermal, bins=bins)
    err_NT = np.sqrt(count_NT)
    
    mask_plot_NT = count_NT > 0
    plt.errorbar(bin_centers[mask_plot_NT], count_NT[mask_plot_NT], yerr=err_NT[mask_plot_NT], 
                 fmt='x', markersize=4, capsize=3, elinewidth=1,
                 color='red', alpha=0.8, label='Non-Thermal (Extracted)')

    # 3. 理論線 (黒破線) - 平均温度を使用
    y_theory = maxwellian_func(bin_centers, kT_eff, N_T, bin_width)
    plt.plot(bin_centers, y_theory, 'k--', linewidth=2, label=f'Maxwellian Fit\n(kT_avg = {kT_eff:.2f} keV)')

    # 装飾
    plt.yscale('log')
    plt.xlabel('Kinetic Energy [keV]')
    plt.ylabel('Count / bin')
    plt.title(f'Distribution Verification (TS: {ts})\nGlobal Aggregation of All Grids')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper right')
    
    # 保存
    plt.savefig(os.path.join(output_dir, f'dist_{ts}.png'), dpi=120)
    plt.close()

def main():
    if len(sys.argv) < 4:
        print("Usage: python verify_dist.py [start] [end] [step]")
        sys.exit(1)

    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # ★ データパス設定
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    
    # 出力フォルダ
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'verification_plots_final')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Checking data in: {DATA_DIR}")
    print(f"Output to: {OUTPUT_DIR}")

    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        fpath = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_e.dat')
        
        if os.path.exists(fpath):
            process_timestep(ts, fpath, OUTPUT_DIR)
        else:
            # print(f"File not found: {ts}") 
            pass

if __name__ == "__main__":
    main()