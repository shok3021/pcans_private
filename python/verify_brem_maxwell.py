import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# ★ 設定 (元のコードと同じパラメータを使用)
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# フィッティングパラメータ
FIT_CUTOFF_RATIO = 3.0 
MAX_ITER = 10
ALPHA_THRESHOLD = 10.0 

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
DELX = 1.0
X_MIN, X_MAX = 0.0, NX * DELX
Y_MIN, Y_MAX = 0.0, NY * DELX

# ヒストグラム表示設定
HIST_BINS = 100            # ビンの数
PLOT_E_RANGE = (0.0, 200.0) # プロットするエネルギー範囲 (keV)

# =======================================================
# 1. 物理計算・分類ロジック
# =======================================================
def get_particle_data(fpath):
    """ファイル読み込みとエネルギー計算"""
    try:
        data = np.loadtxt(fpath)
        if data.size == 0: return None, None, None
        if data.ndim == 1: data = data.reshape(1, -1)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

    X_pos = data[:, 0]
    Y_pos = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]
    vz = data[:, 4]

    v_sq = vx**2 + vy**2 + vz**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J
    
    return X_pos, Y_pos, E_kin_keV

def classify_particles_iterative(X_pos, Y_pos, E_kin_keV):
    """
    反復的マクスウェルフィッティング
    """
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # --- 反復プロセス ---
    mask_fitting_candidate = np.ones(len(E_kin_keV), dtype=bool)
    T_core_map = np.zeros((NY, NX))

    # print(f"  -> Iterative fitting (Max {MAX_ITER})...") # ログ量削減のためコメントアウト
    for i in range(MAX_ITER):
        curr_X = X_pos[mask_fitting_candidate]
        curr_Y = Y_pos[mask_fitting_candidate]
        curr_E = E_kin_keV[mask_fitting_candidate]
        
        if len(curr_E) == 0: break

        H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges])
        H_sum_E, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges], weights=curr_E)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Mean_E = H_sum_E / H_count
            Mean_E[H_count == 0] = 0.0
        
        T_current_iter = (2.0 / 3.0) * Mean_E
        T_current_iter = np.nan_to_num(T_current_iter, nan=0.0)
        T_core_map = T_current_iter.copy()
        
        # フィルタリング
        T_particle = T_core_map[iy, ix]
        safe_T = T_particle.copy()
        safe_T[safe_T < 1e-9] = 1.0
        ratio = E_kin_keV / safe_T
        ratio[T_particle < 1e-9] = 0.0
        
        mask_fitting_candidate = (ratio <= FIT_CUTOFF_RATIO)
    
    # --- 最終判定 ---
    T_final = T_core_map[iy, ix]
    threshold_energy = ALPHA_THRESHOLD * T_final
    valid_T = (T_final > 1e-9)
    
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold_energy[valid_T])
    is_thermal = ~is_nonthermal
    
    return is_thermal, is_nonthermal

# =======================================================
# 2. 理論分布関数 (比較用)
# =======================================================
def maxwellian_pdf_energy(E, kT):
    """
    3次元マクスウェル分布 f(E)
    """
    if kT <= 0: return np.zeros_like(E)
    coeff = 2.0 * np.sqrt(E / np.pi) * (1.0 / kT)**(1.5)
    return coeff * np.exp(-E / kT)

# =======================================================
# 3. メイン検証処理 (1ステップ分)
# =======================================================
def verify_and_plot(fpath, timestep, output_dir):
    print(f"Processing TS: {timestep} ... ", end="", flush=True)
    
    X, Y, E_keV = get_particle_data(fpath)
    if X is None: 
        print("Skipped (No Data)")
        return

    # 分類実行
    mask_T, mask_NT = classify_particles_iterative(X, Y, E_keV)
    
    E_thermal = E_keV[mask_T]
    E_nonthermal = E_keV[mask_NT]
    
    # --- 統計量計算 ---
    stats = {}
    for label, arr in [("Thermal", E_thermal), ("Non-Thermal", E_nonthermal)]:
        if len(arr) > 0:
            stats[label] = {
                "N": len(arr),
                "Mean": np.mean(arr),
                "Std": np.std(arr),
                "Max": np.max(arr)
            }
        else:
            stats[label] = {"N": 0, "Mean": 0, "Std": 0, "Max": 0}

    # 実効温度 T_eff = (2/3) * <E>_thermal
    eff_kT = (2.0 / 3.0) * stats["Thermal"]["Mean"]
    
    print(f"Done. (kT={eff_kT:.2f} keV)")

    # --- プロット作成 ---
    plt.figure(figsize=(10, 7))
    
    bins = np.linspace(PLOT_E_RANGE[0], PLOT_E_RANGE[1], HIST_BINS)
    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # 1. Thermal
    counts_T, _ = np.histogram(E_thermal, bins=bins)
    err_T = np.sqrt(counts_T)
    counts_T = counts_T.astype(float)
    counts_T[counts_T==0] = np.nan
    
    plt.errorbar(bin_centers, counts_T, yerr=err_T, fmt='o', markersize=4, 
                 color='blue', alpha=0.7, label='Extracted Thermal Data')

    # 2. Non-Thermal
    counts_NT, _ = np.histogram(E_nonthermal, bins=bins)
    err_NT = np.sqrt(counts_NT)
    counts_NT = counts_NT.astype(float)
    counts_NT[counts_NT==0] = np.nan
    
    plt.errorbar(bin_centers, counts_NT, yerr=err_NT, fmt='x', markersize=4, 
                 color='red', alpha=0.7, label='Extracted Non-Thermal Data')

    # 3. 理論曲線 (Maxwellian)
    theory_curve = maxwellian_pdf_energy(bin_centers, eff_kT) * stats["Thermal"]["N"] * bin_width
    plt.plot(bin_centers, theory_curve, 'k--', linewidth=2, label=f'Maxwellian (kT={eff_kT:.2f} keV)')

    # --- グラフ装飾 ---
    plt.yscale('log') 
    plt.xlabel('Kinetic Energy [keV]', fontsize=12)
    plt.ylabel('Count / bin', fontsize=12)
    plt.title(f'Maxwellian Fit Check (TS: {timestep})\nGlobal kT: {eff_kT:.2f} keV', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    info_txt = (f"Thermal N: {stats['Thermal']['N']}\n"
                f"Mean: {stats['Thermal']['Mean']:.2f} keV\n"
                f"Non-Therm N: {stats['Non-Thermal']['N']}")
    plt.text(0.95, 0.95, info_txt, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    out_name = os.path.join(output_dir, f'dist_check_{timestep}.png')
    plt.savefig(out_name, dpi=100)
    plt.close()

# =======================================================
# メイン実行部
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python verify_dist_batch.py [start] [end] [step]")
        print("Example: python verify_dist_batch.py 0 14000 500")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = int(sys.argv[3])
    
    # -----------------------------------------------------
    # ★ パス設定 (環境に合わせてください)
    # -----------------------------------------------------
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    
    # 出力先はスクリプトのある場所にフォルダ作成
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'verification_plots_batch')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Batch Distribution Check: {start} -> {end} (step {step}) ---")
    print(f"Data Source: {DATA_DIR}")
    print(f"Output Dir:  {OUTPUT_DIR}")
    
    # ループ処理
    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        
        # ファイル名形式: {ts}_0160-0320_psd_e.dat
        fname = f'{ts}_0160-0320_psd_e.dat'
        fpath = os.path.join(DATA_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"TS: {ts} -> File not found. Skipping.")
            continue
            
        verify_and_plot(fpath, ts, OUTPUT_DIR)
        
    print("\nAll tasks finished.")

if __name__ == "__main__":
    main()