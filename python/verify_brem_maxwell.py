import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# 設定
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
X_MIN, X_MAX = 0.0, NX * 1.0
Y_MIN, Y_MAX = 0.0, NY * 1.0

# 分離パラメータ
FIT_CUTOFF_RATIO = 3.0  
MAX_ITER = 10           
ALPHA_THRESHOLD = 10.0  

# 局所チェックのためにまとめる範囲 (半径)
# 1グリッドだと粒子数が少なすぎてガタガタになるため、周辺数グリッドをまとめる
LOCAL_BOX_RADIUS = 12  # 中心から前後2グリッド (つまり 5x5 グリッドの範囲)

# =======================================================
# 計算ロジック (共通)
# =======================================================
def get_particle_data(fpath):
    try:
        data = np.loadtxt(fpath)
        if data.size == 0: return None, None, None
        if data.ndim == 1: data = data.reshape(1, -1)
    except: return None, None, None

    X_pos = data[:, 0]
    Y_pos = data[:, 1]
    v_sq = data[:, 2]**2 + data[:, 3]**2 + data[:, 4]**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J
    return X_pos, Y_pos, E_kin_keV

def perform_separation_and_get_map(X_pos, Y_pos, E_kin_keV):
    """
    分離を行い、同時に「温度マップ」も返す
    """
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    mask_fitting = np.ones(len(E_kin_keV), dtype=bool)
    T_core_map = np.zeros((NY, NX))

    for i in range(MAX_ITER):
        curr_X = X_pos[mask_fitting]
        curr_Y = Y_pos[mask_fitting]
        curr_E = E_kin_keV[mask_fitting]
        if len(curr_E) == 0: break

        H_count, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges])
        H_sum_E, _, _ = np.histogram2d(curr_Y, curr_X, bins=[y_edges, x_edges], weights=curr_E)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Mean_E = H_sum_E / H_count
            Mean_E[H_count == 0] = 0.0
        
        T_iter = (2.0 / 3.0) * Mean_E
        T_iter = np.nan_to_num(T_iter, nan=0.0)
        T_core_map = T_iter.copy()
        
        T_particle = T_core_map[iy, ix]
        safe_T = T_particle.copy()
        safe_T[safe_T < 1e-9] = 1.0
        ratio = E_kin_keV / safe_T
        ratio[T_particle < 1e-9] = 0.0
        mask_fitting = (ratio <= FIT_CUTOFF_RATIO)
    
    # 最終分離
    T_final_particle = T_core_map[iy, ix]
    threshold = ALPHA_THRESHOLD * T_final_particle
    valid_T = (T_final_particle > 1e-9)
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold[valid_T])
    
    return ~is_nonthermal, is_nonthermal, T_core_map, ix, iy

def maxwellian_func(E, kT, N_total, dE):
    if kT <= 0: return np.zeros_like(E)
    f_E = 2.0 * np.sqrt(E/np.pi) * (1.0/kT)**1.5 * np.exp(-E/kT)
    return f_E * N_total * dE

# =======================================================
# 局所プロット関数
# =======================================================
def plot_local_distribution(ax, E_T, E_NT, local_kT, title_str):
    """ 指定されたAxesに分布を描画 """
    bins = np.linspace(0, 200, 100) # 0-200keV
    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # Thermal
    cnt_T, _ = np.histogram(E_T, bins=bins)
    err_T = np.sqrt(cnt_T)
    mask_T = cnt_T > 0
    ax.errorbar(bin_centers[mask_T], cnt_T[mask_T], yerr=err_T[mask_T], fmt='o', c='blue', ms=4, alpha=0.6, label='Thermal')

    # Non-Thermal
    cnt_NT, _ = np.histogram(E_NT, bins=bins)
    err_NT = np.sqrt(cnt_NT)
    mask_NT = cnt_NT > 0
    ax.errorbar(bin_centers[mask_NT], cnt_NT[mask_NT], yerr=err_NT[mask_NT], fmt='x', c='red', ms=4, alpha=0.6, label='Non-Thermal')

    # 理論線 (その場所の温度を使う！)
    N_total_T = len(E_T)
    y_theory = maxwellian_func(bin_centers, local_kT, N_total_T, bin_width)
    ax.plot(bin_centers, y_theory, 'k--', lw=2, label=f'Maxwellian (kT={local_kT:.1f} keV)')

    ax.set_yscale('log')
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Count')
    ax.set_title(title_str)
    ax.legend()
    ax.grid(alpha=0.3)

def analyze_local_spots(fpath, ts, output_dir):
    print(f"Analyzing TS: {ts} ...")
    X, Y, E = get_particle_data(fpath)
    if X is None: return

    # 全体計算
    mask_T, mask_NT, T_map, ix_arr, iy_arr = perform_separation_and_get_map(X, Y, E)
    print(f"DEBUG: T_map max={np.max(T_map):.2f}, min={np.min(T_map[T_map>0]):.2f}, mean={np.mean(T_map):.2f}")

    # ----------------------------------------------------
    # 場所探し: 一番熱い場所(Hot) と 冷たい場所(Cold)
    # ----------------------------------------------------
    # ノイズ除去のため少し平滑化してから最大値を探す手もあるが、今回は直接最大値を探す
    
    # Hot Spot (最大温度の場所)
    iy_hot, ix_hot = np.unravel_index(np.argmax(T_map), T_map.shape)
    T_hot_val = T_map[iy_hot, ix_hot]
    
    # Cold Spot (平均より少し低い場所を適当に選ぶ。例えば中央付近で探す)
    # 単純に最小値だと真空(T=0)を拾う可能性があるので、平均値に近い場所を探す
    target_cold_T = np.mean(T_map[T_map > 0]) * 0.5
    # T_mapとの差が小さい場所を探す
    diff_map = np.abs(T_map - target_cold_T)
    diff_map[T_map < 1e-9] = 1e9 # 真空は除外
    iy_cold, ix_cold = np.unravel_index(np.argmin(diff_map), T_map.shape)
    T_cold_val = T_map[iy_cold, ix_cold]

    print(f"  Hot Spot @ ({ix_hot}, {iy_hot}), T={T_hot_val:.2f} keV")
    print(f"  Cold Spot @ ({ix_cold}, {iy_cold}), T={T_cold_val:.2f} keV")

    # ----------------------------------------------------
    # 粒子抽出 (Box領域)
    # ----------------------------------------------------
    def extract_particles_in_box(cx, cy):
        x_min_idx = max(0, cx - LOCAL_BOX_RADIUS)
        x_max_idx = min(NX-1, cx + LOCAL_BOX_RADIUS)
        y_min_idx = max(0, cy - LOCAL_BOX_RADIUS)
        y_max_idx = min(NY-1, cy + LOCAL_BOX_RADIUS)
        
        # 該当するインデックスの粒子マスク
        in_region = (ix_arr >= x_min_idx) & (ix_arr <= x_max_idx) & \
                    (iy_arr >= y_min_idx) & (iy_arr <= y_max_idx)
        
        # その領域の平均温度 (Box平均)
        local_T_mean = np.mean(T_map[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1])
        
        return mask_T & in_region, mask_NT & in_region, local_T_mean

    # Hot Region
    h_T_mask, h_NT_mask, h_box_T = extract_particles_in_box(ix_hot, iy_hot)
    # Cold Region
    c_T_mask, c_NT_mask, c_box_T = extract_particles_in_box(ix_cold, iy_cold)

    # ----------------------------------------------------
    # プロット (2枚並べる)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hot Plot
    plot_local_distribution(axes[0], E[h_T_mask], E[h_NT_mask], h_box_T, 
                            f"HOT Region (Around X={ix_hot}, Y={iy_hot})\nLocal T ~ {h_box_T:.1f} keV")
    
    # Cold Plot
    plot_local_distribution(axes[1], E[c_T_mask], E[c_NT_mask], c_box_T, 
                            f"COLD Region (Around X={ix_cold}, Y={iy_cold})\nLocal T ~ {c_box_T:.1f} keV")
    
    plt.tight_layout()
    out_name = os.path.join(output_dir, f'local_check_{ts}.png')
    plt.savefig(out_name)
    plt.close()
    print(f"  Saved: {out_name}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python local_check.py [start] [end] [step]")
        sys.exit(1)
    
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'local_verification_plots')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        fpath = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_e.dat')
        if os.path.exists(fpath):
            analyze_local_spots(fpath, ts, OUTPUT_DIR)

if __name__ == "__main__":
    main()