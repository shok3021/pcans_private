import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge, alpha, pi

# =======================================================
# ★ 設定: 観測したい「光子(X線)」のエネルギーバンド
# =======================================================
# 実際の衛星(RHESSIやFOXSI, Hinodeなど)のチャンネルに合わせると良いです
PHOTON_BINS = [
    (1.0, 100.0,    '001keV_100keV'),
    (100.0, 200.0,  '100keV_200keV'),
    (200.0, 500.0,  '200keV_500keV'),
    (500.0, 1000.0,  '500keV_1000keV'),
    (1000.0, 2000.0,  '1000keV_2000keV'),
    (2000.0, 5000.0,  '2000keV_5000keV'),
    (5000.0, 10000.0,  '5000keV_10000keV'),
    (10000.0, 20000.0,  '10000keV_20000keV'),
    (20000.0, 50000.0,  '20000keV_50000keV'),
    (50000.0, 100000.0, '50000keV_over')
]

# 定数
KEV_TO_J = 1000.0 * elementary_charge
MC2_KEV = 510.998  # 電子静止質量 (keV)

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
X_MIN, X_MAX = 0.0, NX * 1.0
Y_MIN, Y_MAX = 0.0, NY * 1.0

# =======================================================
# 物理計算エンジン
# =======================================================

def calculate_bethe_heitler_weight(E_el_keV, k_ph_keV):
    """
    Bethe-Heitler微分断面積に基づく重み計算
    E_el_keV: 電子の運動エネルギー配列 (keV)
    k_ph_keV: ターゲットとする光子エネルギー (keV)
    
    戻り値: 放射確率(重み)配列。電子エネルギーが光子エネルギーより低い場合は0。
    """
    # 閾値チェック: 電子エネルギー < 光子エネルギー なら放射できない
    mask_enable = E_el_keV > k_ph_keV
    if not np.any(mask_enable):
        return np.zeros_like(E_el_keV)

    # 計算用変数の準備 (有効な粒子のみ)
    E_k = E_el_keV[mask_enable]  # 運動エネルギー T
    k   = k_ph_keV
    
    # 全エネルギー (単位: mc^2)
    # E_total = (T + mc^2) / mc^2 = T_in_mc2 + 1
    E_tot_initial = (E_k / MC2_KEV) + 1.0
    E_tot_final   = ((E_k - k) / MC2_KEV) + 1.0
    
    # 運動量 p (単位: mc)
    p_initial = np.sqrt(E_tot_initial**2 - 1.0)
    p_final   = np.sqrt(E_tot_final**2 - 1.0)
    
    # ベーテ・ハイトラー断面積 (Bethe-Heitler Cross Section)
    # dσ/dk ~ (p_f / p_i) * (1/k) * log term
    # 定数係数は相対比較なら無視できるが、エネルギー依存性は重要
    
    # log項: ln( (Ei*Ef + pi*pf - 1) / k_in_mc2 ) 近似などがあるが、
    # ここでは標準的な形式: ln( (E_i + E_f + p_i + p_f) / (E_i + E_f - p_i - p_f) ) - ...
    # 簡略化した Gluckstern-Hull 形式または相対論的近似を使用
    
    # よく使われる近似形 (L ~ log term)
    L = 2.0 * np.log( (E_tot_initial * E_tot_final + p_initial * p_final - 1.0) / (k / MC2_KEV) )
    
    # クーロン補正などを無視した主要項
    # Weight ~ (1/k) * (p_f / p_i) * [ (4/3) - 2*Ei*Ef * ( (p_f/p_i)^2 + (p_i/p_f)^2 ) ... ] 
    # 天体プラズマの可視化用には「p_final / p_initial」のスケーリングが支配的
    
    # 簡易的だが物理的に正しい依存性を持つ重み:
    # W = (1/k) * log(...)
    weight_val = (1.0 / k) * L
    
    # 結果配列に格納
    weights = np.zeros_like(E_el_keV)
    weights[mask_enable] = weight_val
    
    return weights

def generate_photon_maps(e_data, i_data):
    # 1. イオン密度マップ (n_i)
    i_X, i_Y = i_data[:, 0], i_data[:, 1]
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    
    print("  -> Creating Ion Density Map...")
    Ni_Map, _, _ = np.histogram2d(i_Y, i_X, bins=[y_edges, x_edges])

    # 2. 電子データの準備
    e_X, e_Y = e_data[:, 0], e_data[:, 1]
    v_sq = e_data[:, 2]**2 + e_data[:, 3]**2 + e_data[:, 4]**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_e_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J

    # 電子のグリッド座標
    ix = np.clip(np.digitize(e_X, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(e_Y, y_edges) - 1, 0, NY - 1)

    maps = {}

    # 3. 光子エネルギーバンドごとのループ
    # ここが「電子ループ」から「光子ループ」への転換点
    print("  -> calculating Photon Maps (Convolution)...")
    
    for k_min, k_max, label in PHOTON_BINS:
        # 代表エネルギー (中心値) で重みを計算
        k_target = (k_min + k_max) / 2.0
        
        # この光子を出せるのは、エネルギーが k_target 以上の電子だけ
        # (計算高速化のため、明らかに足りない電子は足切り)
        mask_capable = E_e_keV > k_min
        
        if not np.any(mask_capable):
            print(f"     [Bin: {label}] No electrons energetic enough.")
            maps[label] = np.zeros((NY, NX))
            continue
        
        # 重み計算 (Bethe-Heitler)
        # ターゲット粒子群に対して一括計算
        w_bh = calculate_bethe_heitler_weight(E_e_keV[mask_capable], k_target)
        
        # マップへの蓄積
        # I_photon ~ Σ (Weight_BH)
        flux_map = np.zeros((NY, NX))
        np.add.at(flux_map, (iy[mask_capable], ix[mask_capable]), w_bh)
        
        # 最後にイオン密度を掛ける: I ~ n_i * (Σ σ)
        # これが「観測されるX線強度」
        photon_map = flux_map * Ni_Map
        
        maps[label] = photon_map
        print(f"     [Bin: {label}] Generated.")

    return maps

def save_maps(maps_dict, timestep, output_dir):
    for label, data in maps_dict.items():
        # TXT保存
        subdir = os.path.join(output_dir, label)
        os.makedirs(subdir, exist_ok=True)
        fname = f'photon_intensity_{label}_{timestep}.txt'
        fpath = os.path.join(subdir, fname)
        
        header = f"Photon Intensity Map (Bethe-Heitler weighted)\nBin: {label}\nTS: {timestep}"
        np.savetxt(fpath, data, header=header, fmt='%.6g')

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python calc_photon_map.py [start] [end] [step]")
        sys.exit(1)

    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_photon_intensity')
    
    print("--- Generating Observation-Equivalent X-ray Maps ---")
    
    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        f_e = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_e.dat')
        f_i = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_i.dat')
        
        if os.path.exists(f_e) and os.path.exists(f_i):
            print(f"\nTS: {ts}")
            # 読み込み
            de = np.loadtxt(f_e)
            di = np.loadtxt(f_i)
            if de.ndim==1: de=de.reshape(1,-1)
            if di.ndim==1: di=di.reshape(1,-1)
            
            # 計算
            maps = generate_photon_maps(de, di)
            
            # 保存
            save_maps(maps, ts, OUTPUT_DIR)
        else:
            print(f"Skipping {ts} (File missing)")

if __name__ == "__main__":
    main()