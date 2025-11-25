import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ 設定
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge
ELECTRON_MASS_KEV = 510.998 

# エネルギービン
ENERGY_BINS = [
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

# フィッティングパラメータ
FIT_CUTOFF_RATIO = 3.0 
MAX_ITER = 10
ALPHA_THRESHOLD = 10.0 

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
X_MIN, X_MAX = 0.0, NX * 1.0
Y_MIN, Y_MAX = 0.0, NY * 1.0

# =======================================================
# 計算エンジン
# =======================================================
def calculate_detailed_intensity_maps(e_data, i_data):
    """
    引数:
      e_data: 電子データ (x, y, vx, vy, vz, ...)
      i_data: イオンデータ (x, y, ... 他は不要)
    """
    # --- 1. イオン密度マップ (ni) の作成 ---
    # イオンは位置情報だけでOK
    print("  -> Creating Ion Density Map (ni)...")
    i_X = i_data[:, 0]
    i_Y = i_data[:, 1]
    
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    
    # イオンの個数密度マップを作成
    Ion_Density_Map, _, _ = np.histogram2d(i_Y, i_X, bins=[y_edges, x_edges])

    # --- 2. 電子の処理 ---
    print("  -> Processing Electrons...")
    e_X = e_data[:, 0]
    e_Y = e_data[:, 1]
    # 速度 -> エネルギー -> 運動量
    v_sq = e_data[:, 2]**2 + e_data[:, 3]**2 + e_data[:, 4]**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J

    # グリッド座標
    ix = np.clip(np.digitize(e_X, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(e_Y, y_edges) - 1, 0, NY - 1)

    # --- 3. 反復的温度フィッティング (電子データのみで決定) ---
    print(f"  -> Iterative fitting of Maxwellian core (Max Iter: {MAX_ITER})...")
    mask_fitting = np.ones(len(E_kin_keV), dtype=bool)
    T_core_map = np.zeros((NY, NX))

    for i in range(MAX_ITER):
        curr_X = e_X[mask_fitting]
        curr_Y = e_Y[mask_fitting]
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

    # --- 4. 最終分類 ---
    T_final = T_core_map[iy, ix]
    threshold = ALPHA_THRESHOLD * T_final
    valid_T = (T_final > 1e-9)
    
    is_nonthermal = np.zeros(len(E_kin_keV), dtype=bool)
    is_nonthermal[valid_T] = (E_kin_keV[valid_T] > threshold[valid_T])
    is_thermal = ~is_nonthermal

    # --- 5. 強度マップ作成 ( I ~ p * ni ) ---
    print("  -> Aggregating intensity maps (Weight = p_ele * n_ion)...")
    
    # 運動量 p = sqrt(E^2 + 2mE)
    Momentum_p = np.sqrt(E_kin_keV**2 + 2 * E_kin_keV * ELECTRON_MASS_KEV)
    
    maps = {'Thermal': {}, 'NonThermal': {}}
    for _, _, label in ENERGY_BINS:
        maps['Thermal'][label] = np.zeros((NY, NX))
        maps['NonThermal'][label] = np.zeros((NY, NX))

    for e_min, e_max, label in ENERGY_BINS:
        in_bin = (E_kin_keV >= e_min) & (E_kin_keV < e_max)
        
        # Thermal
        mask_T = is_thermal & in_bin
        if np.any(mask_T):
            flux_map = np.zeros((NY, NX))
            # 電子の寄与 (pの総和)
            np.add.at(flux_map, (iy[mask_T], ix[mask_T]), Momentum_p[mask_T])
            # 実際のイオン密度を掛ける
            maps['Thermal'][label] = flux_map * Ion_Density_Map
            
        # NonThermal
        mask_NT = is_nonthermal & in_bin
        if np.any(mask_NT):
            flux_map = np.zeros((NY, NX))
            np.add.at(flux_map, (iy[mask_NT], ix[mask_NT]), Momentum_p[mask_NT])
            maps['NonThermal'][label] = flux_map * Ion_Density_Map
            
    return maps

def save_category_txt(maps_dict, timestep, output_base):
    for particle_type in ['Thermal', 'NonThermal']:
        type_dir = os.path.join(output_base, particle_type)
        os.makedirs(type_dir, exist_ok=True)
        for e_min, e_max, bin_label in ENERGY_BINS:
            intensity_map = maps_dict[particle_type][bin_label]
            bin_dir = os.path.join(type_dir, bin_label)
            os.makedirs(bin_dir, exist_ok=True)
            txt_name = f'intensity_{particle_type}_{bin_label}_{timestep}.txt'
            txt_path = os.path.join(bin_dir, txt_name)
            header = (f'Bremsstrahlung Intensity Map (Strict Mode: p_e * n_i)\n'
                      f'Type: {particle_type}\n'
                      f'Energy Bin: {bin_label} ({e_min}-{e_max} keV)\n'
                      f'Timestep: {timestep}')
            np.savetxt(txt_path, intensity_map, header=header, fmt='%.6g')

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python detailed_brems_strict_ion.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # ★ パス設定
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_ion_weighted')
    
    print(f"--- Strict Bremsstrahlung (Using Actual Ion Data) ---")
    
    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        print(f"\n=== Processing TS: {ts} ===")
        
        # 電子データ
        fpath_e = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_e.dat')
        # イオンデータ (ファイル名規則を仮定)
        fpath_i = os.path.join(DATA_DIR, f'{ts}_0160-0320_psd_i.dat')
        
        if not os.path.exists(fpath_e):
            print(f"  Electron file not found: {fpath_e}")
            continue
        if not os.path.exists(fpath_i):
            print(f"  Ion file not found: {fpath_i}")
            continue
            
        try:
            # 読み込み
            print("  Reading Electrons...")
            data_e = np.loadtxt(fpath_e)
            if data_e.ndim == 1: data_e = data_e.reshape(1, -1)
            
            print("  Reading Ions...")
            # イオンは位置(0,1列目)だけで良いのでusecols等を使うと速いが、
            # np.loadtxtは行単位処理なので全読みしてメモリで切る
            data_i = np.loadtxt(fpath_i)
            if data_i.ndim == 1: data_i = data_i.reshape(1, -1)
            
            if data_e.size == 0 or data_i.size == 0:
                print("  Data empty.")
                continue

            # 計算実行
            result_maps = calculate_detailed_intensity_maps(data_e, data_i)
            
            # 保存
            save_category_txt(result_maps, ts, OUTPUT_DIR)
            print(f"  Saved TS: {ts}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\nAll Done.")

if __name__ == "__main__":
    main()