import numpy as np
import os
import glob
import sys
import time
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ ステップ1: 物理定数とエネルギービンの定義
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# ★ ご要望に合わせてエネルギービン (keV) を更新
# 1~100, 100~200, 200~300, (300~400を補完), 400~
ENERGY_BINS_KEV = [
    (1.0, 100.0,    '001keV_100keV'),
    (100.0, 200.0,  '100keV_200keV'),
    (200.0, 300.0,  '200keV_300keV'),
    (300.0, 400.0,  '300keV_400keV'), # 隙間を埋めるため追加
    (400.0, 100000.0, '400keV_over')  # 400keV以上
]

print("--- エネルギービン設定 (更新版: 100keV step) ---")
for e_min, e_max, label in ENERGY_BINS_KEV:
    print(f"  {label}: [{e_min}, {e_max}) keV")
print("---------------------------------")


# =======================================================
# ★ ステップ2: グローバル定数を設定
# =======================================================
GLOBAL_NX_GRID_POINTS = 321
GLOBAL_NY_GRID_POINTS = 640

# 物理領域のグリッド数
GLOBAL_NX_PHYS = GLOBAL_NX_GRID_POINTS - 1 # 320
GLOBAL_NY_PHYS = GLOBAL_NY_GRID_POINTS - 1 # 639
DELX = 1.0

# 座標範囲 [0, 320] x [0, 639]
X_MIN = 0.0
X_MAX = GLOBAL_NX_PHYS * DELX 
Y_MIN = 0.0
Y_MAX = GLOBAL_NY_PHYS * DELX 

print(f"--- グリッド設定 ---")
print(f"X方向物理セル数: {GLOBAL_NX_PHYS}, Y方向物理セル数: {GLOBAL_NY_PHYS}")
print(f"★ 空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}] (セル幅: {DELX})")


# =======================================================
# ★ ステップ3: 抽出・計算関数 (Intensity Mode)
# =======================================================
def calculate_xray_proxy_binned(particle_data):
    """
    粒子の生データからエネルギーを計算し、Intensity（エネルギー総和）としてマッピングする。
    """
    
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 
    
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    
    v_norm_x = particle_data[:, 2]
    v_norm_y = particle_data[:, 3]
    v_norm_z = particle_data[:, 4]
    
    N_total = len(X_pos) 

    # --- 1. エネルギー計算 ---
    print("  -> (1/4) 全粒子のエネルギーを計算中...")
    v_norm_sq = v_norm_x**2 + v_norm_y**2 + v_norm_z**2
    v_norm_sq = np.clip(v_norm_sq, 0.0, 1.0 - 1e-12) 

    gamma = 1.0 / np.sqrt(1.0 - v_norm_sq)
    E_kin_J = (gamma - 1.0) * m_e * (c**2)
    E_kin_keV = E_kin_J / KEV_TO_J
    
    if N_total > 0:
        print(f"  -> エネルギー範囲 (keV): [{np.min(E_kin_keV):.2f}, {np.max(E_kin_keV):.2f}]")
    else:
        print("  -> エネルギー範囲 (keV): 粒子なし")

    # --- 2. 空間インデックス計算 ---
    print("  -> (2/4) 空間グリッドへのインデックスを計算中...")
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)

    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    spatial_mask = (X_pos >= x_min) & (X_pos <= x_max) & \
                   (Y_pos >= y_min) & (Y_pos <= y_max)
                   
    N_masked = np.sum(spatial_mask)
    
    if N_masked == 0:
        binned_maps = {}
        for _, _, bin_label in ENERGY_BINS_KEV:
            binned_maps[bin_label] = np.zeros((NY, NX))
        return binned_maps

    # --- 3. エネルギービンごとにIntensity集計 ---
    binned_maps = {}
    print("  -> (3/4) エネルギービンごとにIntensity(エネルギー総和)を集計中...")
    
    ix_spatial = ix[spatial_mask]
    iy_spatial = iy[spatial_mask]
    E_kin_keV_spatial = E_kin_keV[spatial_mask]
    
    for e_min_kev, e_max_kev, bin_label in ENERGY_BINS_KEV:
        energy_mask = (E_kin_keV_spatial >= e_min_kev) & (E_kin_keV_spatial < e_max_kev)
        
        ix_final = ix_spatial[energy_mask]
        iy_final = iy_spatial[energy_mask]
        weights_final = E_kin_keV_spatial[energy_mask] # エネルギーそのものを重みとして足す

        proxy_map = np.zeros((NY, NX))
        np.add.at(proxy_map, (iy_final, ix_final), weights_final)
        
        binned_maps[bin_label] = proxy_map
        print(f"    -> ビン {bin_label} : {len(ix_final)} 粒子寄与")

    print("  -> (4/4) 計算完了。")
    return binned_maps


# =======================================================
# ★ ステップ4: データ読み込み関数
# =======================================================
def load_text_data(filepath):
    if not os.path.exists(filepath):
        print(f"    エラー: ファイルが見つかりません: {filepath}")
        return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            if data.size == 0: return np.array([])
            data = data.reshape(1, -1)
        if data.size == 0: return np.array([])
        return data
    except Exception as e:
        print(f"    エラー: {filepath} 読み込み失敗: {e}")
        return None

# =======================================================
# ★ ステップ5: メイン関数
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python bremsstrahlung_extractor_binned.py [開始] [終了] [間隔]")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: 引数は整数で指定してください。")
        sys.exit(1)
        
    print(f"--- 処理範囲: {start_step} ～ {end_step}, step={step_size} ---")
    
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_binned_txt') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    species_list = [('e', 'electron')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        timestep = f"{current_step:06d}" 
        print(f"\n=== TS: {timestep} ===")

        for suffix, species_label in species_list:
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            print(f"--- {species_label} 処理中 ---")
            particle_data = load_text_data(filepath)
            
            if particle_data is None or particle_data.size == 0:
                continue
            
            # 計算
            binned_proxy_maps = calculate_xray_proxy_binned(particle_data)
            
            # 保存
            for bin_label, proxy_map in binned_proxy_maps.items():
                e_min_str, e_max_str = "N/A", "N/A"
                for e_min, e_max, label in ENERGY_BINS_KEV:
                    if label == bin_label:
                        e_min_str, e_max_str = str(e_min), str(e_max)
                        break
                
                output_bin_dir = os.path.join(OUTPUT_DIR, bin_label)
                os.makedirs(output_bin_dir, exist_ok=True)
                
                output_filename = os.path.join(output_bin_dir, f'xray_proxy_{timestep}_{bin_label}.txt')
                
                header_txt = (
                    f'Soft X-ray Intensity Proxy Map (Sum of Energy in keV)\n'
                    f'Energy Bin: [{e_min_str}, {e_max_str}) keV (Label: {bin_label})\n'
                    f'Shape: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})'
                )
                
                np.savetxt(output_filename, proxy_map, header=header_txt, fmt='%.6g') 
                print(f"    -> 保存: {bin_label}")

    print("\n=== 完了 ===")

if __name__ == "__main__":
    main()