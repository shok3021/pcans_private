import numpy as np
import os
import glob
import sys
import time
# import fnmatch # ★ 不要なため削除
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ ステップ1: 物理定数とエネルギービンの定義 (★ ご要望のビンに更新)
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# ★ ご要望に合わせてエネルギービン (keV) を更新
# (15-20keVが抜けていたため補完しました。不要な範囲は削除してください)
ENERGY_BINS_KEV = [
    (1.0, 5.0,   '01keV_05keV'),
    (5.0, 10.0,  '05keV_10keV'),
    (10.0, 15.0, '10keV_15keV'),
    (15.0, 20.0, '15keV_20keV'), # (10-15 と 20-25 の間を補完)
    (20.0, 25.0, '20keV_25keV'),
    (25.0, 30.0, '25keV_30keV'),
    (30.0, 35.0, '30keV_35keV'),
    (35.0, 40.0, '35keV_40keV'),
    (40.0, 45.0, '40keV_45keV'),
    (45.0, 50.0, '45keV_50keV'),
    (50.0, 500.0, '50keV_over') # 50keV以上 (上限は適当に設定)
]

print("--- エネルギービン設定 (更新版) ---")
for e_min, e_max, label in ENERGY_BINS_KEV:
    print(f"  {label}: [{e_min}, {e_max}) keV")
print("---------------------------------")

# =======================================================
# ★ ステップ2: init_param.dat 読み込み関数 (★ '=>' に修正)
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    (psd_extractor_revised.py に合わせて '=>' を使用)
    """
    params = {}
    print(f"パラメータファイルを読み込み中: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "=>" in line: # ★ '====>' から '=>' に変更
                    parts = line.split("=>") # ★ '====>' から '=>' に変更
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    value_part = value_part.replace('x', ' ')
                    values = value_part.split()
                    if not values:
                        continue
                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0]) 
                        params['NY_GRID_POINTS'] = int(values[1])
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])
                        # (他のパラメータ読み込みは省略)

    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        sys.exit(1)
        
    required_keys = ['NX_GRID_POINTS', 'NY_GRID_POINTS', 'DELX']
    if not all(key in params for key in required_keys):
        print("★★ エラー: 必要なパラメータ ('grid size', 'dx') を抽出できませんでした。")
        sys.exit(1)
        
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
        
    return params

# =======================================================
# ★ ステップ3: グローバル定数を動的に設定
# =======================================================
# (変更なし)
PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

try:
    sim_params = load_simulation_parameters(PARAM_FILE_PATH)
    GLOBAL_NX_PHYS = sim_params['NX_PHYS'] # 320
    GLOBAL_NY_PHYS = sim_params['NY_PHYS'] # 639
    DELX = sim_params['DELX']              # 1.0
except Exception as e:
    print(f"init_param.dat の読み込みに失敗しました: {e}")
    print("スクリプトを終了します。")
    sys.exit(1)

# =======================================================
# ★ ステップ4: 座標範囲を X-Center で動的に設定
# =======================================================
# (変更なし)
X_MIN = -GLOBAL_NX_PHYS * DELX / 2.0  # -> -160.0
X_MAX = GLOBAL_NX_PHYS * DELX / 2.0   # ->  160.0
Y_MIN = 0.0                           # ->    0.0
Y_MAX = GLOBAL_NY_PHYS * DELX         # ->  639.0

print(f"--- グリッド設定 (動的読込) ---")
print(f"X方向物理セル数: {GLOBAL_NX_PHYS}, Y方向物理セル数: {GLOBAL_NY_PHYS}")
print(f"★ 空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}] (セル幅: {DELX})")

# =======================================================
# ★ ステップ5: 抽出・計算関数 (★ デバッグ出力追加)
# =======================================================
def calculate_xray_proxy_binned(particle_data):
    """
    (デバッグ出力を psd_extractor_revised.py から移植)
    """
    
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 
    
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    v_norm_x = particle_data[:, 2] # vx/c
    v_norm_y = particle_data[:, 3] # vy/c
    v_norm_z = particle_data[:, 4] # vz/c
    
    N_total = len(X_pos) 

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
    
    # ★★★ デバッグ出力 (psd_extractor_revised.py から移植) ★★★
    print("  --- デバッグ情報 ---")
    print(f"  X-Range (設定): [{x_min}, {x_max}], Y-Range (設定): [{y_min}, {y_max}]")
    if N_total > 0:
        print(f"  ★ X-pos min/max (粒子): {np.min(X_pos):.3f} / {np.max(X_pos):.3f}")
        print(f"  ★ Y-pos min/max (粒子): {np.min(Y_pos):.3f} / {np.max(Y_pos):.3f}")
    print(f"  全粒子数: {N_total}, マスクされた粒子数 (集計対象): {N_masked}")
    # ★★★ デバッグ出力ここまで ★★★

    if N_masked == 0:
        if N_total > 0:
            print("  -> **警告: マスクされた粒子がゼロです。グリッド範囲と粒子座標が一致していません。**")
        binned_maps = {}
        for _, _, bin_label in ENERGY_BINS_KEV:
            binned_maps[bin_label] = np.zeros((NY, NX))
        return binned_maps

    binned_maps = {}
    print("  -> (3/4) エネルギービンごとにマスクを作成し、カウント中...")
    
    ix_spatial = ix[spatial_mask]
    iy_spatial = iy[spatial_mask]
    E_kin_keV_spatial = E_kin_keV[spatial_mask]
    
    for e_min_kev, e_max_kev, bin_label in ENERGY_BINS_KEV:
        energy_mask = (E_kin_keV_spatial >= e_min_kev) & (E_kin_keV_spatial < e_max_kev)
        ix_final = ix_spatial[energy_mask]
        iy_final = iy_spatial[energy_mask]

        proxy_map = np.zeros((NY, NX))
        np.add.at(proxy_map, (iy_final, ix_final), 1)
        
        binned_maps[bin_label] = proxy_map
        print(f"    -> ビン {bin_label} (E=[{e_min_kev}, {e_max_kev}) keV) : {len(ix_final)} 粒子")

    print("  -> (4/4) 2Dマップ計算完了。")
    return binned_maps

# =======================================================
# ★ ステップ6: データ読み込み関数
# =======================================================
def load_text_data(filepath):
    # (変更なし)
    if not os.path.exists(filepath):
        print(f"    エラー: ファイルが見つかりません: {filepath}")
        return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            if data.size == 0:
                 print("    -> ファイルは空です。")
                 return np.array([])
            data = data.reshape(1, -1)
        if data.size == 0:
            print("    -> ファイルは空です。")
            return np.array([])
        return data
    except Exception as e:
        print(f"    エラー: {filepath} のテキスト読み込みに失敗: {e}")
        return None

# =======================================================
# ★ ステップ7: メイン関数 (★ ファイル読み込みを元に戻す)
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python bremsstrahlung_extractor_binned.py [開始] [終了] [間隔]")
        print("例: python bremsstrahlung_extractor_binned.py 0 14000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: すべての引数 (開始、終了、間隔) は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")
    
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_binned_txt') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 生の粒子データ入力元: {data_dir} ---")
    print(f"--- 2DマップTXTデータ出力先 (ベース): {OUTPUT_DIR} ---")
    
    species_list = [('e', 'electron')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        for suffix, species_label in species_list:
            
            # ★★★ ここを修正 (fnmatch を削除し、ハードコードに戻す) ★★★
            # (スクリーンショットのファイル名に基づき、このファイル名が正しいと仮定)
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            print(f"\n--- {species_label} データ ({filename}) を処理中 ---")

            particle_data = load_text_data(filepath)
            
            if particle_data is None or particle_data.size == 0:
                print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
                continue
            
            print(f"  -> {len(particle_data)} 個の粒子を読み込みました。X線プロキシ (ビン別) を計算中...")

            # --- 1. 粒子データからX線プロキシ (ビン別) を計算 ---
            binned_proxy_maps = calculate_xray_proxy_binned(particle_data)
            
            # --- 2. 各ビンをそれぞれのサブディレクトリにテキストファイルとして保存 ---
            print(f"  -> TXTファイルに保存中...")
        
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
                    f'Soft X-ray Proxy Map (Particle count)\n'
                    f'Energy Bin: [{e_min_str}, {e_max_str}) keV (Label: {bin_label})\n'
                    f'Shape: ({GLOBAL_NY_PHYS}, {GLOBAL_NY_PHYS})' # (★タイポ修正: NX_PHYS)
                )
                
                np.savetxt(output_filename, proxy_map, 
                           header=header_txt,
                           fmt='%d') 
                           
                print(f"    -> {bin_label} マップを {output_filename} に保存しました。 (Max: {np.max(proxy_map)})")
            
            print(f"--- タイムステップ {timestep} の {species_label} データ抽出・保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")

if __name__ == "__main__":
    main()