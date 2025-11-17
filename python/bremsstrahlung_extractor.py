import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# 物理定数
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge
ENERGY_THRESHOLD_KEV = 1.0  # ★ 軟X線源としてカウントするエネルギー閾値 (例: 1.0 keV)

# =======================================================
# ヘルパー関数 (init_param.dat パーサー)
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    init_param.dat (key ====> value 形式) を読み込む。
    """
    params = {}
    print(f"パラメータファイルを読み込み中: {param_filepath}")
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "====>" in line:
                    parts = line.split("====>")
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    value_part = value_part.replace('x', ' ')
                    values = value_part.split()
                    
                    if not values: continue

                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0]) # 321
                        params['NY_GRID_POINTS'] = int(values[1]) # 640
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])    # 1.0000
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
# ヘルパー関数 (粒子データ読み込み)
# =======================================================
def load_raw_particle_data(filepath):
    """
    Fortranが出力した psd_*.dat (生の粒子リスト) を読み込む。
    """
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
# メイン計算関数
# =======================================================
def calculate_xray_proxy_map(particle_data, nx, ny, x_min, x_max, y_min, y_max, threshold_kev):
    """
    粒子データを空間グリッドに振り分け、
    指定エネルギー以上の粒子数をカウントする。
    """
    
    # 粒子データの各列
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    v_norm_x = particle_data[:, 2] # vx/c
    v_norm_y = particle_data[:, 3] # vy/c
    v_norm_z = particle_data[:, 4] # vz/c
    
    # --- 1. 全粒子のエネルギーを計算 ---
    v_norm_sq = v_norm_x**2 + v_norm_y**2 + v_norm_z**2
    v_norm_sq = np.clip(v_norm_sq, 0.0, 1.0 - 1e-12) # 0 <= v^2/c^2 < 1

    gamma = 1.0 / np.sqrt(1.0 - v_norm_sq)
    E_kin_J = (gamma - 1.0) * m_e * (c**2)
    E_kin_keV = E_kin_J / KEV_TO_J

    # --- 2. 高エネルギー電子のマスクを作成 ---
    high_energy_mask = (E_kin_keV >= threshold_kev)

    # --- 3. 空間グリッドへのインデックス計算 ---
    x_bins = np.linspace(x_min, x_max, nx + 1)
    y_bins = np.linspace(y_min, y_max, ny + 1)
    
    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)
    
    ix = np.clip(bin_x - 1, 0, nx - 1)
    iy = np.clip(bin_y - 1, 0, ny - 1)
    
    # 空間範囲内かのマスク (psd_extractor と同様)
    spatial_mask = (X_pos >= x_min) & (X_pos <= x_max) & \
                   (Y_pos >= y_min) & (Y_pos <= y_max)
                   
    # --- 4. 空間範囲内 *かつ* 高エネルギー の粒子のみを抽出 ---
    final_mask = spatial_mask & high_energy_mask
    
    ix_masked = ix[final_mask]
    iy_masked = iy[final_mask]

    # --- 5. 2Dマップにカウント ---
    proxy_map = np.zeros((ny, nx))
    np.add.at(proxy_map, (iy_masked, ix_masked), 1)
    
    return proxy_map

# =======================================================
# メイン処理 (実行)
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python bremsstrahlung_calc_2d_map.py [開始] [終了] [間隔]")
        print("例: python bremsstrahlung_calc_2d_map.py 0 14000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: 引数は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")

    # --- 1. init_param.dat からグリッド設定を読み込む ---
    # (visual_fields.py や psd_extractor_revised.py と同じパスを指定)
    PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 
    try:
        sim_params = load_simulation_parameters(PARAM_FILE_PATH)
        GLOBAL_NX_PHYS = sim_params['NX_PHYS'] # 320
        GLOBAL_NY_PHYS = sim_params['NY_PHYS'] # 639
        DELX = sim_params['DELX']              # 1.0
    except Exception as e:
        print(f"init_param.dat の読み込みに失敗: {e}")
        sys.exit(1)
    
    # ★ 座標範囲 (X軸中心) を設定 (psd_extractor_revised.py の修正版と同じ)
    X_MIN = -GLOBAL_NX_PHYS * DELX / 2.0  # -> -160.0
    X_MAX = GLOBAL_NX_PHYS * DELX / 2.0   # ->  160.0
    Y_MIN = 0.0                           # ->    0.0
    Y_MAX = GLOBAL_NY_PHYS * DELX         # ->  639.0
    
    print(f"--- グリッド設定: {GLOBAL_NX_PHYS} x {GLOBAL_NY_PHYS}")
    print(f"--- 空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}]")
    print(f"--- X線プロキシ閾値: E >= {ENERGY_THRESHOLD_KEV} keV の電子数")

    # --- 2. 入出力ディレクトリ設定 ---
    # ★ 入力: Fortran が psd_*.dat を出力するディレクトリ
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    # ★ 出力: 2DマップTXTデータを保存するディレクトリ
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_2dmap_txt') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 生の粒子データ入力元: {data_dir} ---")
    print(f"--- 2DマップTXTデータ出力先: {OUTPUT_DIR} ---")
    
    species_suffix = 'e'
    species_label = 'electron'

    # --- 3. ループ処理 ---
    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        filename = f'{timestep}_0160-0320_psd_{species_suffix}.dat'
        filepath = os.path.join(data_dir, filename)
        
        print(f"--- {species_label} の生粒子データ ({filename}) を処理中 ---")

        particle_data = load_raw_particle_data(filepath)
        
        if particle_data is None or particle_data.size == 0:
            print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
            continue
        
        print(f"  -> {len(particle_data)} 個の粒子を読み込みました。2Dマップを計算中...")

        # --- 4. 2Dマップ計算 ---
        proxy_map = calculate_xray_proxy_map(
            particle_data,
            GLOBAL_NX_PHYS, GLOBAL_NY_PHYS,
            X_MIN, X_MAX, Y_MIN, Y_MAX,
            ENERGY_THRESHOLD_KEV
        )
        
        print(f"  -> 2Dマップ計算完了。 (最大粒子数: {np.max(proxy_map)})")

        # --- 5. TXTファイルへの保存 ---
        output_filename = os.path.join(OUTPUT_DIR, f'soft_xray_proxy_map_{timestep}.txt')
        np.savetxt(output_filename, proxy_map, 
                   header=f'Soft X-ray Proxy Map (Particle count with E >= {ENERGY_THRESHOLD_KEV} keV)\nShape: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})',
                   fmt='%.3f') # 粒子数なので整数でも良いが、将来の拡張性のため
                   
        print(f"  -> 2Dマップデータを {output_filename} に保存しました。")
        print(f"--- タイムステップ {timestep} のTXTデータ保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")

if __name__ == "__main__":
    main()