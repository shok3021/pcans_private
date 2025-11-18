import numpy as np
import os
import glob
import sys
import time
from scipy.constants import m_e, c, elementary_charge # ★ 物理定数を追加

# =======================================================
# ★ ステップ1: 物理定数とエネルギービンの定義
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# ★ エネルギービン (keV) を定義 (下限, 上限, ラベル)
# (ご要望に合わせて自由にカスタマイズしてください)
ENERGY_BINS_KEV = [
    (0.01, 0.30, '010eV_300eV'), # 10eV - 300eV
    (0.30, 0.50, '300eV_500eV'), # 300eV - 500eV
    (0.50, 1.00, '500eV_1keV'),  # 500eV - 1keV
    (1.00, 5.00, '1keV_5keV'),   # 1keV - 5keV
    (5.00, 200.0, '5keV_over')  # 5keV以上 (上限は適当に設定)
]
print("--- エネルギービン設定 ---")
for e_min, e_max, label in ENERGY_BINS_KEV:
    print(f"  {label}: [{e_min}, {e_max}) keV")
print("------------------------")

# =======================================================
# ★ ステップ2: init_param.dat 読み込み関数
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    (ご提示いただいたコードから変更なし)
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
                    if not values:
                        continue
                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0]) 
                        params['NY_GRID_POINTS'] = int(values[1])
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])
                        params['DT'] = float(values[1])
                        params['C_LIGHT'] = float(values[2])
                    # (以降のパラメータ読み込みも省略)

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

# (ご提示いただいたコードから変更なし)
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
# (ご提示いただいたコードから変更なし)
X_MIN = -GLOBAL_NX_PHYS * DELX / 2.0  # -> -160.0
X_MAX = GLOBAL_NX_PHYS * DELX / 2.0   # ->  160.0
Y_MIN = 0.0                           # ->    0.0
Y_MAX = GLOBAL_NY_PHYS * DELX         # ->  639.0

print(f"--- グリッド設定 (動的読込) ---")
print(f"X方向物理セル数: {GLOBAL_NX_PHYS}, Y方向物理セル数: {GLOBAL_NY_PHYS}")
print(f"★ 空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}] (セル幅: {DELX})")

# =======================================================
# ★ ステップ5: 抽出・計算関数 (エネルギービン別カウントに変更)
# =======================================================
def calculate_xray_proxy_binned(particle_data):
    """
    粒子の生データからエネルギーを計算し、
    エネルギービン別に2Dマップにカウントする。
    (グローバル変数 NX, NY, X_MIN, X_MAX などを使用)
    """
    
    # グローバル変数を使用
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 
    
    # 粒子データの各列
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    
    # ★ ご提示いただいた最初のスクリプトに基づき、
    # ★ 2,3,4列目は v/c (規格化速度) と仮定
    v_norm_x = particle_data[:, 2] # vx/c
    v_norm_y = particle_data[:, 3] # vy/c
    v_norm_z = particle_data[:, 4] # vz/c
    
    N_total = len(X_pos) # 全粒子数

    # --- 1. 全粒子のエネルギーを計算 (相対論的) ---
    print("  -> (1/4) 全粒子のエネルギーを計算中...")
    v_norm_sq = v_norm_x**2 + v_norm_y**2 + v_norm_z**2
    # v^2/c^2 が 1.0 を超えないようにクリップ (数値誤差対策)
    v_norm_sq = np.clip(v_norm_sq, 0.0, 1.0 - 1e-12) 

    gamma = 1.0 / np.sqrt(1.0 - v_norm_sq)
    E_kin_J = (gamma - 1.0) * m_e * (c**2)
    E_kin_keV = E_kin_J / KEV_TO_J
    print(f"  -> エネルギー範囲 (keV): [{np.min(E_kin_keV):.2f}, {np.max(E_kin_keV):.2f}]")


    # --- 2. 空間インデックス計算 ---
    # (ご提示いただいた 'calculate_moments_from_particle_list' と同じロジック)
    print("  -> (2/4) 空間グリッドへのインデックスを計算中...")
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)

    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    # 空間マスク (ご提示いただいた 'calculate_moments_from_particle_list' と同じ)
    spatial_mask = (X_pos >= x_min) & (X_pos <= x_max) & \
                   (Y_pos >= y_min) & (Y_pos <= y_max)
                   
    N_masked = np.sum(spatial_mask)
    
    print("  --- デバッグ情報 ---")
    print(f"  X-Range (設定): [{x_min}, {x_max}], Y-Range (設定): [{y_min}, {y_max}]")
    if N_total > 0:
        print(f"  X-pos min/max (粒子): {np.min(X_pos):.3f} / {np.max(X_pos):.3f}")
        print(f"  Y-pos min/max (粒子): {np.min(Y_pos):.3f} / {np.max(Y_pos):.3f}")
    print(f"  全粒子数: {N_total}, マスクされた粒子数 (集計対象): {N_masked}")
    
    if N_masked == 0:
        if N_total > 0:
            print("  -> **致命的警告: マスクされた粒子がゼロです。グリッド範囲と粒子座標が一致していません。**")
        # 全てのビンに対して空のマップを返す
        binned_maps = {}
        for _, _, bin_label in ENERGY_BINS_KEV:
            binned_maps[bin_label] = np.zeros((NY, NX))
        return binned_maps

    # --- 3. エネルギービンごとにマップを作成 ---
    binned_maps = {}
    
    print("  -> (3/4) エネルギービンごとにマスクを作成し、カウント中...")
    
    # 空間マスク済みのインデックスとエネルギー
    ix_spatial = ix[spatial_mask]
    iy_spatial = iy[spatial_mask]
    E_kin_keV_spatial = E_kin_keV[spatial_mask]
    
    for e_min_kev, e_max_kev, bin_label in ENERGY_BINS_KEV:
        
        # --- 3a. エネルギーマスクを作成 (E_min <= E < E_max) ---
        # (すでに空間マスクされた粒子のみを対象)
        energy_mask = (E_kin_keV_spatial >= e_min_kev) & (E_kin_keV_spatial < e_max_kev)
        
        # --- 3b. 最終的なインデックスを抽出 ---
        ix_final = ix_spatial[energy_mask]
        iy_final = iy_spatial[energy_mask]

        # --- 3c. 2Dマップにカウント ---
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
    # (ご提示いただいたコードから変更なし)
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

# (save_data_to_txt は main 関数内で処理するため削除)

# =======================================================
# ★ ステップ7: メイン関数 (X線ビン出力に変更)
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
    
    # (ご提示いただいたコードから変更なし)
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    # ★ 出力先ディレクトリ名を変更
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_binned_txt') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 生の粒子データ入力元: {data_dir} ---")
    print(f"--- 2DマップTXTデータ出力先 (ベース): {OUTPUT_DIR} ---")
    
    # ★ 制動放射は電子のみが対象
    species_list = [('e', 'electron')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        for suffix, species_label in species_list:
            
            # ★ ご提示いただいた "動作する" スクリプトのファイル名指定をそのまま使用
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            print(f"\n--- {species_label} データ ({filename}) を処理中 ---")

            particle_data = load_text_data(filepath)
            
            if particle_data is None or particle_data.size == 0:
                print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
                continue
            
            print(f"  -> {len(particle_data)} 個の粒子を読み込みました。X線プロキシ (ビン別) を計算中...")

            # --- 1. ★ 粒子データからX線プロキシ (ビン別) を計算 ---
            binned_proxy_maps = calculate_xray_proxy_binned(particle_data)
            
            # --- 2. ★ 各ビンをそれぞれのサブディレクトリにテキストファイルとして保存 ---
            print(f"  -> TXTファイルに保存中...")
        
            # binned_proxy_maps は辞書 { 'bin_label': map_data }
            for bin_label, proxy_map in binned_proxy_maps.items():
                
                # (bin_label から e_min, e_max を復元)
                e_min_str, e_max_str = "N/A", "N/A"
                for e_min, e_max, label in ENERGY_BINS_KEV:
                    if label == bin_label:
                        e_min_str, e_max_str = str(e_min), str(e_max)
                        break
                
                # ★ ビンごとにサブディレクトリを作成 (例: .../bremsstrahlung_data_binned_txt/1keV_5keV/)
                output_bin_dir = os.path.join(OUTPUT_DIR, bin_label)
                os.makedirs(output_bin_dir, exist_ok=True)
                
                output_filename = os.path.join(output_bin_dir, f'xray_proxy_{timestep}_{bin_label}.txt')
                
                header_txt = (
                    f'Soft X-ray Proxy Map (Particle count)\n'
                    f'Energy Bin: [{e_min_str}, {e_max_str}) keV (Label: {bin_label})\n'
                    f'Shape: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})'
                )
                
                # 粒子数は整数 (%d) で保存
                np.savetxt(output_filename, proxy_map, 
                           header=header_txt,
                           fmt='%d') 
                           
                print(f"    -> {bin_label} マップを {output_filename} に保存しました。 (Max: {np.max(proxy_map)})")

            
            print(f"--- タイムステップ {timestep} の {species_label} データ抽出・保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()