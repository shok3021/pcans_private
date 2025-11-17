import numpy as np
import os
import glob
import sys
import time
# =======================================================
# ★ ステップ1: init_param.dat 読み込み関数 (前回からコピー)
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    init_param.dat を読み込み、必要なパラメータを抽出する。
    (C_LIGHT, FPI, DT, FGI, VA0, MI, QI に加え、
     ★ NX_PHYS, NY_PHYS, DELX を抽出する機能を追加 ★)
    """
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = (None,) * 7
    NX_PHYS, NY_PHYS, DELX = (None,) * 3

    print(f"パラメータファイルを読み込み中: {param_filepath}")
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()

                # --- (C, FPI, DT, FGI, VA0, MI, QI の抽出 ... 省略) ---
                # (前回の visual_fields.py からコピーしてください)
                
                # ★ グリッドサイズ (nx, ny) とセル幅 (dx) の抽出を追加 ★
                if stripped_line.startswith('nx, ny'):
                    try:
                        parts = stripped_line.split()
                        # 'nx = 321' の '321' を取得
                        NX_GRID_POINTS = int(parts[2]) 
                        # 'ny = 640' の '640' を取得
                        NY_GRID_POINTS = int(parts[5]) 
                        
                        # Fortranのセル数 (NX_PHYS) は グリッドポイント数 - 1
                        NX_PHYS = NX_GRID_POINTS - 1
                        NY_PHYS = NY_GRID_POINTS - 1
                        print(f"  -> 'nx' (Grid Points) を検出: {NX_GRID_POINTS} (セル数: {NX_PHYS})")
                        print(f"  -> 'ny' (Grid Points) を検出: {NY_GRID_POINTS} (セル数: {NY_PHYS})")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'nx, ny' の値の解析に失敗。")

                if stripped_line.startswith('dx, dt, c'):
                    try:
                        parts = stripped_line.split()
                        DELX = float(parts[2]) # 3番目の要素 (dx)
                        DT = float(parts[5])      # 6番目の要素 (dt)
                        C_LIGHT = float(parts[6]) # 7番目の要素 (c)
                        print(f"  -> 'dx' (DELX) の値を検出: {DELX}")
                        print(f"  -> 'dt' の値を検出: {DT}")
                        print(f"  -> 'c' の値を検出: {C_LIGHT}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'dx, dt, c' の値の解析に失敗。")
        
        # (エラーチェックに NX_PHYS, NY_PHYS, DELX を追加)
        if NX_PHYS is None or NY_PHYS is None or DELX is None:
            print("★★ エラー: グリッドパラメータ ('nx', 'ny', 'dx') を抽出できませんでした。")
            sys.exit(1)
            
        # (C_LIGHT など、他のパラメータのエラーチェックもここで行う)
            
        return NX_PHYS, NY_PHYS, DELX # ★ 返り値に追加

    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        sys.exit(1)

# =======================================================
# ★ ステップ2 & 3: グローバル定数を動的に設定
# =======================================================

# --- init_param.dat のパスを指定 ---
# (visual_fields.py と同じパスを指定してください)
PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# --- パラメータの読み込み ---
try:
    GLOBAL_NX_PHYS, GLOBAL_NY_PHYS, DELX = load_simulation_parameters(PARAM_FILE_PATH)
except Exception as e:
    print(f"init_param.dat の読み込みに失敗しました: {e}")
    print("ハードコードされた値で続行しますが、不正確な可能性があります。")
    # (フォールバック)
    GLOBAL_NX_PHYS = 320 
    GLOBAL_NY_PHYS = 639
    DELX = 1.0
    
# ★★★ 座標範囲の仮修正 (Fortranの真の座標系に合わせるべきです) ★★★
# ここでは、Fortranの座標系が [0.0, 320.0] x [0.0, 639.0] であると仮定します。
X_MIN = 0.0            
X_MAX = GLOBAL_NX_PHYS * DELX 
Y_MIN = 0.0            
Y_MAX = GLOBAL_NY_PHYS * DELX

print(f"--- グリッド設定 (修正案) ---")
print(f"X方向物理セル数: {GLOBAL_NX_PHYS}, Y方向物理セル数: {GLOBAL_NY_PHYS}")
print(f"空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}] (セル幅: {DELX})")

# =======================================================
# データ抽出・計算関数 (GLOBAL定数を更新)
# =======================================================
def calculate_moments_from_particle_list(particle_data):
    """
    粒子の生データ (X, Y, Vx, Vy, Vz) から空間グリッド上の平均速度を計算する。
    """
    
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    
    # 空間グリッドの範囲
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    # 粒子データの各列
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    Vx_raw = particle_data[:, 2]
    Vy_raw = particle_data[:, 3]
    Vz_raw = particle_data[:, 4]
    
    N_total = len(X_pos) # 全粒子数

    # --- インデックス計算 ---
    
    # N 個のセルを区切る N+1 個の境界 (x_min, ..., x_max) を生成
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    # np.digitizeでインデックスを計算
    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)

    # 物理セルインデックス (0 <= index < N) にクリップ
    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    # 粒子が空間グリッド範囲内にあるかのマスクを作成
    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)
           
    # 修正後のインデックスと粒子データにマスクを適用
    ix_masked = ix[mask]
    iy_masked = iy[mask]
    vx_masked = Vx_raw[mask]
    vy_masked = Vy_raw[mask]
    vz_masked = Vz_raw[mask]
    
    N_masked = len(ix_masked)
    
    # =======================================================
    # ★★★ デバッグ出力の追加 ★★★
    # =======================================================
    print("  --- デバッグ情報 ---")
    print(f"  X-Range (設定): [{x_min}, {x_max}], Y-Range (設定): [{y_min}, {y_max}]")
    print(f"  X-pos min/max (粒子): {np.min(X_pos):.3f} / {np.max(X_pos):.3f}")
    print(f"  Y-pos min/max (粒子): {np.min(Y_pos):.3f} / {np.max(Y_pos):.3f}")
    print(f"  全粒子数: {N_total}, マスクされた粒子数 (集計対象): {N_masked}")
    
    if N_masked == 0:
        print("  -> **致命的警告: マスクされた粒子がゼロです。グリッド範囲と粒子座標が一致していません。**")
        # 0の配列を返して処理を継続
        density = np.zeros((NY, NX))
        vx_sum = np.zeros((NY, NX))
        vy_sum = np.zeros((NY, NX))
        vz_sum = np.zeros((NY, NX))
    else:
        # マスクされた粒子のインデックスの最小値と最大値
        print(f"  IX_masked min/max: {np.min(ix_masked)} / {np.max(ix_masked)} (NX={NX})")
        print(f"  IY_masked min/max: {np.min(iy_masked)} / {np.max(iy_masked)} (NY={NY})")
        
        # 空間グリッド (NY, NX) を作成
        density = np.zeros((NY, NX))
        vx_sum = np.zeros((NY, NX))
        vy_sum = np.zeros((NY, NX))
        vz_sum = np.zeros((NY, NX))
        
        # 各粒子を対応するグリッドセルに集計 (ベクトル化)
        np.add.at(density, (iy_masked, ix_masked), 1)
        np.add.at(vx_sum, (iy_masked, ix_masked), vx_masked)
        np.add.at(vy_sum, (iy_masked, ix_masked), vy_masked)
        np.add.at(vz_sum, (iy_masked, ix_masked), vz_masked)
    
    # 平均速度を計算 (ゼロ除算回避)
    density_safe = np.where(density > 0, density, 1e-12)
    
    average_vx = vx_sum / density_safe
    average_vy = vy_sum / density_safe
    average_vz = vz_sum / density_safe
    
    return density, average_vx, average_vy, average_vz

# --- (load_text_data および save_data_to_txt 関数は変更なし) ---
def load_text_data(filepath):
    # ... (変更なし)
    if not os.path.exists(filepath):
        return None
        
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"    エラー: {filepath} のテキスト読み込みに失敗: {e}")
        return None

def save_data_to_txt(data_2d, label, timestep, species, out_dir, filename):
    # ... (変更なし)
    output_file = os.path.join(out_dir, f'data_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> {species}の {label} データを {output_file} に保存しました。")


# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python psd_extractor_revised.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python psd_extractor_revised.py 000000 014000 500")
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
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 出力先ディレクトリ: {OUTPUT_DIR} ---")
    
    species_list = [('e', 'electron'), ('i', 'ion')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        for suffix, species_label in species_list:
            
            filename = f'{timestep}_0160-0320_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            print(f"\n--- {species_label} データ ({filename}) を処理中 ---")

            particle_data = load_text_data(filepath)
            
            if particle_data is None or particle_data.size == 0:
                print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
                continue
            
            print(f"  -> {len(particle_data)} 個の粒子を読み込みました。モーメントを計算中...")

            # --- 1. 粒子データからモーメント (平均速度) を計算し、空間グリッドにマップ ---
            # 粒子データの各列
            X_pos = particle_data[:, 0]
            Y_pos = particle_data[:, 1]
            Vx_raw = particle_data[:, 2]
            Vy_raw = particle_data[:, 3]
            Vz_raw = particle_data[:, 4]

            # calculate_moments_from_particle_list の実行
            density, average_vx, average_vy, average_vz = calculate_moments_from_particle_list(particle_data)
            
            # --- 2. 各物理量をテキストファイルに保存 ---
            save_data_to_txt(density, 'Particle Count (Density Proxy)', 
                             timestep, species_label, OUTPUT_DIR, 'density_count')
            save_data_to_txt(average_vx, 'Average Velocity (Vx)', 
                             timestep, species_label, OUTPUT_DIR, 'Vx')
            save_data_to_txt(average_vy, 'Average Velocity (Vy)', 
                             timestep, species_label, OUTPUT_DIR, 'Vy')
            save_data_to_txt(average_vz, 'Average Velocity (Vz)', 
                             timestep, species_label, OUTPUT_DIR, 'Vz')
            
            print(f"--- タイムステップ {timestep} の {species_label} データ抽出・保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()