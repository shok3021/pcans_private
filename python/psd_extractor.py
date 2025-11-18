import numpy as np
import os
import glob
import sys
import time

# =======================================================
# 設定 (Fortran const モジュールから取得した正確な値)
# =======================================================
# ★★★ Fortran const モジュールからの値を使用 ★★★
GLOBAL_NX_GRID_POINTS = 321  
GLOBAL_NY_GRID_POINTS = 640

# 物理領域のグリッド数 (セル数: Grid Points - 1)
GLOBAL_NX_PHYS = GLOBAL_NX_GRID_POINTS - 1 # 320 セル
GLOBAL_NY_PHYS = GLOBAL_NY_GRID_POINTS - 1 # 639 セル
DELX = 1.0 

# ★★★ 座標範囲の仮修正 (Fortranの真の座標系に合わせるべきです) ★★★
# ここでは、Fortranの座標系が [0.0, 320.0] x [0.0, 639.0] であると仮定します。
X_MIN = 0.0           
X_MAX = GLOBAL_NX_PHYS * DELX # -> 320.0
Y_MIN = 0.0           
Y_MAX = GLOBAL_NY_PHYS * DELX # -> 639.0

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
    x_bins = np.linspace(x_min, x_max, NX + 1)
    y_bins = np.linspace(y_min, y_max, NY + 1)

    bin_x = np.digitize(X_pos, x_bins)
    bin_y = np.digitize(Y_pos, y_bins)

    ix = np.clip(bin_x - 1, 0, NX - 1)
    iy = np.clip(bin_y - 1, 0, NY - 1)

    # 粒子が空間グリッド範囲内にあるかのマスクを作成
    mask = (X_pos >= x_min) & (X_pos <= x_max) & \
           (Y_pos >= y_min) & (Y_pos <= y_max)

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
    if N_total > 0:
        print(f"  X-pos min/max (粒子): {np.min(X_pos):.3f} / {np.max(X_pos):.3f}")
        print(f"  Y-pos min/max (粒子): {np.min(Y_pos):.3f} / {np.max(Y_pos):.3f}")
    else:
        print("  粒子データが空です。")
    print(f"  全粒子数: {N_total}, マスクされた粒子数 (集計対象): {N_masked}")

    if N_masked == 0:
        if N_total > 0:
            print("  -> **警告: マスクされた粒子がゼロです。グリッド範囲と粒子座標が一致していません。**")
        density = np.zeros((NY, NX))
        vx_sum = np.zeros((NY, NX))
        vy_sum = np.zeros((NY, NX))
        vz_sum = np.zeros((NY, NX))
    else:
        print(f"  IX_masked min/max: {np.min(ix_masked)} / {np.max(ix_masked)} (NX={NX})")
        print(f"  IY_masked min/max: {np.min(iy_masked)} / {np.max(iy_masked)} (NY={NY})")

        density = np.zeros((NY, NX))
        vx_sum = np.zeros((NY, NX))
        vy_sum = np.zeros((NY, NX))
        vz_sum = np.zeros((NY, NX))

        np.add.at(density, (iy_masked, ix_masked), 1)
        np.add.at(vx_sum, (iy_masked, ix_masked), vx_masked)
        np.add.at(vy_sum, (iy_masked, ix_masked), vy_masked)
        np.add.at(vz_sum, (iy_masked, ix_masked), vz_masked)

    density_safe = np.where(density > 0, density, 1e-12)
    average_vx = vx_sum / density_safe
    average_vy = vy_sum / density_safe
    average_vz = vz_sum / density_safe
    
    # 粒子が0だったセルは 0.0 に戻す
    average_vx[density == 0] = 0.0
    average_vy[density == 0] = 0.0
    average_vz[density == 0] = 0.0

    return density, average_vx, average_vy, average_vz

# --- (load_text_data および save_data_to_txt 関数は変更なし) ---
def load_text_data(filepath):
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

def save_data_to_txt(data_2d, label, timestep, species, out_dir, filename):
    output_file = os.path.join(out_dir, f'data_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> {species}の {label} データを {output_file} に保存しました。")


# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python psd_extractor_revised.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python psd_extractor_revised.py 0 14000 500")
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