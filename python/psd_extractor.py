import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge

# =======================================================
# 設定: データ読み込みの制御
# =======================================================
# Fortranが出力した粒子位置データが「物理座標(0~320)」なら True
# 「グリッド番号(0~1600)」なら False にしてください。
# ★左下に縮まるなら、ここを True に変えてください★
IS_RAW_DATA_NORMALIZED = False 

PARAM_FILE_PATH = os.path.join('/data/shok/dat/init_param.dat')
REST_MASS_E_EV = (m_e * c**2) / elementary_charge

# =======================================================
# パラメータ読み込み (Clean版)
# =======================================================
def load_grid_params(param_filepath):
    """ init_param.dat からシミュレーション設定を読み込む """
    # デフォルト値
    params = {'nx': 1601, 'ny': 640, 'delx': 0.2}
    
    if not os.path.exists(param_filepath):
        print(f"警告: {param_filepath} が見つかりません。デフォルト値を使います。")
        return params

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('grid size'):
                    parts = stripped.split()
                    params['nx'] = int(parts[5].replace('x', ''))
                    params['ny'] = int(parts[6])
                elif stripped.startswith('dx, dt'):
                    parts = stripped.split()
                    params['delx'] = float(parts[4])
    except Exception as e:
        print(f"パラメータ読込エラー: {e}")
        
    return params

# =======================================================
# 定数設定
# =======================================================
p = load_grid_params(PARAM_FILE_PATH)

GLOBAL_NX = p['nx'] - 1  # 物理セル数 (1600)
GLOBAL_NY = p['ny'] - 1  # 物理セル数 (639)
DELX      = p['delx']

# ヒストグラムを作成する範囲 (常に物理サイズで定義)
PHYS_W = GLOBAL_NX * DELX  # 320.0
PHYS_H = GLOBAL_NY * DELX  # 127.8

print(f"--- 設定: Grid=[{GLOBAL_NX}x{GLOBAL_NY}], dx={DELX} ---")
print(f"--- 物理領域: X=[0, {PHYS_W:.1f}], Y=[0, {PHYS_H:.1f}] ---")

# =======================================================
# 計算エンジン
# =======================================================
def calculate_moments(particle_data):
    # 1. 座標データの取得
    raw_X = particle_data[:, 0]
    raw_Y = particle_data[:, 1]
    
    # 2. 単位の正規化 (ここが重要)
    # ヒストグラムは「物理座標」の枠で作るため、データを物理座標に合わせる
    if IS_RAW_DATA_NORMALIZED:
        # すでに物理座標なら何もしない
        X_pos = raw_X
        Y_pos = raw_Y
    else:
        # グリッド番号なら、delxを掛けて物理座標にする
        X_pos = raw_X * DELX
        Y_pos = raw_Y * DELX

    ux, uy, uz = particle_data[:, 2], particle_data[:, 3], particle_data[:, 4]
    
    # エネルギー計算 (相対論)
    gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    E_kin = gamma - 1.0 

    # 3. ヒストグラム(ビン)の設定
    # 物理領域全体を、グリッド数分のビンで分割する
    x_bins = np.linspace(0.0, PHYS_W, GLOBAL_NX + 1)
    y_bins = np.linspace(0.0, PHYS_H, GLOBAL_NY + 1)

    # 4. ビニング実行 (物理座標空間で行う)
    # weightsを指定することで、その場所にある粒子の物理量を積算する
    H_count, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins])
    H_sum_ux, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins], weights=ux)
    H_sum_uy, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins], weights=uy)
    H_sum_uz, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins], weights=uz)
    H_sum_E, _, _  = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins], weights=E_kin)

    # 5. 平均値の計算 (ゼロ除算回避)
    with np.errstate(divide='ignore', invalid='ignore'):
        density = H_count.copy()
        density[density == 0] = 1.0 # 割り算用
        
        av_ux = H_sum_ux / density
        av_uy = H_sum_uy / density
        av_uz = H_sum_uz / density
        mean_E = H_sum_E / density

    # 温度計算
    u_bulk_sq = av_ux**2 + av_uy**2 + av_uz**2
    gamma_fluid = np.sqrt(1.0 + u_bulk_sq)
    E_bulk = gamma_fluid - 1.0
    E_thermal = np.clip(mean_E - E_bulk, 0.0, None)
    T_norm = (2.0 / 3.0) * E_thermal

    # データがない場所はゼロ埋め
    mask_zero = (H_count == 0)
    for arr in [av_ux, av_uy, av_uz, T_norm]:
        arr[mask_zero] = 0.0
    
    # 速度を光速単位に変換して返す (u = gamma * v => v = u / gamma)
    return H_count, av_ux/gamma_fluid, av_uy/gamma_fluid, av_uz/gamma_fluid, T_norm

# =======================================================
# IO関数
# =======================================================
def load_mi():
    try:
        with open(PARAM_FILE_PATH, 'r') as f:
            for line in f:
                if 'Mi, Me' in line: return float(line.split()[3])
    except: pass
    return 100.0

def save_txt(data, name, ts, sp, out_dir, tag):
    path = os.path.join(out_dir, f'data_{ts}_{sp}_{tag}.txt')
    np.savetxt(path, data, fmt='%.6e', delimiter=',')
    print(f"    Saved: {name}")

# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 6:
        print("Usage: python psd_extractor.py [start] [end] [step] [id1] [id2]")
        sys.exit(1)

    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fid1, fid2 = sys.argv[4], sys.argv[5]

    data_dir = os.path.join('/data/shok/psd/')
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    out_dir = os.path.join(script_dir, 'extracted_psd_data_moments')
    os.makedirs(out_dir, exist_ok=True)

    mi_ratio = load_mi()
    print(f"--- Ion Mass Ratio: {mi_ratio} ---")
    print(f"--- Mode: {'Normalized (0-320)' if IS_RAW_DATA_NORMALIZED else 'Grid (0-1600)'} ---")

    for ts_val in range(start, end + step, step):
        ts = f"{ts_val:06d}"
        print(f"\n=== Processing TS: {ts} ===")

        for sp_suffix, sp_name in [('e', 'electron'), ('i', 'ion')]:
            fname = f'{ts}_{fid1}-{fid2}_psd_{sp_suffix}.dat'
            path = os.path.join(data_dir, fname)
            
            if not os.path.exists(path):
                print(f"  Skip: {sp_name} (No file)")
                continue

            try:
                raw_data = np.loadtxt(path)
                if raw_data.size == 0: raise ValueError("Empty")
                if raw_data.ndim == 1: raw_data = raw_data.reshape(1, -1)
            except:
                print(f"  Skip: {sp_name} (Load Error)")
                continue

            den, vx, vy, vz, t_norm = calculate_moments(raw_data)
            
            # 温度のeV変換
            t_ev = t_norm * REST_MASS_E_EV * (mi_ratio if sp_name == 'ion' else 1.0)

            save_txt(den, 'Density', ts, sp_name, out_dir, 'density_count')
            save_txt(vx,  'Vx', ts, sp_name, out_dir, 'Vx')
            save_txt(vy,  'Vy', ts, sp_name, out_dir, 'Vy')
            save_txt(vz,  'Vz', ts, sp_name, out_dir, 'Vz')
            save_txt(t_ev, 'Temp(eV)', ts, sp_name, out_dir, 'T')

    print("\nDone.")

if __name__ == "__main__":
    main()