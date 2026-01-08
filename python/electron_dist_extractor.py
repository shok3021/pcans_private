import numpy as np
import os
import sys

# =======================================================
# 設定
# =======================================================
IS_RAW_DATA_NORMALIZED = True
PARAM_FILE_PATH = os.path.join('/data/shok/dat/init_param.dat')

def load_grid_params(param_filepath):
    params = {'nx': 1601, 'ny': 640, 'delx': 0.2}
    if os.path.exists(param_filepath):
        try:
            with open(param_filepath, 'r') as f:
                for line in f:
                    if line.strip().startswith('grid size'):
                        parts = line.strip().split()
                        params['nx'] = int(parts[5].replace('x', ''))
                        params['ny'] = int(parts[6])
                    elif line.strip().startswith('dx, dt'):
                        params['delx'] = float(line.strip().split()[4])
        except: pass
    return params

p = load_grid_params(PARAM_FILE_PATH)
GLOBAL_NX, GLOBAL_NY = p['nx'] - 1, p['ny'] - 1
DELX = p['delx']
PHYS_W, PHYS_H = GLOBAL_NX * DELX, GLOBAL_NY * DELX

# =======================================================
# 計算エンジン
# =======================================================
def calculate_velocity_variance_tensor(particle_data):
    # 座標
    raw_X, raw_Y = particle_data[:, 0], particle_data[:, 1]
    X_pos = raw_X if IS_RAW_DATA_NORMALIZED else raw_X * DELX
    Y_pos = raw_Y if IS_RAW_DATA_NORMALIZED else raw_Y * DELX
    
    # 速度 (u = gamma * v だが、非相対論近似で vとして扱うか、
    # あるいは相対論的運動エネルギーなら (gamma-1)mc^2 を使うべきだが
    # ここではユーザー指定の mv^2/2 の直感に合わせ、u (空間成分) の分散を取る)
    ux, uy, uz = particle_data[:, 2], particle_data[:, 3], particle_data[:, 4]

    # ビニング
    x_bins = np.linspace(0.0, PHYS_W, GLOBAL_NX + 1)
    y_bins = np.linspace(0.0, PHYS_H, GLOBAL_NY + 1)

    def get_hist(w):
        H, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_bins, x_bins], weights=w)
        return H

    # 1次モーメント (和)
    H_n  = get_hist(None)
    H_ux = get_hist(ux)
    H_uy = get_hist(uy)
    H_uz = get_hist(uz)

    # 2次モーメント (積の和)
    H_uxux, H_uyuy, H_uzuz = get_hist(ux*ux), get_hist(uy*uy), get_hist(uz*uz)
    H_uxuy, H_uxuz, H_uyuz = get_hist(ux*uy), get_hist(ux*uz), get_hist(uy*uz)

    # 平均と分散の計算 <u_i u_j> - <u_i><u_j>
    with np.errstate(divide='ignore', invalid='ignore'):
        den = H_n.copy()
        den[den == 0] = 1.0
        
        av_ux, av_uy, av_uz = H_ux/den, H_uy/den, H_uz/den
        
        # これが速度分散テンソル (Velocity Dispersion Tensor)
        # Unit: (v/c)^2 (シミュレーション単位による)
        S_xx = (H_uxux / den) - (av_ux * av_ux)
        S_yy = (H_uyuy / den) - (av_uy * av_uy)
        S_zz = (H_uzuz / den) - (av_uz * av_uz)
        S_xy = (H_uxuy / den) - (av_ux * av_uy)
        S_xz = (H_uxuz / den) - (av_ux * av_uz)
        S_yz = (H_uyuz / den) - (av_uy * av_uz)

    # 粒子がいない場所は0
    mask = (H_n == 0)
    for arr in [S_xx, S_yy, S_zz, S_xy, S_xz, S_yz]: arr[mask] = 0.0

    return H_n, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz

def save_txt(data, sp, out_dir, tag, ts):
    path = os.path.join(out_dir, f'data_{ts}_{sp}_{tag}.txt')
    np.savetxt(path, data, fmt='%.6e', delimiter=',')

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 6:
        print("Usage: python tensor_extractor.py [start] [end] [step] [id1] [id2]")
        sys.exit(1)

    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fid1, fid2 = sys.argv[4], sys.argv[5]
    
    data_dir = '/data/shok/psd/'
    out_dir = os.path.join(os.path.dirname(__file__), 'extracted_moments_tensor')
    os.makedirs(out_dir, exist_ok=True)

    for ts_val in range(start, end + step, step):
        ts = f"{ts_val:06d}"
        print(f"Processing TS: {ts}")

        # 電子のみ (必要なら ion 追加)
        for sp_suffix, sp_name in [('e', 'electron')]:
            fname = f'{ts}_{fid1}-{fid2}_psd_{sp_suffix}.dat'
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path): continue

            try:
                raw = np.loadtxt(path)
                if raw.ndim == 1: raw = raw.reshape(1, -1)
            except: continue

            den, Sxx, Syy, Szz, Sxy, Sxz, Syz = calculate_velocity_variance_tensor(raw)

            # 保存 (ここでは単なる速度分散。エネルギー換算はプロット時に行う)
            save_txt(den, sp_name, out_dir, 'density', ts)
            save_txt(Sxx, sp_name, out_dir, 'Sxx', ts)
            save_txt(Syy, sp_name, out_dir, 'Syy', ts)
            save_txt(Szz, sp_name, out_dir, 'Szz', ts)
            save_txt(Sxy, sp_name, out_dir, 'Sxy', ts)
            save_txt(Sxz, sp_name, out_dir, 'Sxz', ts)
            save_txt(Syz, sp_name, out_dir, 'Syz', ts)
            
            print(f"  Saved Tensor: {sp_name}")

if __name__ == "__main__":
    main()