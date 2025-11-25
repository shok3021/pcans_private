import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Scipy互換性
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# =======================================================
# ★ ユーザー設定
# =======================================================
# 計算コードの出力先 (bremsstrahlung_photon_intensity)
INPUT_DATA_DIR = os.path.join(os.path.abspath('.'), 'bremsstrahlung_photon_intensity')
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data')
OUTPUT_PLOT_DIR = os.path.join(os.path.abspath('.'), 'photon_map_plots')

# パラメータファイル
PARAM_FILE_PATH = './dat/init_param.dat' 

# ★ ここが重要: 計算コードで定義した PHOTON_BINS の名前と合わせる
TARGET_BINS = [
    '001keV_100keV',
    '100keV_200keV',
    '200keV_500keV',
    '500keV_1000keV',
    '1000keV_2000keV',
    '2000keV_5000keV',
    '5000keV_10000keV',
    '10000keV_20000keV',
    '20000keV_50000keV',
    '50000keV_over'
]

# グリッド設定
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# カラーバースケール
FIXED_SCALE = False 
FIXED_VMIN = 0.0
FIXED_VMAX = 1.0e-2

# =======================================================
# 1. データ読み込みヘルパー
# =======================================================
def load_simulation_parameters(param_filepath):
    params = {'C_LIGHT': 1.0, 'FPI': 1.0, 'DT': 1.0, 'FGI': 0.1, 'MI': 1.0, 'QI': 1.0}
    if not os.path.exists(param_filepath): return params
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if 'dx, dt, c' in line:
                    params['DT'] = float(parts[5]); params['C_LIGHT'] = float(parts[6])
                elif 'Fpe, Fge' in line:
                    params['FPI'] = float(parts[7]); params['FGI'] = float(parts[8])
    except: pass
    return params

def load_photon_map(timestep, bin_label):
    """
    光子強度マップ(txt)を読み込む
    パス構成: INPUT_DIR / BinLabel / photon_intensity_BinLabel_timestep.txt
    """
    target_dir = os.path.join(INPUT_DATA_DIR, bin_label)
    filename = f'photon_intensity_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    
    if not os.path.exists(filepath):
        # ファイルがない場合はNoneを返す
        return None
        
    try:
        data = np.loadtxt(filepath)
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                return data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_field_data(timestep):
    bx_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    by_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    if not os.path.exists(bx_path): return None, None
    try:
        bx = np.loadtxt(bx_path, delimiter=',')[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        by = np.loadtxt(by_path, delimiter=',')[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        return bx, by
    except: return None, None

# =======================================================
# 2. プロット描画
# =======================================================
def plot_timestep(timestep_str, params):
    c_light = params['C_LIGHT']
    fpi = params['FPI']
    d_i = c_light / fpi if fpi != 0 else 1.0
    
    # 時刻表示
    omega_t = params['FGI'] * float(timestep_str) * params['DT']
    time_label = fr"$\Omega_{{ci}}t = {omega_t:.2f}$"

    # 座標
    x = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS) / d_i
    y = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / d_i
    X, Y = np.meshgrid(x, y)

    # 磁場 (ベクトルポテンシャル)
    Bx, By = load_field_data(timestep_str)
    Psi = None
    if Bx is not None:
        Psi = cumtrapz(By, dx=1.0, axis=1, initial=0)

    # --- 各エネルギーバンドごとにプロット ---
    for bin_label in TARGET_BINS:
        data = load_photon_map(timestep_str, bin_label)
        
        if data is None:
            # ファイルが見つからない場合
            continue
            
        max_val = np.max(data)
        if max_val == 0:
            print(f"  [Skip] {bin_label}: All zero.")
            continue

        # スケール設定
        vmin = FIXED_VMIN if FIXED_SCALE else 0.0
        vmax = FIXED_VMAX if FIXED_SCALE else max_val * 1.05

        # 図の作成
        fig, ax = plt.subplots(figsize=(7, 10))
        
        # コンタープロット
        levels = np.linspace(vmin, vmax, 100)
        im = ax.contourf(X, Y, data, levels=levels, cmap='inferno', extend='max')
        
        # 磁力線
        if Psi is not None:
            ax.contour(X, Y, Psi, levels=25, colors='white', linewidths=0.5, alpha=0.5)
        
        # 装飾
        ax.set_title(f"X-ray Intensity: {bin_label}\n{time_label}", fontsize=14)
        ax.set_xlabel(r'$x / d_i$', fontsize=12)
        ax.set_ylabel(r'$y / d_i$', fontsize=12)
        ax.set_aspect('equal')
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Photon Flux (Arb. Unit)', fontsize=12)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        # 保存
        out_folder = os.path.join(OUTPUT_PLOT_DIR, bin_label)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f'photon_{bin_label}_{timestep_str}.png')
        
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
    print(f"  -> Processed TS: {timestep_str}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_photon_maps.py [start] [end] [step]")
        sys.exit(1)
        
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    params = load_simulation_parameters(PARAM_FILE_PATH)
    
    print(f"--- Plotting Photon Maps ---")
    print(f"Input: {INPUT_DATA_DIR}")
    print(f"Output: {OUTPUT_PLOT_DIR}")
    
    for t_int in range(start, end + step, step):
        ts_str = f"{t_int:06d}"
        plot_timestep(ts_str, params)

if __name__ == "__main__":
    main()