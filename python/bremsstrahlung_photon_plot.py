import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Scipyのバージョン互換性対応
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# =======================================================
# ★ ユーザー設定
# =======================================================
# ※ 先ほどの計算コードの出力先と合わせる
INPUT_DATA_DIR = os.path.join(os.path.abspath('.'), 'bremsstrahlung_photon_intensity') # or 'bremsstrahlung_ion_weighted'
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data') # 磁場データがある場所
OUTPUT_PLOT_DIR = os.path.join(os.path.abspath('.'), 'bremsstrahlung_plots_combined')

# パラメータファイル (なければデフォルト値で動きます)
PARAM_FILE_PATH = './dat/init_param.dat' 

# エネルギービン (計算コードと同じ定義)
ENERGY_BINS = [
    '001keV_100keV', '100keV_200keV', '200keV_500keV',
    '500keV_1000keV', '1000keV_2000keV', '2000keV_5000keV',
    '5000keV_10000keV', '10000keV_20000keV', '20000keV_50000keV',
    '50000keV_over'
]

# グリッド物理サイズ
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# ★ カラーバースケール設定
# Trueなら下のFIXED値を使用。Falseならその時刻の最大値で自動調整。
FIXED_SCALE = False 
FIXED_VMIN = 0.0
FIXED_VMAX = 1.0e-2  # ※計算モデルが変わったので値のオーダーに注意！

# =======================================================
# 1. データ読み込みヘルパー
# =======================================================
def load_simulation_parameters(param_filepath):
    """パラメータファイルを読み込む。失敗したらデフォルト値を返す"""
    # デフォルト値
    params = {'C_LIGHT': 1.0, 'FPI': 1.0, 'DT': 1.0, 'FGI': 0.1, 'MI': 1.0, 'QI': 1.0}
    
    if not os.path.exists(param_filepath):
        return params

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if 'dx, dt, c' in line:
                    params['DT'] = float(parts[5])
                    params['C_LIGHT'] = float(parts[6])
                elif 'Mi, Me' in line:
                    params['MI'] = float(parts[3])
                elif 'Qi, Qe' in line:
                    params['QI'] = float(parts[3])
                elif 'Fpe, Fge' in line:
                    params['FPI'] = float(parts[7])
                    params['FGI'] = float(parts[8])
    except Exception as e:
        print(f"Warning: Param read error ({e}). Using defaults.")
    
    return params

def load_brems_map(timestep, p_type, bin_label):
    """強度マップ(txt)を読み込む"""
    # フォルダ構成: root / Type / BinLabel / intensity_Type_Bin_timestep.txt
    target_dir = os.path.join(INPUT_DATA_DIR, p_type, bin_label)
    filename = f'intensity_{p_type}_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    
    if not os.path.exists(filepath):
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        
    try:
        data = np.loadtxt(filepath)
        # シェイプ補正
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                return data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_field_data(timestep):
    """磁場データを読み込む (Bx, By)"""
    bx_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    by_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    
    if not os.path.exists(bx_path) or not os.path.exists(by_path):
        return None, None
        
    try:
        bx = np.loadtxt(bx_path, delimiter=',')
        by = np.loadtxt(by_path, delimiter=',')
        # シェイプ補正
        bx = bx[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        by = by[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
        return bx, by
    except:
        return None, None

# =======================================================
# 2. プロット描画コア関数
# =======================================================
def plot_combined_timestep(timestep_str, params):
    """指定されたタイムステップの3パネルプロットを作成・保存する"""
    
    # 物理定数・正規化
    c_light = params['C_LIGHT']
    fpi = params['FPI']
    dt = params['DT']
    fgi = params['FGI']
    mi = params['MI']
    qi = params['QI']
    
    d_i = c_light / fpi if fpi != 0 else 1.0
    omega_t = fgi * float(timestep_str) * dt
    time_label = fr"$\Omega_{{ci}}t = {omega_t:.2f}$"

    # 座標作成
    x = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS) / d_i
    y = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS) / d_i
    X, Y = np.meshgrid(x, y)

    # 磁場データ (あればポテンシャル計算)
    Bx, By = load_field_data(timestep_str)
    Psi = None
    if Bx is not None and By is not None:
        # ベクトルポテンシャル Az (By = dAz/dx -> Az = int By dx)
        # cumtrapzは積分定数が不定だが、磁力線を見るだけなら相対値でOK
        Psi = cumtrapz(By, dx=1.0, axis=1, initial=0)

    # --- ビンごとのループ ---
    for bin_label in ENERGY_BINS:
        # データ読み込み
        map_th = load_brems_map(timestep_str, 'Thermal', bin_label)
        map_nt = load_brems_map(timestep_str, 'NonThermal', bin_label)
        map_total = map_th + map_nt
        
        # データが全部ゼロならスキップ（またはWarning）
        max_val = np.max(map_total)
        if max_val == 0:
            print(f"  [Skip] {bin_label}: All zero.")
            continue

        # カラーバーのレンジ決定
        if FIXED_SCALE:
            vmin, vmax = FIXED_VMIN, FIXED_VMAX
        else:
            vmin = 0.0
            vmax = max_val * 1.05 # 少し余裕を持たせる

        # --- プロット作成 (1行3列) ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0.1, right=0.9)
        
        fig.suptitle(f"Bremsstrahlung Intensity: {bin_label}\n{time_label}", fontsize=16)
        
        dataset = [
            ('Thermal', map_th),
            ('Non-Thermal', map_nt),
            ('Total', map_total)
        ]
        
        im = None
        for i, (title, data) in enumerate(dataset):
            ax = axes[i]
            
            # コンタープロット
            # levelsを明示的に指定して、色の境界を滑らかに
            levels = np.linspace(vmin, vmax, 100)
            im = ax.contourf(X, Y, data, levels=levels, cmap='inferno', extend='max')
            
            # 磁力線 (あれば)
            if Psi is not None:
                ax.contour(X, Y, Psi, levels=20, colors='white', linewidths=0.5, alpha=0.5)
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(r'$x / d_i$', fontsize=12)
            if i == 0:
                ax.set_ylabel(r'$y / d_i$', fontsize=12)
            
            # アスペクト比を物理座標に合わせる
            ax.set_aspect('equal')

        # --- 共通カラーバー ---
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity (Arb. Unit)', fontsize=12)
        # 指数表記にする
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        # --- 保存 ---
        out_folder = os.path.join(OUTPUT_PLOT_DIR, bin_label)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f'combined_{bin_label}_{timestep_str}.png')
        
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  -> Processed TS: {timestep_str}")

# =======================================================
# メイン
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_brems_distribution.py [start] [end] [step]")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = int(sys.argv[3])
    
    # パラメータ読み込み
    params = load_simulation_parameters(PARAM_FILE_PATH)
    
    print("--- Plotting Bremsstrahlung Maps (Combined) ---")
    print(f"Input Dir: {INPUT_DATA_DIR}")
    print(f"Output Dir: {OUTPUT_PLOT_DIR}")
    print(f"Scale: {'Fixed' if FIXED_SCALE else 'Dynamic (Auto)'}")
    
    for t_int in range(start, end + step, step):
        ts_str = f"{t_int:06d}"
        plot_combined_timestep(ts_str, params)

    print("\nAll Done.")

if __name__ == "__main__":
    main()