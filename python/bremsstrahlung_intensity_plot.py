import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ★ ユーザー設定: ディレクトリとパス (環境に合わせて変更してください)
# =======================================================
# 1. Bremsstrahlung計算スクリプトが出力したディレクトリ
BREMS_DATA_DIR = os.path.join(os.path.abspath('.'), 'bremsstrahlung_detailed_intensity')

# 2. 磁場データ(Bx, By)があるディレクトリ (磁力線描画用)
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data')

# 3. パラメータファイル (規格化定数取得用)
PARAM_FILE_PATH = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'

# 4. 出力先ディレクトリ
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'bremsstrahlung_plots')

# エネルギービンの定義 (計算スクリプトと一致させる)
ENERGY_BINS = [
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

# 粒子タイプ
PARTICLE_TYPES = ['Thermal', 'NonThermal']

# グリッド設定 (物理計算サイズ)
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# ★ カラーバーの固定レンジ設定
FIXED_VMIN = 0.0
FIXED_VMAX = 1.0e2  # 10^5

# =======================================================
# ヘルパー関数: パラメータ読み込み
# =======================================================
def load_simulation_parameters(param_filepath):
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = None, None, None, None, None, None, None
    print(f"Parameter file: {param_filepath}")
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                s = line.strip()
                parts = s.split()
                if s.startswith('dx, dt, c'):
                    DT, C_LIGHT = float(parts[5]), float(parts[6])
                elif s.startswith('Mi, Me'):
                    MI = float(parts[3])
                elif s.startswith('Qi, Qe'):
                    QI = float(parts[3])
                elif s.startswith('Fpe, Fge, Fpi Fgi'):
                    FPI, FGI = float(parts[7]), float(parts[8])
                elif s.startswith('Va, Vi, Ve'):
                    VA0 = float(parts[7])
    except Exception as e:
        print(f"Error loading params: {e}")
        sys.exit(1)
        
    if None in [C_LIGHT, FPI, DT, FGI, VA0, MI, QI]:
        print("Error: Failed to parse necessary parameters.")
        sys.exit(1)
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# パラメータロードと定数計算
try:
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
    DI = C_LIGHT / FPI
    B0 = (FGI * MI * C_LIGHT) / QI
    print(f"--- Normalization: d_i = {DI:.4f}, B0 = {B0:.4f}")
except:
    # ファイルがない場合のフォールバック（テスト用）
    print("Warning: Using fallback normalization parameters.")
    DI = 1.0
    B0 = 1.0
    DT = 0.1
    FGI = 0.1

# =======================================================
# データ読み込み関数
# =======================================================
def load_brems_txt(timestep, p_type, bin_label):
    """
    計算スクリプトの出力構造: output/Type/BinLabel/intensity_Type_BinLabel_timestep.txt
    """
    # フォルダ階層
    target_dir = os.path.join(BREMS_DATA_DIR, p_type, bin_label)
    filename = f'intensity_{p_type}_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    
    try:
        data = np.loadtxt(filepath)
        # 計算スクリプトの出力サイズが (NY+1, NX+1) か (NY, NX) かを確認し、形状を合わせる
        # 今回のターゲット: GLOBAL_NY_PHYS x GLOBAL_NX_PHYS
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            # サイズが合わない場合のリサイズ処理 (簡易的なスライス)
            # データが1大きい場合 (ヒストグラムの仕様など)
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                data = data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception as e:
        # ファイルが存在しない、または読み込めない場合はゼロ配列
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_field_data_for_lines(timestep):
    """
    磁力線を描くための Bx, By を読み込む
    """
    bx_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    by_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    
    try:
        bx = np.loadtxt(bx_path, delimiter=',')
        by = np.loadtxt(by_path, delimiter=',')
        # 形状チェック
        if bx.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS): bx = np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        if by.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS): by = np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return bx, by
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS)), np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def create_coordinates():
    x_phys = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS)
    y_phys = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS)
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    return np.meshgrid(x_norm, y_norm)

# =======================================================
# プロット関数
# =======================================================
def plot_brems_panel(ax, X, Y, Intensity, Bx, By, title, omega_t_str):
    """
    Bremsstrahlung強度をプロットし、磁力線を重ねる
    """
    # 1. 強度マップ (固定レンジ 0 ~ 10^5)
    cmap = plt.cm.inferno # 強度マップに適したカラーマップ (黒->赤->黄)
    levels = np.linspace(FIXED_VMIN, FIXED_VMAX, 100)
    
    cf = ax.contourf(X, Y, Intensity, levels=levels, cmap=cmap)
    
    # カラーバー設定
    cbar = plt.colorbar(cf, ax=ax, format='%.0e', ticks=np.linspace(FIXED_VMIN, FIXED_VMAX, 6))
    cbar.set_label(r'Intensity (Arbitrary Units)', fontsize=10)
    cbar.ax.yaxis.set_offset_position('left')
    
    # 2. 磁力線 (Psi) のオーバーレイ
    try:
        # 規格化前の生データ(Bx, By)を使ってPsiを計算
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
        # グレーで細く描画
        ax.contour(X, Y, Psi_local, levels=25, colors='gray', linewidths=0.5, alpha=0.7)
    except Exception as e:
        pass

    # 3. テキスト情報
    ax.text(0.98, 0.98, omega_t_str, 
            transform=ax.transAxes, fontsize=12, fontweight='bold', color='black',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))

    # 4. 軸ラベル・タイトル
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)

# =======================================================
# メイン処理
# =======================================================
def process_timestep(timestep_str, output_base_dir):
    ts_int = int(timestep_str)
    omega_t_val = FGI * float(ts_int) * DT
    omega_t_str = fr"$\Omega_{{ci}}t = {omega_t_val:.2f}$"
    
    print(f"Processing TS: {timestep_str} ({omega_t_str})")

    # 1. 磁場データの読み込み (共通)
    Bx, By = load_field_data_for_lines(timestep_str)
    
    # 2. 座標作成
    X, Y = create_coordinates()

    # 3. 各カテゴリ・各ビンでプロット
    for p_type in PARTICLE_TYPES:
        for bin_label in ENERGY_BINS:
            # データの読み込み
            intensity_map = load_brems_txt(timestep_str, p_type, bin_label)
            
            # データが全て0ならスキップするか、0のままプロットするか
            # ここでは0でもプロットを作成します（時系列比較のため）
            
            # 出力ディレクトリ作成
            # output/Thermal/001keV_100keV/ のように保存
            save_dir = os.path.join(output_base_dir, p_type, bin_label)
            os.makedirs(save_dir, exist_ok=True)
            
            # プロット作成
            fig, ax = plt.subplots(figsize=(10, 8))
            
            title = f"Bremsstrahlung: {p_type} [{bin_label}]"
            plot_brems_panel(ax, X, Y, intensity_map, Bx, By, title, omega_t_str)
            
            fig.tight_layout()
            
            out_name = f"plot_brems_{p_type}_{bin_label}_{timestep_str}.png"
            plt.savefig(os.path.join(save_dir, out_name), dpi=150)
            plt.close(fig)
            
    print(f"  -> Saved plots for {timestep_str}")

if __name__ == "__main__":
    # フォント設定 (LaTeX風)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'

    if len(sys.argv) < 4:
        print("Usage: python plot_brems_visual.py [start] [end] [step]")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = int(sys.argv[3])
    
    print(f"--- Visualizing Bremsstrahlung Maps ---")
    print(f"Data Source: {BREMS_DATA_DIR}")
    print(f"Field Source: {FIELD_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Range: {start} - {end} (step {step})")
    print(f"Colorbar Limit: {FIXED_VMIN} to {FIXED_VMAX}")

    for t in range(start, end + step, step):
        ts_str = f"{t:06d}"
        process_timestep(ts_str, OUTPUT_DIR)
        
    print("\nDone.")