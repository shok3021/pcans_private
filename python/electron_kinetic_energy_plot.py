import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz
# カラーバー配置調整用のモジュール
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =======================================================
# ★ ユーザー設定: ディレクトリとパス (環境に合わせて変更してください)
# =======================================================
BREMS_DATA_DIR = os.path.join(os.path.abspath('.'), 'electron_energy_density')
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data')
PARAM_FILE_PATH = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'electron_energy_plots') # 出力先フォルダ名を変更

# エネルギービンの定義
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

# グリッド設定
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# ★ カラーバーの固定レンジ設定
FIXED_VMIN = 0.0
FIXED_VMAX = 1.0e7  # 10^7

# =======================================================
# ヘルパー関数: パラメータ読み込み & データ読み込み (変更なし)
# =======================================================
def load_simulation_parameters(param_filepath):
    C_LIGHT, FPI, DT, FGI, VA0, MI, QI = None, None, None, None, None, None, None
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
        print(f"Error loading params: {e}. Using defaults.")
        return 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0 # Fallback setting
        
    if None in [C_LIGHT, FPI, DT, FGI, VA0, MI, QI]:
        print("Error: Failed to parse parameters. Using defaults.")
        return 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0 # Fallback setting
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI if FPI != 0 else 1.0
B0 = (FGI * MI * C_LIGHT) / QI if QI != 0 else 1.0
print(f"--- Normalization: d_i = {DI:.4f}, B0 = {B0:.4f}, dt = {DT:.4f}")


def load_brems_txt(timestep, p_type, bin_label):
    target_dir = os.path.join(BREMS_DATA_DIR, p_type, bin_label)
    filename = f'intensity_{p_type}_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    try:
        data = np.loadtxt(filepath)
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                data = data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_field_data_for_lines(timestep):
    bx_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    by_path = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    try:
        bx = np.loadtxt(bx_path, delimiter=',')
        by = np.loadtxt(by_path, delimiter=',')
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
# プロット関数 (変更: カラーバーを描画せず、cfオブジェクトを返す)
# =======================================================
def plot_brems_panel_subplot(ax, X, Y, Intensity, Bx, By, subtitle, omega_t_str, show_ylabel=True):
    cmap = plt.cm.inferno
    levels = np.linspace(FIXED_VMIN, FIXED_VMAX, 100)
    
    # コンタープロット（カラーバーはまだ描かない）
    cf = ax.contourf(X, Y, Intensity, levels=levels, cmap=cmap)
    
    # 磁力線
    try:
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
        ax.contour(X, Y, Psi_local, levels=25, colors='gray', linewidths=0.5, alpha=0.7)
    except Exception:
        pass

    # 時刻表示 (右端のプロットだけに表示するなど調整可能だが、今は全てに表示)
    ax.text(0.96, 0.96, omega_t_str, 
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='black',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8))

    ax.set_xlabel('$x/d_i$')
    # 左端のプロットのみY軸ラベルを表示
    if show_ylabel:
        ax.set_ylabel('$y/d_i$')
    
    ax.set_title(subtitle, fontsize=14)
    ax.tick_params(direction='in', top=True, right=True)
    
    return cf # カラーバー作成のためにcfを返す

# =======================================================
# メイン処理 (変更: 3パネルを1つの図にまとめる)
# =======================================================
def process_timestep_combined(timestep_str, output_base_dir):
    ts_int = int(timestep_str)
    omega_t_val = FGI * float(ts_int) * DT
    omega_t_str = fr"$\Omega_{{ci}}t = {omega_t_val:.2f}$"
    
    print(f"Processing TS: {timestep_str} ({omega_t_str})")

    # 1. 共通データの準備
    Bx, By = load_field_data_for_lines(timestep_str)
    X, Y = create_coordinates()

    # 2. エネルギービンごとのループ
    for bin_label in ENERGY_BINS:
        # --- データの読み込みと計算 ---
        data_thermal = load_brems_txt(timestep_str, 'Thermal', bin_label)
        data_nonthermal = load_brems_txt(timestep_str, 'NonThermal', bin_label)
        data_total = data_thermal + data_nonthermal
        
        # プロットするデータとタイトルのリスト
        plot_list = [
            ('Thermal', data_thermal),
            ('NonThermal', data_nonthermal),
            ('Total', data_total)
        ]

        # --- 図の作成 (1行3列) ---
        # sharey=TrueでY軸を共有し、ラベルの重複を防ぐ
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
        
        # 全体のタイトル
        fig.suptitle(f"Bremsstrahlung Intensity [{bin_label}] at {omega_t_str}", fontsize=16, y=0.98)
        
        cf_for_cbar = None # カラーバー用

        # 3つのパネルを描画するループ
        for i, (p_type, data) in enumerate(plot_list):
            ax = axes[i]
            # 左端(i==0)だけYラベルを表示
            cf = plot_brems_panel_subplot(ax, X, Y, data, Bx, By, p_type, omega_t_str, show_ylabel=(i==0))
            cf_for_cbar = cf # 最後のcfを保持しておく（Vmin/Vmaxは共通なのでどれでも良い）

        # --- 共通カラーバーの追加 ---
        # 図の右側にスペースを空けてカラーバーを配置
        fig.subplots_adjust(right=0.88, wspace=0.1)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
        cbar = fig.colorbar(cf_for_cbar, cax=cbar_ax, format='%.0e')
        cbar.set_label(r'Intensity (Arbitrary Units)', fontsize=12)
        cbar.set_ticks(np.linspace(FIXED_VMIN, FIXED_VMAX, 6))

        # --- 保存 ---
        # 出力先: output_dir/bin_label/combined_....png
        save_dir = os.path.join(output_base_dir, bin_label)
        os.makedirs(save_dir, exist_ok=True)
        
        out_name = f"combined_brems_{bin_label}_{timestep_str}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=150, bbox_inches='tight')
        plt.close(fig)
            
    print(f"  -> Saved combined plots for {timestep_str}")

if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'

    if len(sys.argv) < 4:
        print("Usage: python plot_brems_combined.py [start] [end] [step]")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = int(sys.argv[3])
    
    print(f"--- Visualizing Combined Bremsstrahlung Maps ---")
    print(f"Data Source: {BREMS_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    for t in range(start, end + step, step):
        ts_str = f"{t:06d}"
        process_timestep_combined(ts_str, OUTPUT_DIR)
        
    print("\nDone.")