import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors # LogNorm用にインポート
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ★ ユーザー設定
# =======================================================
# ディレクトリ設定 (計算コードの出力先に合わせる)
INPUT_DATA_DIR = os.path.join(os.path.abspath('.'), 'electron_energy_density')
FIELD_DATA_DIR = os.path.join(os.path.abspath('.'), 'extracted_data')
PARAM_FILE_PATH = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'electron_energy_plots')

# エネルギービン (計算コードと同じ定義)
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

# ★ プロットのレンジ設定 (対数スケール用)
# エネルギー密度(keV)なので、値は大きくなります。データに合わせて調整してください。
VMIN_LOG = 1e0    # 最小値 (これより小さい値は無視)
VMAX_LOG = 1e4    # 最大値

# =======================================================
# ヘルパー関数
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
        print(f"Warning: Could not load params ({e}). Using defaults.")
        return 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0
        
    if None in [C_LIGHT, FPI, DT, FGI, VA0, MI, QI]:
        return 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI if FPI != 0 else 1.0
B0 = (FGI * MI * C_LIGHT) / QI if QI != 0 else 1.0

def load_energy_density_txt(timestep, p_type, bin_label):
    """
    計算コードの出力ファイル名 'energy_density_...' を読み込む
    """
    target_dir = os.path.join(INPUT_DATA_DIR, p_type, bin_label)
    # ★修正箇所: ファイル名のプレフィックスを energy_density に変更
    filename = f'energy_density_{p_type}_{bin_label}_{timestep}.txt'
    filepath = os.path.join(target_dir, filename)
    
    try:
        data = np.loadtxt(filepath)
        # サイズ補正
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            if data.shape[0] >= GLOBAL_NY_PHYS and data.shape[1] >= GLOBAL_NX_PHYS:
                data = data[:GLOBAL_NY_PHYS, :GLOBAL_NX_PHYS]
            else:
                return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception:
        # ファイルがない場合などはゼロ行列を返す
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
# プロット関数 (LogNorm対応)
# =======================================================
def plot_panel_subplot(ax, X, Y, Data, Bx, By, subtitle, omega_t_str, show_ylabel=True):
    # カラーマップ
    cmap = plt.cm.inferno
    
    # ★データの前処理: ログプロットのために0以下を極小値に置換
    # コピーを作成して元のデータを破壊しないようにする
    plot_data = Data.copy()
    plot_data[plot_data <= 0] = 1e-10 
    
    # ★LogNormを使用
    norm = colors.LogNorm(vmin=VMIN_LOG, vmax=VMAX_LOG)
    
    # コンタープロット
    # LogScaleの場合、levelsは自動生成に任せるか、np.logspaceで生成する
    cf = ax.pcolormesh(X, Y, plot_data, norm=norm, cmap=cmap, shading='auto')
    
    # 磁力線 (エネルギー密度がメインなので、磁力線は薄く描画)
    try:
        if np.any(By):
            Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
            ax.contour(X, Y, Psi_local, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    except Exception:
        pass

    # 時刻表示
    ax.text(0.96, 0.96, omega_t_str, 
            transform=ax.transAxes, fontsize=10, fontweight='bold', color='white',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.5))

    ax.set_xlabel('$x/d_i$')
    if show_ylabel:
        ax.set_ylabel('$y/d_i$')
    
    ax.set_title(subtitle, fontsize=14)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_aspect('equal')
    
    return cf

# =======================================================
# メイン処理
# =======================================================
def process_timestep_combined(timestep_str, output_base_dir):
    ts_int = int(timestep_str)
    omega_t_val = FGI * float(ts_int) * DT
    omega_t_str = fr"$\Omega_{{ci}}t = {omega_t_val:.2f}$"
    
    print(f"Processing TS: {timestep_str} ({omega_t_str})")

    Bx, By = load_field_data_for_lines(timestep_str)
    X, Y = create_coordinates()

    for bin_label in ENERGY_BINS:
        # データの読み込み (修正した関数を使用)
        data_thermal = load_energy_density_txt(timestep_str, 'Thermal', bin_label)
        data_nonthermal = load_energy_density_txt(timestep_str, 'NonThermal', bin_label)
        data_total = data_thermal + data_nonthermal
        
        # データが全て0の場合はスキップしない（真っ暗な図を作る）が、警告は出さない
        
        plot_list = [
            ('Thermal', data_thermal),
            ('NonThermal', data_nonthermal),
            ('Total', data_total)
        ]

        # 図の作成 (1行3列)
        fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True, sharex=True)
        fig.suptitle(f"Energy Density Map: {bin_label}", fontsize=16, y=0.95)
        
        cf_for_cbar = None

        for i, (p_type, data) in enumerate(plot_list):
            ax = axes[i]
            cf = plot_panel_subplot(ax, X, Y, data, Bx, By, p_type, omega_t_str, show_ylabel=(i==0))
            cf_for_cbar = cf

        # 共通カラーバー
        fig.subplots_adjust(right=0.88, wspace=0.05)
        cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5]) # [left, bottom, width, height]
        
        if cf_for_cbar:
            cbar = fig.colorbar(cf_for_cbar, cax=cbar_ax)
            cbar.set_label(r'Energy Density (keV / cell)', fontsize=12)
            # LogScaleのラベルを見やすくする
            cbar.minorticks_on()

        # 保存
        save_dir = os.path.join(output_base_dir, bin_label)
        os.makedirs(save_dir, exist_ok=True)
        
        out_name = f"energy_map_{bin_label}_{timestep_str}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=150, bbox_inches='tight')
        plt.close(fig)
            
    print(f"  -> Saved plots for {timestep_str}")

if __name__ == "__main__":
    # LaTeXフォント設定 (環境になければコメントアウトしてください)
    try:
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
    except:
        pass

    if len(sys.argv) < 4:
        print("Usage: python plot_energy_density.py [start] [end] [step]")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = int(sys.argv[3])
    
    print(f"--- Plotting Energy Density Maps (Log Scale) ---")
    print(f"Input: {INPUT_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    for t in range(start, end + step, step):
        ts_str = f"{t:06d}"
        process_timestep_combined(ts_str, OUTPUT_DIR)
        
    print("\nDone.")