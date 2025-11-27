import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
from scipy.ndimage import gaussian_filter

# =======================================================
# CONFIGURATION
# =======================================================
# 1. パラメータファイルのパス (正規化に必須)
#    init_param.dat の場所を指定してください
PARAM_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dat', 'init_param.dat') 
# ※ 見つからない場合のデフォルト値 (参照コードより)
DEFAULT_DI = 100.0 

# 2. ディレクトリ設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GCA_INPUT_DIR = os.path.join(SCRIPT_DIR, 'particle_gca_results') 
MOMENT_INPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'particle_gca_phys_axes_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 表示設定
FIXED_LIMIT = 5.0      # カラーバーのレンジ (+/-)
LIN_THRESH = 0.05      # 対数表示の閾値
SMOOTH_SIGMA = 1.0     # スムージング係数

# =======================================================
# HELPER: PARAMETER LOADING (正規化係数の取得)
# =======================================================
def get_normalization_scale(param_path):
    """
    init_param.dat からパラメータを読み込み、正規化長さ d_i を計算する
    """
    C_LIGHT = None
    FPI = None
    
    print(f"Reading parameters from: {param_path}")
    try:
        with open(param_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # 参照コードの読み込みロジック
                if line.startswith('dx, dt, c'):
                    # 例: dx, dt, c = ... ... ... 1.00 0.01 50.0
                    # parts[6] が c の位置にあると仮定
                    try: C_LIGHT = float(parts[6])
                    except: pass
                elif line.startswith('Fpe, Fge, Fpi'):
                    # 例: Fpe, Fge, Fpi ... = ... ... 0.5
                    # parts[7] が Fpi の位置にあると仮定
                    try: FPI = float(parts[7])
                    except: pass
    except FileNotFoundError:
        print("  Warning: Parameter file not found.")

    if C_LIGHT is not None and FPI is not None and FPI != 0:
        di = C_LIGHT / FPI
        print(f"  -> Calculated d_i (c/omega_pi) = {di:.4f}")
        return di
    else:
        print(f"  -> Warning: Could not calculate d_i. Using default: {DEFAULT_DI}")
        return DEFAULT_DI

# 正規化係数の計算 (グローバル)
DI = get_normalization_scale(PARAM_FILE_PATH)
DELX = 1.0 # 生データのグリッド間隔
DX_PHYS = DELX / DI # 物理単位でのグリッドサイズ (d_i単位)

# =======================================================
# DATA LOADING
# =======================================================
def load_data_generic(filepath):
    try:
        # カンマ区切りかスペース区切りか自動判定トライ
        try:
            return np.loadtxt(filepath, delimiter=',')
        except ValueError:
            return np.loadtxt(filepath)
    except:
        return None

def load_gca_and_density(timestep):
    # GCA
    t1 = load_data_generic(os.path.join(GCA_INPUT_DIR, f'gca_term1_par_{timestep}.txt'))
    t2 = load_data_generic(os.path.join(GCA_INPUT_DIR, f'gca_term2_betatron_{timestep}.txt'))
    t3 = load_data_generic(os.path.join(GCA_INPUT_DIR, f'gca_term3_fermi_{timestep}.txt'))
    total = load_data_generic(os.path.join(GCA_INPUT_DIR, f'gca_total_{timestep}.txt'))
    Bx = load_data_generic(os.path.join(GCA_INPUT_DIR, f'field_Bx_{timestep}.txt'))
    By = load_data_generic(os.path.join(GCA_INPUT_DIR, f'field_By_{timestep}.txt'))
    
    # Density
    ne = load_data_generic(os.path.join(MOMENT_INPUT_DIR, f'data_{timestep}_electron_density_count.txt'))

    if any(x is None for x in [t1, t2, t3, total, Bx, By, ne]):
        return None
    
    # 形状合わせ (念のため)
    if t1.shape != ne.shape:
        # GCA側のサイズにneを合わせるなどの処理が必要ならここに記述
        # 今回は同じと仮定
        pass
        
    return t1, t2, t3, total, Bx, By, ne

# =======================================================
# PLOTTING
# =======================================================
def plot_normalized(timestep):
    print(f"\n--- Plotting Timestep: {timestep} ---")
    
    data = load_gca_and_density(timestep)
    if data is None:
        print("  Skipping due to missing data.")
        return

    t1_raw, t2_raw, t3_raw, total_raw, Bx, By, ne = data

    # 加熱項の計算 (密度重み付け)
    term_Par = t1_raw * ne
    term_Fermi = t3_raw * ne
    term_Betatron = t2_raw * ne
    term_Total = total_raw * ne

    # スムージング
    if SMOOTH_SIGMA > 0:
        term_Par = gaussian_filter(term_Par, sigma=SMOOTH_SIGMA)
        term_Fermi = gaussian_filter(term_Fermi, sigma=SMOOTH_SIGMA)
        term_Betatron = gaussian_filter(term_Betatron, sigma=SMOOTH_SIGMA)
        term_Total = gaussian_filter(term_Total, sigma=SMOOTH_SIGMA)

    # -------------------------------------------------------
    # ★座標軸の作成 (ここがリクエストの核心)★
    # -------------------------------------------------------
    ny, nx = term_Total.shape
    
    # X軸: 中心を0にする (-Lx/2 ~ +Lx/2)
    # 参照コード: x_phys = np.linspace(-nx * DX_PHYS / 2.0, nx * DX_PHYS / 2.0, nx)
    lx_phys = nx * DX_PHYS
    x_coords = np.linspace(-lx_phys / 2.0, lx_phys / 2.0, nx)
    
    # Y軸: 0からスタート (0 ~ Ly)
    ly_phys = ny * DX_PHYS
    y_coords = np.linspace(0.0, ly_phys, ny)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    # -------------------------------------------------------

    # プロット準備
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True) # figsizeを正方形に近くしておく
    axs = axes.flatten()

    titles = [
        r'(a) Total Power ($n_e d\epsilon/dt$)',
        r'(b) Parallel ($E_\parallel J_\parallel$)',
        r'(c) Fermi ($v_\parallel^2 \mathbf{u}_E \cdot \mathbf{\kappa}$)',
        r'(d) Betatron ($\mu \mathbf{u}_E \cdot \nabla B$)'
    ]
    plot_data = [term_Total, term_Par, term_Fermi, term_Betatron]

    # コンターのレベル設定 (固定)
    fixed_levels = np.linspace(-FIXED_LIMIT, FIXED_LIMIT, 200)
    
    for ax, d, title in zip(axs, plot_data, titles):
        ax.set_facecolor('#d3d3d3')
        
        # SymLogNorm
        norm = colors.SymLogNorm(linthresh=LIN_THRESH, linscale=1.0, 
                                 vmin=-FIXED_LIMIT, vmax=FIXED_LIMIT, base=10)
        
        # コンター描画
        cf = ax.contourf(X, Y, d, levels=fixed_levels, cmap='gist_ncar', norm=norm, extend='both')
        
        # 磁力線 (ストリームプロット)
        # ※ストリームプロットも正規化座標(X,Y)で描画する必要があります
        stride = 25
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride], 
                      Bx[::stride, ::stride], By[::stride, ::stride], 
                      color='black', linewidth=0.5, arrowstyle='-')
        
        # 軸設定
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$x/d_i$', fontsize=12) # 単位表記
        ax.set_ylabel(r'$y/d_i$', fontsize=12)
        
        # ★アスペクト比を等倍に固定★ (物理的な形状を保つ)
        ax.set_aspect('equal')
        
        # カラーバー
        cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        # cbar.set_label('Power Density', fontsize=8)

    # 全体タイトル
    fig.suptitle(f"GCA Power Density (Normalized Axes) - TS {timestep}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, f'gca_norm_{timestep}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_gca_normalized.py [start] [end] [step]")
        # テスト用
        # plot_normalized("015000")
    else:
        s, e, st = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        for ts in range(s, e+st, st):
            plot_normalized(f"{ts:06d}")