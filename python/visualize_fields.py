import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ヘルパー関数 (load_simulation_parameters, etc.)
# =======================================================

def load_simulation_parameters(param_filepath):
    """
    init_param.dat (または同等のログファイル) を読み込み、
    C_LIGHT, FPI (omega_pi), DT (dt), FGI (Omega_ci),
    VA0 (Alfvén velocity), MI (イオン質量), QI (イオン電荷) を抽出する。
    """
    C_LIGHT = None
    FPI = None
    DT = None  # (dt)
    FGI = None # (Omega_ci)
    VA0 = None # (Alfvén velocity)
    MI = None  # (イオン質量 r(1))
    QI = None  # (イオン電荷 q(1))
    
    print(f"パラメータファイルを読み込み中: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                
                if stripped_line.startswith('dx, dt, c'):
                    try:
                        parts = stripped_line.split()
                        DT = float(parts[5])
                        C_LIGHT = float(parts[6])
                        print(f"  -> 'dt' の値を検出: {DT}")
                        print(f"  -> 'c' の値を検出: {C_LIGHT}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'dx, dt, c' の値の解析に失敗。行: {line}")

                elif stripped_line.startswith('Mi, Me'):
                    try:
                        parts = stripped_line.split()
                        MI = float(parts[3])
                        print(f"  -> 'Mi' (MI) の値を検出: {MI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Mi, Me' の値の解析に失敗。行: {line}")

                elif stripped_line.startswith('Qi, Qe'):
                    try:
                        parts = stripped_line.split()
                        QI = float(parts[3])
                        print(f"  -> 'Qi' (QI) の値を検出: {QI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Qi, Qe' の値の解析に失敗。行: {line}")
                        
                elif stripped_line.startswith('Fpe, Fge, Fpi Fgi'):
                    try:
                        parts = stripped_line.split()
                        FPI = float(parts[7])
                        FGI = float(parts[8])
                        print(f"  -> 'Fpi' の値を検出: {FPI}")
                        print(f"  -> 'Fgi' (FGI) の値を検出: {FGI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Fpe, Fge, Fpi Fgi' の値の解析に失敗。行: {line}")

                elif stripped_line.startswith('Va, Vi, Ve'):
                    try:
                        parts = stripped_line.split()
                        VA0 = float(parts[7])
                        print(f"  -> 'Va' (VA0) の値を検出: {VA0}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Va, Vi, Ve' の値の解析に失敗。行: {line}")

    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        print("     規格化パラメータの読み込みに失敗しました。スクリプトを終了します。")
        sys.exit(1)
        
    if C_LIGHT is None or FPI is None or DT is None or FGI is None or VA0 is None or MI is None or QI is None:
        print("★★ エラー: ファイルから必要なパラメータ ('c', 'Fpi', 'dt', 'Fgi', 'Va', 'Mi', 'Qi') のいずれかを抽出できませんでした。")
        print("     ファイルの内容を確認してください。スクリプトを終了します。")
        sys.exit(1)
        
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# =======================================================
# 設定と定数
# =======================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'allcombined'), exist_ok=True) # 統合パネル用

PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI
B0 = (FGI * MI * C_LIGHT) / QI

print(f"--- 規格化スケール (空間): d_i = {DI:.4f}")
print(f"--- 規格化スケール (磁場): B0 = {B0:.4f}")
print(f"--- 規格化スケール (速度): VA0 = {VA0:.4f}")
print(f"--- 時間スケール: dt = {DT}, Fgi (Omega_ci) = {FGI}")

# ★★★ 設定変更: Proxy -> Te, Ti に変更 ★★★
GLOBAL_PLOT_RANGES = {
    'Bx': (-1.0, 1.0),
    'By': (-1.5, 1.5),
    'Bz': (-0.5, 0.5),
    'Ex': (-0.3, 0.3),
    'Ey': (-0.05, 0.05),
    'Ez': (-0.1, 0.1),
    'Jx': (-0.5, 0.5),
    'Jy': (-0.75, 0.75),
    'Jz': (-1.75, 1.75),
    'ne': (0, 500),
    'ni': (0, 500),
    'Psi': (-5000, 5000), 
    'Te': (0, 5),   # ★ Te Proxyより少しレンジを調整 (必要に応じて変更してください)
    'Ti': (0, 1),   # ★ Ti
    'Vxe': (-1.0, 1.0),
    'Vxi': (-0.5, 0.5),
    'Vye': (-2.5, 2.5),
    'Vyi': (-1.0, 1.0),
    'Vze': (-4.0, 4.0),
    'Vzi': (-2.0, 2.0),
}

GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# =======================================================
# ヘルパー関数 (データ読み込み・計算)
# =======================================================
def load_2d_field_data(timestep, component):
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(FIELD_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS}) と異なります。ゼロ配列を返します。")
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data 
    except Exception as e:
        print(f"警告: {filepath} の読み込みに失敗しました ({e})。ゼロ配列を返します。")
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_2d_moment_data(timestep, species, component):
    filename = f'data_{timestep}_{species}_{component}.txt'
    filepath = os.path.join(MOMENT_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS}) と異なります。ゼロ配列を返します。")
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception as e:
        print(f"警告: {filepath} の読み込みに失敗しました ({e})。ゼロ配列を返します。")
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def calculate_current_density(Bx, By, Ex, Ey, Ez, J_list, B0):
    n_e_count = J_list['density_count_e']
    n_i_count = J_list['density_count_i']
    Vx_e = J_list['Vx_e']
    Vx_i = J_list['Vx_i']
    Vy_e = J_list['Vy_e']
    Vy_i = J_list['Vy_i']
    Vz_e = J_list['Vz_e']
    Vz_i = J_list['Vz_i']

    N0 = (n_e_count + n_i_count) / 2.0
    N0_filtered = N0[N0 > 1e-1]
    if len(N0_filtered) == 0:
        avg_N0 = 1.0
    else:
        avg_N0 = np.mean(N0_filtered)
        
    if avg_N0 < 1: avg_N0 = 1 
    n_proxy = N0 / avg_N0
    
    J_x = n_proxy * (Vx_i - Vx_e)
    J_y = n_proxy * (Vy_i - Vy_e)
    J_z = n_proxy * (Vz_i - Vz_e)
    Ez_non_ideal = Ez 
    return J_x, J_y, J_z, Ez_non_ideal

def calculate_magnetic_flux(Bx, By, DELX):
    Psi_approx = cumtrapz(By, dx=DELX, axis=1, initial=0)
    return Psi_approx

def create_coordinates(NX, NY):
    x_phys = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, NX)
    y_phys = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, NY)
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    return np.meshgrid(x_norm, y_norm)

def get_plot_range(Z, tag=None):
    """
    ゼロ配列の場合にプロット範囲を調整。
    """
    try:
        if Z.size == 0 or np.all(np.isnan(Z)):
             raise ValueError("All NaN or empty array")
             
        max_abs = np.nanmax(np.abs(Z))

    except (ValueError, RuntimeWarning):
        max_abs = 0.0

    if np.isclose(max_abs, 0.0) or not np.isfinite(max_abs):
        return -1e-6, 1e-6
    else:
        return -max_abs, max_abs

# =======================================================
# プロット関数
# =======================================================

def plot_single_panel(ax, X, Y, Z, Bx, By, title, label, omega_t_str, cmap='RdBu_r', vmin=None, vmax=None, tag_key=None):
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
        
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap) 
    
    formatter_str = '%.2f'
    if (vmax > 0 and np.abs(vmax) < 0.01) or np.abs(vmax) > 1000:
         formatter_str = '%.1e'
         
    # --- 目盛りの手動設定 ---
    plot_ticks = None
    # ★ Te, Ti もここに追加
    if tag_key in ['ne', 'ni', 'Te', 'Ti', 'Psi']: 
        num_ticks = 6 
    else: # 対称なもの
        num_ticks = 7 
        
    if not np.isclose(vmin, vmax):
        plot_ticks = np.linspace(vmin, vmax, num_ticks)
    # ---

    cbar = plt.colorbar(cf, ax=ax, format=formatter_str, ticks=plot_ticks)
    
    cbar.set_label(label)
    
    # stride_x = max(1, Bx.shape[1] // 30) 
    # stride_y = max(1, Bx.shape[0] // 30) 
    # ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
    #               Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
    #               color='gray', linewidth=0.5, density=1.0, 
    #               arrowstyle='-', minlength=0.1, zorder=1)
    try:
        # dxは相対的な形状には影響しないので1.0で計算
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0) 
        
        # 等高線を描画 (これが磁力線になります)
        # levels: 線の本数 (20〜30くらいが適切)
        # colors: 線の色 (薄いグレーや黒)
        # linewidths: 線の太さ
        ax.contour(X, Y, Psi_local, levels=25, colors='gray', linewidths=0.5, alpha=0.8)
        
    except Exception as e:
        print(f"磁力線描画エラー: {e}")
    
    ax.text(0.98, 0.98, omega_t_str, 
            transform=ax.transAxes, 
            fontsize=12, 
            fontweight='bold',
            color='black',
            ha='right', 
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7))
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)


def plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap, vmin, vmax, tag_key, stream_color='gray', stream_density=1.0):
    
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    levels = np.linspace(vmin, vmax, 100)
    
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

    formatter_str = '%.2f'
    if (vmax > 0 and np.abs(vmax) < 0.01) or np.abs(vmax) > 1000:
         formatter_str = '%.1e'
         
    plot_ticks = None
    num_ticks = 5 
    
    if not np.isclose(vmin, vmax):
        plot_ticks = np.linspace(vmin, vmax, num_ticks)

    cbar = plt.colorbar(cf, ax=ax, format=formatter_str, 
                        shrink=0.9, aspect=30, pad=0.02,
                        ticks=plot_ticks)
    
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # stride_x = max(1, Bx.shape[1] // 15)
    # stride_y = max(1, Bx.shape[0] // 15)
    
    # ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
    #               Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
    #               color=stream_color, linewidth=0.5, density=stream_density, 
    #               arrowstyle='-', minlength=0.1, zorder=1)
    try:
        Psi_local = cumtrapz(By, dx=1.0, axis=1, initial=0)
        
        # levelsの数を少し減らしてスッキリさせる (combined用)
        ax.contour(X, Y, Psi_local, levels=15, colors=stream_color, linewidths=0.5, alpha=0.8)
        
    except Exception as e:
        print(f"磁力線描画エラー: {e}")
                  
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$x/d_i$', fontsize=8)
    ax.set_ylabel('$y/d_i$', fontsize=8)
    ax.tick_params(direction='in', top=True, right=True, labelsize=7)
    
    return cf


# =======================================================
# メイン実行関数 (単一ステップ処理用)
# =======================================================

def process_timestep(timestep):
    """指定された単一のタイムステップのデータを処理・可視化する"""
    
    print(f"\n=============================================")
    print(f"--- ターゲットタイムステップ: {timestep} の処理開始 ---")
    print(f"=============================================")

    try:
        it0 = int(timestep)
        omega_t_value = FGI * float(it0) * DT 
        omega_t_str = fr"$\Omega_{{ci}}t = {omega_t_value:.2f}$"
        print(f"計算: {omega_t_str} (it0={it0}, Fgi={FGI}, dt={DT})")
    except Exception as e:
        print(f"警告: Omega_ci * t の計算に失敗しました: {e}")
        omega_t_str = r"$\Omega_{ci}t = N/A$"

    # --- 1. 必要なデータの読み込み (Rawデータ) ---
    print("電磁場データを読み込み中...")
    Bx_raw = load_2d_field_data(timestep, 'Bx')
    By_raw = load_2d_field_data(timestep, 'By')
    Bz_raw = load_2d_field_data(timestep, 'Bz')
    Ex_raw = load_2d_field_data(timestep, 'Ex')
    Ey_raw = load_2d_field_data(timestep, 'Ey')
    Ez_raw = load_2d_field_data(timestep, 'Ez')
        
    print("粒子モーメントデータを読み込み中...")
    Vxe_raw = load_2d_moment_data(timestep, 'electron', 'Vx')
    Vye_raw = load_2d_moment_data(timestep, 'electron', 'Vy')
    Vze_raw = load_2d_moment_data(timestep, 'electron', 'Vz')
    Vxi_raw = load_2d_moment_data(timestep, 'ion', 'Vx')
    Vyi_raw = load_2d_moment_data(timestep, 'ion', 'Vy')
    Vzi_raw = load_2d_moment_data(timestep, 'ion', 'Vz')
    
    # ★★★ 温度データの読み込み (追加) ★★★
    # 前回のスクリプトで保存した 'T' ファイルを読み込む
    Te_raw = load_2d_moment_data(timestep, 'electron', 'T')
    Ti_raw = load_2d_moment_data(timestep, 'ion', 'T')
    
    ne_count = load_2d_moment_data(timestep, 'electron', 'density_count')
    ni_count = load_2d_moment_data(timestep, 'ion', 'density_count')
    
    # --- 2. 規格化と派生量の計算 ---
    print("データを B0 と VA0 で規格化中...")

    Bx = Bx_raw / B0
    By = By_raw / B0
    Bz = Bz_raw / B0
    
    Ex = Ex_raw / B0
    Ey = Ey_raw / B0
    Ez = Ez_raw / B0 

    Vxe = Vxe_raw / VA0
    Vye = Vye_raw / VA0
    Vze = Vze_raw / VA0
    Vxi = Vxi_raw / VA0
    Vyi = Vyi_raw / VA0
    Vzi = Vzi_raw / VA0
    
    ne = ne_count
    ni = ni_count

    print("派生量を計算中...")
    
    Psi = calculate_magnetic_flux(Bx_raw, By_raw, DELX)
    
    # ★★★ 温度の規格化 ★★★
    # 温度(速度の二乗)を VA0^2 で規格化する
    Te = Te_raw / (VA0**2)
    Ti = Ti_raw / (VA0**2)
    
    # Proxy計算部分は削除
    # Te_proxy = ... 
    # Ti_proxy = ...
    
    J_data = {'density_count_e': ne, 'density_count_i': ni,
              'Vx_e': Vxe_raw, 'Vx_i': Vxi_raw, 
              'Vy_e': Vye_raw, 'Vy_i': Vyi_raw,
              'Vz_e': Vze_raw, 'Vz_i': Vzi_raw}
              
    Jx, Jy, Jz, _ = calculate_current_density(Bx_raw, By_raw, Ex_raw, Ey_raw, Ez_raw, J_data, B0)
    
    Ez_non_ideal_ion_raw = Ez_raw + (Vxi_raw * By_raw - Vyi_raw * Bx_raw)
    Ez_non_ideal_ion = Ez_non_ideal_ion_raw / B0 

    Ez_non_ideal_electron_raw = Ez_raw + (Vxe_raw * By_raw - Vye_raw * Bx_raw)
    Ez_non_ideal_electron = Ez_non_ideal_electron_raw / B0
    
    
    # --- 3. 座標グリッドの作成 ---
    X, Y = create_coordinates(GLOBAL_NX_PHYS, GLOBAL_NY_PHYS)

    # --- 4. 可視化実行 ---
    
    # (a) 個別プロット用のリスト 
    # ★ Te_proxy -> Te, Ti_proxy -> Ti に変更
    plot_components = [
        ('Bx', Bx, r'Magnetic Field ($B_x/B_0$)', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, r'Magnetic Field ($B_y/B_0$)', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, r'Magnetic Field ($B_z/B_0$)', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Ex', Ex, r'Electric Field ($E_x/B_0$)', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, r'Electric Field ($E_y/B_0$)', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, r'Electric Field ($E_z/B_0$)', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', ne, 'Electron Density', r'$n_e$ (Counts)', plt.cm.viridis),
        ('Te', Te, 'Electron Temperature', r'$T_e / (m_e V_{A0}^2)$', plt.cm.plasma), # ★
        ('ni', ni, 'Ion Density', r'$n_i$ (Counts)', plt.cm.viridis),
        ('Ti', Ti, 'Ion Temperature', r'$T_i / (m_i V_{A0}^2)$', plt.cm.plasma),      # ★
        ('Psi', Psi, r'Magnetic Flux $\Psi$', r'$\Psi$', plt.cm.seismic),
        ('Jx', Jx, 'Current Density (Jx)', r'$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, 'Current Density (Jy)', r'$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, 'Current Density (Jz)', r'$J_z$', plt.cm.RdBu_r),
        ('Vxi', Vxi, r'Ion Velocity ($V_{ix}/V_{A0}$)', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        ('Vyi', Vyi, r'Ion Velocity ($V_{iy}/V_{A0}$)', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        ('Vzi', Vzi, r'Ion Velocity ($V_{iz}/V_{A0}$)', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        ('Vxe', Vxe, r'Electron Velocity ($V_{ex}/V_{A0}$)', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        ('Vye', Vye, r'Electron Velocity ($V_{ey}/V_{A0}$)', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        ('Vze', Vze, r'Electron Velocity ($V_{ez}/V_{A0}$)', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
        ('Ez_non_ideal_e', Ez_non_ideal_electron, r'Non-Ideal $E_z$ (Electron)', r'$(E_z + (\mathbf{V}_e \times \mathbf{B})_z)/B_0$', plt.cm.jet),
        ('Ez_non_ideal_i', Ez_non_ideal_ion, r'Non-Ideal $E_z$ (Ion)', r'$(E_z + (\mathbf{V}_i \times \mathbf{B})_z)/B_0$', plt.cm.jet),
    ]

    # --- プロット A: 各成分を個別のサブディレクトリに出力 ---
    print("個別のプロットを生成中 (サブディレクトリに保存)...")
    
    for tag_key, Z, title, label, cmap in plot_components:
        SUB_DIR = os.path.join(OUTPUT_DIR, tag_key.replace('/', '_'))
        os.makedirs(SUB_DIR, exist_ok=True)
        
        if tag_key in GLOBAL_PLOT_RANGES:
            vmin, vmax = GLOBAL_PLOT_RANGES[tag_key]
        else:
            vmin, vmax = get_plot_range(Z, tag=tag_key)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        plot_single_panel(ax, X, Y, Z, Bx, By,
                          f"Timestep {timestep}: {title}", label, 
                          omega_t_str, cmap=cmap, 
                          vmin=vmin, vmax=vmax,
                          tag_key=tag_key)
        
        fig.tight_layout()
        output_filename = os.path.join(SUB_DIR, f'plot_{timestep}_{tag_key}.png')
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
    
    print(f"-> 個別プロット {len(plot_components)} 点を {OUTPUT_DIR} 配下に保存しました。")

    # --- プロット B: 全ての重要な要素を1枚の統合パネルに出力 ---
    print("\n統合パネルを生成中 (ルートディレクトリに保存)...")
    
    fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
    ax_list = axes.flatten()
    fig.suptitle(f"Timestep: {timestep}  ({omega_t_str})", fontsize=16, fontweight='bold')
    
    # (b) 統合パネル用のリスト
    # ★ Te_proxy -> Te, Ti_proxy -> Ti に変更
    combined_plots = [
        ('Bx', Bx, r'(a) $B_x/B_0$', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, r'(b) $B_y/B_0$', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, r'(c) $B_z/B_0$', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Psi', Psi, r'(d) $\Psi$', r'$\Psi$', plt.cm.seismic),
        ('Ex', Ex, r'(e) $E_x/B_0$', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, r'(f) $E_y/B_0$', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, r'(g) $E_z/B_0$', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', ne, '(h) $n_e$', r'$n_e$ Count', plt.cm.viridis),
        ('Jx', Jx, '(i) $J_x$', r'$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, '(j) $J_y$', r'$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, '(k) $J_z$', r'$J_z$', plt.cm.RdBu_r),
        ('ni', ni, '(l) $n_i$', r'$n_i$ Count', plt.cm.viridis),
        ('Vxi', Vxi, r'(m) $V_{ix}/V_{A0}$', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        ('Vyi', Vyi, r'(n) $V_{iy}/V_{A0}$)', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        ('Vzi', Vzi, r'(o) $V_{iz}/V_{A0}$)', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        ('Ti', Ti, '(p) $T_i$', r'$T_i$', plt.cm.plasma),          # ★
        ('Vxe', Vxe, r'(q) $V_{ex}/V_{A0}$)', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        ('Vye', Vye, r'(r) $V_{ey}/V_{A0}$)', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        ('Vze', Vze, r'(s) $V_{ez}/V_{A0}$)', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
        ('Te', Te, '(t) $T_e$', r'$T_e$', plt.cm.plasma),          # ★
    ]
    
    for i, (tag_key, Z, title, label, cmap) in enumerate(combined_plots):
        if i < len(ax_list):
            ax = ax_list[i]
            
            if tag_key in GLOBAL_PLOT_RANGES:
                vmin, vmax = GLOBAL_PLOT_RANGES[tag_key]
            else:
                vmin, vmax = get_plot_range(Z, tag=tag_key)

            stream_color = 'white' if cmap == plt.cm.seismic else 'gray'
            
            cf = plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap=cmap, vmin=vmin, vmax=vmax, tag_key=tag_key, stream_color=stream_color)
        else:
            break

    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=0.5)
    
    output_filename_combined = os.path.join(OUTPUT_DIR, 'allcombined', f'plot_combined_{timestep}.png')
    
    plt.savefig(output_filename_combined, dpi=300)
    plt.close(fig)
    print(f"-> 全てを含む統合パネル (5x4) を {output_filename_combined} に保存しました。")
    print(f"--- タイムステップ: {timestep} の処理完了 ---")
    
# =======================================================
# スクリプト実行ブロック (ループ処理)
# =======================================================
if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    if len(sys.argv) != 4:
        print("使用方法: python visual_fields.py [start_timestep] [end_timestep] [interval]")
        print("       (タイムステップは \"000500\" ではなく 500 のように数値で指定してください)")
        print("例: python visual_fields.py 0 14000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step = int(sys.argv[2])
        interval = int(sys.argv[3])
    except ValueError:
        print("エラー: タイムステップと間隔は整数で指定してください。")
        sys.exit(1)

    if interval <= 0:
        print("エラー: 間隔 (interval) は正の整数である必要があります。")
        sys.exit(1)
        
    if start_step > end_step:
        print("エラー: start_timestep は end_timestep 以下である必要があります。")
        sys.exit(1)

    print(f"--- ループ処理を開始します (Start: {start_step}, End: {end_step}, Interval: {interval}) ---")

    current_step = start_step
    while current_step <= end_step:
        
        timestep_str = f"{current_step:06d}"
        
        try:
            process_timestep(timestep_str)
            
        except Exception as e:
            print(f"★★ 重大なエラー: タイムステップ {timestep_str} の処理中に例外が発生しました: {e}")
            import traceback
            traceback.print_exc()
            print("     処理を続行します...")
            
        current_step += interval

    print(f"\n--- 全てのタイムステップの処理が完了しました ---")