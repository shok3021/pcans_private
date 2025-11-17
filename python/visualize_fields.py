import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ヘルパー関数 (load_simulation_parameters, load_2d_field_data, etc.)
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
                
                # 'dt' と 'c' の値を抽出
                if stripped_line.startswith('dx, dt, c'):
                    try:
                        parts = stripped_line.split()
                        DT = float(parts[5])      # 6番目の要素 (dt)
                        C_LIGHT = float(parts[6]) # 7番目の要素 (c)
                        print(f"  -> 'dt' の値を検出: {DT}")
                        print(f"  -> 'c' の値を検出: {C_LIGHT}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'dx, dt, c' の値の解析に失敗。行: {line}")

                # ★ 'Mi' (イオン質量) の値を抽出 ★
                elif stripped_line.startswith('Mi, Me'):
                    try:
                        parts = stripped_line.split()
                        MI = float(parts[3]) # 4番目の要素 (Mi)
                        print(f"  -> 'Mi' (MI) の値を検出: {MI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Mi, Me' の値の解析に失敗。行: {line}")

                # ★ 'Qi' (イオン電荷) の値を抽出 ★
                elif stripped_line.startswith('Qi, Qe'):
                    try:
                        parts = stripped_line.split()
                        QI = float(parts[3]) # 4番目の要素 (Qi)
                        print(f"  -> 'Qi' (QI) の値を検出: {QI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Qi, Qe' の値の解析に失敗。行: {line}")
                        
                # 'Fpi' と 'Fgi' の値を抽出
                elif stripped_line.startswith('Fpe, Fge, Fpi Fgi'):
                    try:
                        parts = stripped_line.split()
                        FPI = float(parts[7]) # 8番目の要素 (Fpi)
                        FGI = float(parts[8]) # 9番目の要素 (Fgi)
                        print(f"  -> 'Fpi' の値を検出: {FPI}")
                        print(f"  -> 'Fgi' (FGI) の値を検出: {FGI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Fpe, Fge, Fpi Fgi' の値の解析に失敗。行: {line}")

                # 'Va' (VA0) の値を抽出
                elif stripped_line.startswith('Va, Vi, Ve'):
                    try:
                        parts = stripped_line.split()
                        # ★★★ インデックスを 4 から 6 に修正 (これは元のコードの修正を維持) ★★★
                        VA0 = float(parts[6]) # 7番目の要素 (Va)
                        print(f"  -> 'Va' (VA0) の値を検出: {VA0}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Va, Vi, Ve' の値の解析に失敗。行: {line}")

    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        print("     規格化パラメータの読み込みに失敗しました。スクリプトを終了します。")
        sys.exit(1)
        
    # ★ エラーチェックに MI と QI を追加 ★
    if C_LIGHT is None or FPI is None or DT is None or FGI is None or VA0 is None or MI is None or QI is None:
        print("★★ エラー: ファイルから必要なパラメータ ('c', 'Fpi', 'dt', 'Fgi', 'Va', 'Mi', 'Qi') のいずれかを抽出できませんでした。")
        print("     ファイルの内容を確認してください。スクリプトを終了します。")
        sys.exit(1)
        
    # ★ return に MI と QI を追加 ★
    return C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# =======================================================
# 設定と定数
# =======================================================
# ( __file__ が未定義の場合のエラーを防ぐためのフォールバック)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.') # (Jupyterなどでの実行用)

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- init_param.dat のパスを指定 ---
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# --- パラメータの読み込みと di の計算 ---
C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI # イオンスキンデプス (di = c / omega_pi)

# --- ★ 規格化定数の定義 (B0を計算) ★ ---
# B0 = 1.0  # <- ハードコードされた値を削除
# Fortranコード (init__set_param) の定義に基づいて B0 を計算
# b0  = fgi*r(1)*c/q(1)
B0 = (FGI * MI * C_LIGHT) / QI
# VA0 は init_param.dat から読み込んだ値を使用

print(f"--- 規格化スケール (空間): d_i = {DI:.4f}")
print(f"--- 規格化スケール (磁場): B0 = {B0:.4f}")
print(f"--- 規格化スケール (速度): VA0 = {VA0:.4f}")
print(f"--- 時間スケール: dt = {DT}, Fgi (Omega_ci) = {FGI}")

# --- Fortran const モジュールからの値 ---
GLOBAL_NX_PHYS = 320 # X方向セル数
GLOBAL_NY_PHYS = 639 # Y方向セル数
DELX = 1.0 # セル幅

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
        # ★ ループ処理のため、エラーがあっても停止させず、警告を出す
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
        # ★ ループ処理のため、エラーがあっても停止させず、警告を出す
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
    # ゼロ割を避ける
    N0_filtered = N0[N0 > 1e-1]
    if len(N0_filtered) == 0:
        avg_N0 = 1.0 # データが全くない場合
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

def get_plot_range(Z):
    """ゼロ配列の場合にプロット範囲を調整"""
    try:
        max_abs = np.nanmax(np.abs(Z))
    except ValueError: # すべてNaNの場合
        max_abs = 0.0
        
    if np.isclose(max_abs, 0.0) or not np.isfinite(max_abs):
        return -1e-6, 1e-6
    return -max_abs, max_abs

# =======================================================
# プロット関数 (変更なし)
# =======================================================

def plot_single_panel(ax, X, Y, Z, Bx, By, title, label, omega_t_str, cmap='RdBu_r', vmin=None, vmax=None):
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(cf, ax=ax, format='%.2f')
    cbar.set_label(label)
    stride_x = max(1, Bx.shape[1] // 30) 
    stride_y = max(1, Bx.shape[0] // 30) 
    ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                  Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                  color='gray', linewidth=0.5, density=1.0, 
                  arrowstyle='-', minlength=0.1, zorder=1)
    
    # ★★★ テキスト追加 (図の右上に配置) ★★★
    ax.text(0.98, 0.98, omega_t_str, 
            transform=ax.transAxes, 
            fontsize=12, 
            fontweight='bold',
            color='black',
            ha='right', 
            va='top',
            # 白い背景ボックスを追加
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7))
    # ★★★★★★★★★★★★★★★★★★★★★★★
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)

def plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap='RdBu_r', vmin=None, vmax=None, stream_color='gray', stream_density=1.0):
    if vmin is None: vmin, vmax = get_plot_range(Z) # ゼロ対応
    if vmax is None: vmin, vmax = get_plot_range(Z) # ゼロ対応
        
    levels = np.linspace(vmin, vmax, 100)
    
    # nanを特定の色 (例: 'lightgray') でマスクする
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')

    # --- ★ ユーザーの要望によりカラーバーを追加 ★ ---
    # カラーバーのフォーマットを調整
    formatter_str = '%.2f'
    if (vmax > 0 and np.abs(vmax) < 0.01) or np.abs(vmax) > 100:
         formatter_str = '%.1e'
         
    # 統合パネル用にカラーバーを小さく設定 (shrink, aspect, pad)
    cbar = plt.colorbar(cf, ax=ax, format=formatter_str, 
                        shrink=0.9, aspect=30, pad=0.02)
    
    # ラベルと目盛りのフォントサイズも小さくする
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    # --- ★★★★★★★★★★★★★★★★★★★★★★★ ---

    stride_x = max(1, Bx.shape[1] // 15)
    stride_y = max(1, Bx.shape[0] // 15)
    
    ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                  Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                  color=stream_color, linewidth=0.5, density=stream_density, 
                  arrowstyle='-', minlength=0.1, zorder=1)
                  
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$x/d_i$', fontsize=8)
    ax.set_ylabel('$y/d_i$', fontsize=8)
    ax.tick_params(direction='in', top=True, right=True, labelsize=7)
    
    return cf
# =======================================================
# ★★★ メイン実行関数 (単一ステップ処理用) ★★★
# =======================================================

def process_timestep(timestep):
    """指定された単一のタイムステップのデータを処理・可視化する"""
    
    print(f"\n=============================================")
    print(f"--- ターゲットタイムステップ: {timestep} の処理開始 ---")
    print(f"=============================================")

    # --- Omega_ci * t の計算 ---
    try:
        it0 = int(timestep) # 文字列を整数に (例: 500)
        # グローバル変数の DT と FGI を使用
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
    
    ne_count = load_2d_moment_data(timestep, 'electron', 'density_count')
    ni_count = load_2d_moment_data(timestep, 'ion', 'density_count')
    
    # --- 2. ★ 規格化と派生量の計算 ★ ---
    print("データを B0 と VA0 で規格化中...")

    # 磁場 (B / B0)
    Bx = Bx_raw / B0
    By = By_raw / B0
    Bz = Bz_raw / B0
    
    # 電場 (E / B0)
    Ex = Ex_raw / B0
    Ey = Ey_raw / B0
    Ez = Ez_raw / B0 # <- 規格化済みEz

    # 速度 (V / VA0)
    Vxe = Vxe_raw / VA0
    Vye = Vye_raw / VA0
    Vze = Vze_raw / VA0
    Vxi = Vxi_raw / VA0
    Vyi = Vyi_raw / VA0
    Vzi = Vzi_raw / VA0
    
    # 密度 (規格化なし)
    ne = ne_count
    ni = ni_count

    # 派生量 (規格化済みの値を使って計算)
    print("派生量を計算中...")
    
    Psi = calculate_magnetic_flux(Bx_raw, By_raw, DELX)
    Te_proxy = Vxe_raw**2 + Vye_raw**2 + Vze_raw**2
    Ti_proxy = Vxi_raw**2 + Vyi_raw**2 + Vzi_raw**2
    
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
    
    # (a) 個別プロット用のリスト (★ タイトルを規格化表記に変更 ★)
    plot_components = [
        ('Bx', Bx, r'Magnetic Field ($B_x/B_0$)', r'$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, r'Magnetic Field ($B_y/B_0$)', r'$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, r'Magnetic Field ($B_z/B_0$)', r'$B_z/B_0$', plt.cm.RdBu_r),
        ('Ex', Ex, r'Electric Field ($E_x/B_0$)', r'$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, r'Electric Field ($E_y/B_0$)', r'$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, r'Electric Field ($E_z/B_0$)', r'$E_z/B_0$', plt.cm.coolwarm),
        ('ne', ne, 'Electron Density', r'$n_e$ (Counts)', plt.cm.viridis),
        ('Te', Te_proxy, 'Electron Temperature (Proxy)', r'$T_e$ (Proxy)', plt.cm.plasma),
        ('ni', ni, 'Ion Density', r'$n_i$ (Counts)', plt.cm.viridis),
        ('Ti', Ti_proxy, 'Ion Temperature (Proxy)', r'$T_i$ (Proxy)', plt.cm.plasma),
        ('Psi', Psi, r'Magnetic Flux $\Psi$', r'$\Psi$', plt.cm.seismic), # <- r'' 追加
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
    
    for tag, Z, title, label, cmap in plot_components:
        SUB_DIR = os.path.join(OUTPUT_DIR, tag.replace('/', '_'))
        os.makedirs(SUB_DIR, exist_ok=True)
        vmin, vmax = get_plot_range(Z)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_panel(ax, X, Y, Z, Bx, By, # Bx, Byは規格化済み
                          f"Timestep {timestep}: {title}", label, 
                          omega_t_str, cmap=cmap, 
                          vmin=vmin, vmax=vmax)
        
        fig.tight_layout()
        output_filename = os.path.join(SUB_DIR, f'plot_{timestep}_{tag}.png')
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
    
    print(f"-> 個別プロット {len(plot_components)} 点を {OUTPUT_DIR} 配下に保存しました。")

    # --- プロット B: 全ての重要な要素を1枚の統合パネルに出力 ---
    print("\n統合パネルを生成中 (ルートディレクトリに保存)...")
    
    fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
    ax_list = axes.flatten()
    fig.suptitle(f"Timestep: {timestep}  ({omega_t_str})", fontsize=16, fontweight='bold')
    
    # (b) 統合パネル用のリスト (★ タイトルを規格化表記に変更 ★)
    combined_plots = [
        (Bx, r'(a) $B_x/B_0$', r'$B_x/B_0$', plt.cm.RdBu_r),
        (By, r'(b) $B_y/B_0$', r'$B_y/B_0$', plt.cm.RdBu_r),
        (Bz, r'(c) $B_z/B_0$', r'$B_z/B_0$', plt.cm.RdBu_r),
        (Psi, r'(d) $\Psi$', r'$\Psi$', plt.cm.seismic), # <- r'' 追加
        (Ex, r'(e) $E_x/B_0$', r'$E_x/B_0$', plt.cm.coolwarm),
        (Ey, r'(f) $E_y/B_0$', r'$E_y/B_0$', plt.cm.coolwarm),
        (Ez, r'(g) $E_z/B_0$', r'$E_z/B_0$', plt.cm.coolwarm),
        (ne, '(h) $n_e$', r'$n_e$ Count', plt.cm.viridis),
        (Jx, '(i) $J_x$', r'$J_x$', plt.cm.RdBu_r),
        (Jy, '(j) $J_y$', r'$J_y$', plt.cm.RdBu_r),
        (Jz, '(k) $J_z$', r'$J_z$', plt.cm.RdBu_r),
        (ni, '(l) $n_i$', r'$n_i$ Count', plt.cm.viridis),
        (Vxi, r'(m) $V_{ix}/V_{A0}$', r'$V_{ix}/V_{A0}$', plt.cm.RdBu_r),
        (Vyi, r'(n) $V_{iy}/V_{A0}$', r'$V_{iy}/V_{A0}$', plt.cm.RdBu_r),
        (Vzi, r'(o) $V_{iz}/V_{A0}$', r'$V_{iz}/V_{A0}$', plt.cm.RdBu_r),
        (Ti_proxy, '(p) $T_i$ (Proxy)', r'$T_i$', plt.cm.plasma),
        (Vxe, r'(q) $V_{ex}/V_{A0}$', r'$V_{ex}/V_{A0}$', plt.cm.RdBu_r),
        (Vye, r'(r) $V_{ey}/V_{A0}$', r'$V_{ey}/V_{A0}$', plt.cm.RdBu_r),
        (Vze, r'(s) $V_{ez}/V_{A0}$', r'$V_{ez}/V_{A0}$', plt.cm.RdBu_r),
        (Te_proxy, '(t) $T_e$ (Proxy)', r'$T_e$', plt.cm.plasma),
    ]
    
    for i, (Z, title, label, cmap) in enumerate(combined_plots):
        if i < len(ax_list):
            ax = ax_list[i]
            vmin, vmax = get_plot_range(Z)
            stream_color = 'white' if cmap == plt.cm.seismic else 'gray'
            # (Bx, By は規格化済みの値を渡す)
            cf = plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap=cmap, vmin=vmin, vmax=vmax, stream_color=stream_color)
        else:
            break

    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=0.5)
    output_filename_combined = os.path.join(OUTPUT_DIR, 'allcombined', f'plot_combined_{timestep}.png')
    plt.savefig(output_filename_combined, dpi=300)
    plt.close(fig)
    print(f"-> 全てを含む統合パネル (5x4) を {output_filename_combined} に保存しました。")
    print(f"--- タイムステップ: {timestep} の処理完了 ---")
    
# =======================================================
# ★★★ スクリプト実行ブロック (ループ処理) ★★★
# =======================================================
if __name__ == "__main__":
    # Matplotlibのフォント設定
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    # --- 引数の解析 (start, end, interval) ---
    
    # 引数の数をチェック (スクリプト名 + 3つの引数 = 合計4つ)
    if len(sys.argv) != 4:
        print("使用方法: python visual_fields.py [start_timestep] [end_timestep] [interval]")
        print("       (タイムステップは \"000500\" ではなく 500 のように数値で指定してください)")
        print("例: python visual_fields.py 0 14000 500")
        sys.exit(1)
        
    try:
        # 引数を整数として読み込む
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

    # --- ループ処理の実行 ---
    current_step = start_step
    while current_step <= end_step:
        
        # タイムステップを6桁のゼロ埋め文字列にフォーマット
        # (例: 500 -> "000500", 14000 -> "014000")
        timestep_str = f"{current_step:06d}"
        
        try:
            # メインの処理関数を呼び出し
            process_timestep(timestep_str)
            
        except Exception as e:
            # ★ 重大なエラーが発生しても、次のステップに進む
            print(f"★★ 重大なエラー: タイムステップ {timestep_str} の処理中に例外が発生しました: {e}")
            import traceback
            traceback.print_exc()
            print("     処理を続行します...")
            
        # 次のステップへ
        current_step += interval

    print(f"\n--- 全てのタイムステップの処理が完了しました ---")