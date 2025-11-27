import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 1. 設定と定数 (既存コードと互換性を維持)
# =======================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

# データのパス設定
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'mirror_analysis') # 保存先を変更
os.makedirs(OUTPUT_DIR, exist_ok=True)

# パラメータファイルのパス (環境に合わせて修正してください)
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# グローバル定数 (初期値)
GLOBAL_NX_PHYS = 320
GLOBAL_NY_PHYS = 639
DELX = 1.0

# =======================================================
# 2. ヘルパー関数: パラメータ読み込み
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

# パラメータロード実行
C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI
B0 = (FGI * MI * C_LIGHT) / QI

print(f"--- 規格化スケール: d_i={DI:.2f}, B0={B0:.2f}, VA0={VA0:.2f}")

# =======================================================
# 3. ヘルパー関数: データ読み込み
# =======================================================
def load_data(timestep, subdir, prefix, species=None):
    """汎用データ読み込み関数"""
    if species:
        filename = f'data_{timestep}_{species}_{prefix}.txt'
        path = os.path.join(MOMENT_DATA_DIR, filename)
    else:
        filename = f'data_{timestep}_{prefix}.txt'
        path = os.path.join(FIELD_DATA_DIR, filename)
    
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

# =======================================================
# 4. ★重要★ ミラー解析用計算関数
# =======================================================
def calculate_mirror_metrics(Bx, By, Bz, Vx, Vy, Vz, delx):
    """
    磁気ミラー効果に関連する物理量を計算する
    
    Returns:
        B_mag: 磁場強度 |B|
        V_par: 磁力線に平行な速度成分 (V dot b)
        Mirror_Force: ミラー力プロキシ (- grad_para |B|)
    """
    # 1. 磁場強度 |B|
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mag_safe = np.where(B_mag < 1e-6, 1e-6, B_mag) # ゼロ除算防止
    
    # 2. 磁力線単位ベクトル b = B / |B|
    bx = Bx / B_mag_safe
    by = By / B_mag_safe
    bz = Bz / B_mag_safe
    
    # 3. 平行速度 V_par
    V_par = Vx * bx + Vy * by + Vz * bz
    
    # 4. ミラー力プロキシの計算
    # 勾配 (gradient) の計算: np.gradient は (axis=0(y), axis=1(x)) の順で返す
    # ここでは相対的な形状が重要なので delx=1.0 でも傾向は見えるが、delxを入れるのが正確
    grad_B_y, grad_B_x = np.gradient(B_mag, delx)
    
    # 磁力線方向の勾配: (b dot grad) |B|
    # 注: 2Dシミュレーションなので dz(|B|) = 0 と仮定
    grad_para_B = bx * grad_B_x + by * grad_B_y
    
    # ミラー力 F = -mu * grad_para_B
    # 正の値 = 磁場が弱くなる方向への押し返し力 (加速)
    # 負の値 = 磁場が強くなる方向への力 (減速・反射)
    # ここでは「粒子を弱い磁場領域へ押し返す力」として定義
    Mirror_Force = -grad_para_B
    
    return B_mag, V_par, Mirror_Force

# =======================================================
# 5. プロット関数
# =======================================================
def plot_mirror_analysis(timestep, B_vec, V_ion, V_ele, Psi, omega_t_str):
    """
    4パネルのミラー解析図を作成
    (a) |B| (磁場の壁)
    (b) Mirror Force (押し返す力)
    (c) Ion Parallel Velocity (イオンの反射挙動)
    (d) Electron Parallel Velocity (電子の反射挙動)
    """
    Bx, By, Bz = B_vec
    Vxi, Vyi, Vzi = V_ion
    Vxe, Vye, Vze = V_ele
    
    # 計算実行
    B_mag, Vpar_i, F_mirror_i = calculate_mirror_metrics(Bx, By, Bz, Vxi, Vyi, Vzi, DELX)
    _,     Vpar_e, F_mirror_e = calculate_mirror_metrics(Bx, By, Bz, Vxe, Vye, Vze, DELX) # 力場は同じだが一応
    
    # 座標系
    x_phys = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS)
    y_phys = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, GLOBAL_NY_PHYS)
    X, Y = np.meshgrid(x_phys / DI, y_phys / DI)
    
    # 磁力線用 (Psiの等高線)
    Psi_contour = Psi

    # プロット設定
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    ax_list = axes.flatten()
    
    # 各パネルの設定
    panels = [
        {
            'data': B_mag, 
            'title': r'(a) Magnetic Field Strength $|B|/B_0$', 
            'cmap': 'inferno', 
            'label': r'$|B|/B_0$',
            'diverging': False
        },
        {
            'data': F_mirror_i, 
            'title': r'(b) Mirror Force Proxy ($-\nabla_{\parallel}|B|$)', 
            'cmap': 'PuOr', 
            'label': 'Force (a.u.)',
            'diverging': True # 0を中心に
        },
        {
            'data': Vpar_i, 
            'title': r'(c) Ion Parallel Velocity ($V_{\parallel i}/V_{A0}$)', 
            'cmap': 'RdBu_r', 
            'label': r'$V_{\parallel i}/V_{A0}$',
            'diverging': True
        },
        {
            'data': Vpar_e, 
            'title': r'(d) Electron Parallel Velocity ($V_{\parallel e}/V_{A0}$)', 
            'cmap': 'RdBu_r', 
            'label': r'$V_{\parallel e}/V_{A0}$',
            'diverging': True
        }
    ]
    
    fig.suptitle(f"Mirror Instability Analysis: Timestep {timestep}  ({omega_t_str})", fontsize=16)

    for ax, p in zip(ax_list, panels):
        data = p['data']
        
        # Range設定
        vmax = np.nanmax(np.abs(data))
        if p['diverging']:
            vmin, vmax = -vmax, vmax
        else:
            vmin = np.nanmin(data)
        
        # 描画
        cf = ax.contourf(X, Y, data, levels=100, cmap=p['cmap'], vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cf, ax=ax, format='%.2f')
        cbar.set_label(p['label'])
        
        # 磁力線の重ね書き (重要: 粒子はこれに沿って動く)
        ax.contour(X, Y, Psi_contour, levels=20, colors='gray', linewidths=0.6, alpha=0.6)
        
        ax.set_title(p['title'])
        ax.set_ylabel('$y/d_i$')
        ax.set_xlabel('$x/d_i$')
        ax.tick_params(direction='in')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(OUTPUT_DIR, f'mirror_plot_{timestep}.png')
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> 保存完了: {out_path}")

# =======================================================
# 6. メインループ
# =======================================================
def process_timestep(timestep):
    print(f"処理中: {timestep} ...")
    
    # データの読み込み
    Bx_raw = load_data(timestep, FIELD_DATA_DIR, 'Bx')
    By_raw = load_data(timestep, FIELD_DATA_DIR, 'By')
    Bz_raw = load_data(timestep, FIELD_DATA_DIR, 'Bz')
    
    Vxi_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vx', 'ion')
    Vyi_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vy', 'ion')
    Vzi_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vz', 'ion')
    
    Vxe_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vx', 'electron')
    Vye_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vy', 'electron')
    Vze_raw = load_data(timestep, MOMENT_DATA_DIR, 'Vz', 'electron')

    # 規格化
    B_vec = (Bx_raw/B0, By_raw/B0, Bz_raw/B0)
    V_ion = (Vxi_raw/VA0, Vyi_raw/VA0, Vzi_raw/VA0)
    V_ele = (Vxe_raw/VA0, Vye_raw/VA0, Vze_raw/VA0)
    
    # 磁束関数の計算 (磁力線描画用)
    Psi = cumtrapz(By_raw/B0, dx=1.0, axis=1, initial=0)
    
    # 時間表記
    try:
        omega_t = FGI * float(timestep) * DT
        omega_t_str = fr"$\Omega_{{ci}}t = {omega_t:.2f}$"
    except:
        omega_t_str = ""

    # プロット実行
    plot_mirror_analysis(timestep, B_vec, V_ion, V_ele, Psi, omega_t_str)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visual_mirror.py [start] [end] [interval]")
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    interval = int(sys.argv[3])

    for step in range(start, end + 1, interval):
        process_timestep(f"{step:06d}")