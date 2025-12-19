import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 物理定数 (SI単位系)
# =======================================================
m_e_SI = 9.109e-31  # kg
m_i_SI = 1.672e-27  # kg (Proton assumption, adjust if needed)
e_SI = 1.602e-19    # C

# =======================================================
# 共通関数 (パラメータ読み込みなど)
# =======================================================
def load_simulation_parameters(param_filepath):
    """init_param.dat を読み込みパラメータを抽出"""
    NX = NY = DELX = C_LIGHT = FPI = DT = FGI = VA0 = MI = QI = None
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                sline = line.strip()
                if sline.startswith('grid size, debye lngth'):
                    parts = sline.split()
                    raw_nx = int(parts[5].replace('x', ''))
                    raw_ny = int(parts[6])
                    NX = raw_nx - 1
                    NY = raw_ny - 1
                    if len(parts) > 7: DELX = float(parts[7])
                elif sline.startswith('dx, dt, c'):
                    parts = sline.split()
                    if DELX is None: DELX = float(parts[4])
                    DT = float(parts[5])
                    C_LIGHT = float(parts[6])
                elif sline.startswith('Va, Vi, Ve'):
                    VA0 = float(sline.split()[7])
                elif sline.startswith('Mi, Me'):
                    MI = float(sline.split()[3])
                elif sline.startswith('Qi, Qe'):
                    QI = float(sline.split()[3])
                elif sline.startswith('Fpe, Fge, Fpi Fgi'):
                    parts = sline.split()
                    FPI = float(parts[7])
                    FGI = float(parts[8])
    except FileNotFoundError:
        print(f"File not found: {param_filepath}")
        sys.exit(1)
    
    return NX, NY, DELX, C_LIGHT, FPI, DT, FGI, VA0, MI, QI

# =======================================================
# 設定
# =======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.path.abspath('.')
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'larmor_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_FILE_PATH = os.path.join('/data/shok/dat/init_param.dat') 
# ※ ローカルテスト用パスなど適宜変更してください

# パラメータロード
NX_PHYS, NY_PHYS, DELX, C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)

DI = C_LIGHT / FPI
B0 = (FGI * MI * C_LIGHT) / QI

print(f"--- Scale: di={DI:.4f}, B0={B0:.4f}, VA0={VA0:.4f}")

# =======================================================
# データ読み込みヘルパー
# =======================================================
def load_data(path, shape):
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.shape != shape: return np.zeros(shape)
        return data
    except: return np.zeros(shape)

# =======================================================
# ラーマ半径 計算ロジック
# =======================================================
def calculate_larmor_radius(Te_eV, Ti_eV, Bx, By, Bz, B0_val, di_val):
    """
    Te, Ti (eV), B (normalized by B0) を用いて、
    Larmor半径 r_L を計算し、d_i で規格化して返す。
    r_L = m * v_perp / (q * B) ~ m * v_th / (q * B)
    """
    # 磁場の大きさ (Tesla換算用係数 B0_val を掛ける)
    # ここでは相対比を見るため、数式を一貫させます
    B_norm_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # ゼロ割り防止 (非常に小さな磁場はクリップ)
    B_norm_mag = np.maximum(B_norm_mag, 1e-4) 

    # 物理量への変換 (概算)
    # B_phys = B_norm_mag * B0_val (Tesla)
    # v_th_e = sqrt(2 * Te_eV * e / m_e)
    # r_Le = m_e * v_th_e / (e * B_phys)
    #      = sqrt(2 * m_e * Te_eV / e) / B_phys
    
    # 定数係数
    coeff_e = np.sqrt(2 * m_e_SI * e_SI) / e_SI  # Electron
    coeff_i = np.sqrt(2 * m_i_SI * e_SI) / e_SI  # Ion (Proton)

    # ラーマ半径 (メートル)
    # Te_eV はすでに eV 単位
    r_Le_meters = (coeff_e * np.sqrt(Te_eV)) / (B_norm_mag * B0_val)
    r_Li_meters = (coeff_i * np.sqrt(Ti_eV)) / (B_norm_mag * B0_val)

    # d_i で規格化
    return r_Le_meters / di_val, r_Li_meters / di_val

# =======================================================
# プロット関数
# =======================================================
def plot_larmor(timestep):
    print(f"Processing Timestep: {timestep}")
    
    # データ読み込み
    f_Bx = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    f_By = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    f_Bz = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bz.txt')
    
    f_Te = os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_T.txt')
    f_Ti = os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_ion_T.txt')
    
    shape = (NY_PHYS, NX_PHYS)
    Bx = load_data(f_Bx, shape) / B0  # Normalize B
    By = load_data(f_By, shape) / B0
    Bz = load_data(f_Bz, shape) / B0
    Te = load_data(f_Te, shape)       # eV
    Ti = load_data(f_Ti, shape)       # eV

    # ラーマ半径計算 (単位: di)
    rLe_norm, rLi_norm = calculate_larmor_radius(Te, Ti, Bx, By, Bz, B0, DI)
    
    # 磁束 (等高線用)
    Psi = cumtrapz(By * B0, dx=DELX, axis=1, initial=0) # Raw B for flux

    # 座標
    x = np.linspace(-NX_PHYS*DELX/2, NX_PHYS*DELX/2, NX_PHYS) / DI
    y = np.linspace(0, NY_PHYS*DELX, NY_PHYS) / DI
    X, Y = np.meshgrid(x, y)
    
    # プロットリスト
    plots = [
        ('Electron_Larmor', rLe_norm, r'Electron Larmor Radius ($r_{Le} / d_i$)', plt.cm.inferno, (0, 0.5)),
        ('Ion_Larmor', rLi_norm, r'Ion Larmor Radius ($r_{Li} / d_i$)', plt.cm.inferno, (0, 2.0))
    ]
    
    for name, data, title, cmap, clim in plots:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 外れ値除去 (最大値を制限して色を見やすくする)
        # 平均値の数倍などでクリップする場合: vmax = np.mean(data)*5 など
        # ここでは固定レンジまたは自動
        vmin, vmax = clim
        
        cf = ax.contourf(X, Y, data, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cf, ax=ax, format='%.2f')
        cbar.set_label(title, fontsize=16)
        
        # 磁力線
        ax.contour(X, Y, Psi, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        
        ax.set_title(f"Timestep {timestep}: {name}", fontsize=18)
        ax.set_xlabel('$x/d_i$', fontsize=16)
        ax.set_ylabel('$y/d_i$', fontsize=16)
        
        out_path = os.path.join(OUTPUT_DIR, f'{name}_{timestep}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f" Saved: {out_path}")

# =======================================================
# メイン実行
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_larmor.py [timestep] (e.g., 10000)")
        sys.exit()
    
    steps = sys.argv[1:]
    for step in steps:
        plot_larmor(step)