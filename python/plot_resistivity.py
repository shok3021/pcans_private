import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 設定・共通関数
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.path.abspath('.')
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resistivity_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_FILE_PATH = os.path.join('/data/shok/dat/init_param.dat')
NX_PHYS, NY_PHYS, DELX, C_LIGHT, FPI, DT, FGI, VA0, MI, QI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI
B0 = (FGI * MI * C_LIGHT) / QI

print(f"--- Setup: B0={B0:.4f}, VA0={VA0:.4f}")

# =======================================================
# 抵抗計算ヘルパー
# =======================================================
def load_data(path, shape):
    try: return np.loadtxt(path, delimiter=',')
    except: return np.zeros(shape)

def calculate_resistivities(timestep):
    shape = (NY_PHYS, NX_PHYS)
    
    # 1. 場の読み込み (正規化前)
    Bx = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt'), shape)
    By = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt'), shape)
    # Bz = ... (必要なら)
    Ez = load_data(os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Ez.txt'), shape)
    
    # 2. 粒子データの読み込み
    Vxe = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_Vx.txt'), shape)
    Vye = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_Vy.txt'), shape)
    
    # 電流密度 Jz の計算 (n_e * (Vzi - Vze) * q だが、簡易的に Jz ~ Curl B で代用または提供データを使用)
    # ここではモーメントデータから Jz を再構成します (あるいは extracted_data に Jz があればそれを使用)
    Vze = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_Vz.txt'), shape)
    Vzi = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_ion_Vz.txt'), shape)
    ne = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_density_count.txt'), shape)
    
    # 電子温度 (eV)
    Te = load_data(os.path.join(MOMENT_DATA_DIR, f'data_{timestep}_electron_T.txt'), shape)

    # --- Jz の概算 (相対値) ---
    # J = n * q * (Vi - Ve). ここでは簡易的に相対強度として計算
    # 密度が非常に低い場所はノイズになるのでフィルタ
    ne_mean = np.mean(ne[ne > 0])
    n_proxy = ne / (ne_mean if ne_mean > 0 else 1.0)
    Jz = n_proxy * (Vzi - Vze)
    
    # --- A. 異常抵抗 (Anomalous Resistivity) ---
    # 定義: eta_anom = E_non_ideal_z / Jz
    # E_non_ideal_z = Ez + (Ve x B)z = Ez + (Vxe*By - Vye*Bx)
    # 全て生のコード単位で計算してから割るのが安全
    
    Ez_non_ideal = Ez + (Vxe * By - Vye * Bx)
    
    # ゼロ割り防止マスク
    J_threshold = 0.1 * np.std(Jz)  # Jzの標準偏差の10%以下は計算除外
    valid_mask = np.abs(Jz) > J_threshold
    
    eta_anom = np.zeros_like(Ez)
    eta_anom[valid_mask] = Ez_non_ideal[valid_mask] / Jz[valid_mask]
    
    # 絶対値をとる場合や、散逸(E'.J > 0)のみを見る場合は調整してください
    # ここでは符号込みの値を出しますが、プロット時に絶対値または正の部分を見ることが多いです
    
    # --- B. スパイツァー抵抗 (Spitzer Resistivity) ---
    # eta_spitzer ∝ Te^(-3/2)
    # 定数係数は 1.0e-4 程度ですが、ここでは相対分布を見るため係数は1とします
    # Te = 0 の場所での発散を防ぐ
    Te_safe = np.maximum(Te, 0.1) # 0.1 eV 下限
    eta_spitzer = np.power(Te_safe, -1.5)
    
    # 値のレンジが全く異なるため、それぞれ適した正規化をするか、logプロットを推奨
    
    return eta_anom, eta_spitzer, By, B0

# =======================================================
# プロット実行
# =======================================================
def plot_resistivity(timestep):
    print(f"Processing Timestep: {timestep}")
    
    eta_anom, eta_spitzer, By, B0_val = calculate_resistivities(timestep)
    
    # 磁束線用
    Psi = cumtrapz(By, dx=DELX, axis=1, initial=0)
    
    # 座標
    x = np.linspace(-NX_PHYS*DELX/2, NX_PHYS*DELX/2, NX_PHYS) / DI
    y = np.linspace(0, NY_PHYS*DELX, NY_PHYS) / DI
    X, Y = np.meshgrid(x, y)

    # プロット設定
    # 異常抵抗は値が飛びやすいので、vmin/vmaxを手動設定するか、ロバストな範囲指定を推奨
    anom_max = np.percentile(np.abs(eta_anom), 98) # 上位2%を最大値にする（外れ値除去）
    
    plots = [
        ('Anomalous_Resistivity', eta_anom, r'Anomalous Resistivity $\eta_{anom}$', plt.cm.RdBu_r, (-anom_max, anom_max)),
        ('Spitzer_Resistivity', eta_spitzer, r'Spitzer Resistivity ($\propto T_e^{-1.5}$)', plt.cm.plasma, (None, None))
    ]
    
    for name, data, title, cmap, (vmin, vmax) in plots:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if vmin is None: vmin = np.min(data)
        if vmax is None: vmax = np.max(data)
        
        cf = ax.contourf(X, Y, data, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # logスケールにしたい場合は下記を使用:
        # from matplotlib.colors import LogNorm
        # cf = ax.contourf(X, Y, np.abs(data)+1e-9, levels=100, cmap=cmap, norm=LogNorm())

        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(title, fontsize=16)
        
        ax.contour(X, Y, Psi, levels=20, colors='gray', linewidths=0.5, alpha=0.8)
        
        ax.set_title(f"Timestep {timestep}: {name}", fontsize=18)
        ax.set_xlabel('$x/d_i$', fontsize=16)
        ax.set_ylabel('$y/d_i$', fontsize=16)
        
        out_path = os.path.join(OUTPUT_DIR, f'{name}_{timestep}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f" Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_resistivity.py [timestep]")
        sys.exit()
    
    steps = sys.argv[1:]
    for step in steps:
        plot_resistivity(step)