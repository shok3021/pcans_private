import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.constants import m_e, c, elementary_charge
from multiprocessing import Pool, cpu_count

# =======================================================
# 定数・設定
# =======================================================
PARAM_FILE_PATH = '/data/shok/dat_default/init_param.dat'
MC2_EV_ELECTRON = (m_e * c**2) / elementary_charge # 0.511 MeV

def load_params():
    p = {'nx': 1600, 'ny': 639, 'delx': 0.2, 'mi': 100.0} # デフォルト
    try:
        with open(PARAM_FILE_PATH, 'r') as f:
            for line in f:
                if 'grid size' in line:
                    parts = line.split()
                    p['nx'], p['ny'] = int(parts[5].replace('x',''))-1, int(parts[6])-1
                    if len(parts)>7: p['delx'] = float(parts[7])
                elif 'Mi, Me' in line: p['mi'] = float(line.split()[3])
    except: pass
    return p

P = load_params()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
FIELD_DIR = os.path.join(SCRIPT_DIR, 'past_data/default/extracted_data')
MOMENT_DIR = os.path.join(SCRIPT_DIR, 'past_data/default/extracted_moments_tensor')
OUT_DIR = os.path.join(SCRIPT_DIR, 'past_data/default/energy_plots')

def load_2d(path):
    try: return pd.read_csv(path, header=None, delimiter=',', engine='c').values
    except: return np.zeros((P['ny'], P['nx']))

def process_timestep(ts_int):
    ts = f"{ts_int:06d}"
    print(f"Plotting: {ts}")
    
    # 1. 磁場読み込み (単位ベクトル作成用)
    Bx = load_2d(os.path.join(FIELD_DIR, f'data_{ts}_Bx.txt'))
    By = load_2d(os.path.join(FIELD_DIR, f'data_{ts}_By.txt'))
    Bz = load_2d(os.path.join(FIELD_DIR, f'data_{ts}_Bz.txt'))
    
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        bx, by, bz = Bx/B_mag, By/B_mag, Bz/B_mag
    for b in [bx, by, bz]: b[np.isnan(b)] = 0.0

    # 2. 速度分散テンソル読み込み (単位: (v/c)^2)
    Sxx = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Sxx.txt'))
    Syy = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Syy.txt'))
    Szz = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Szz.txt'))
    Sxy = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Sxy.txt'))
    Sxz = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Sxz.txt'))
    Syz = load_2d(os.path.join(MOMENT_DIR, f'data_{ts}_electron_Syz.txt'))

    # 3. エネルギー計算 (Energy = 1/2 * m * sigma^2) [eV]
    # 係数: 0.5 * mc^2 [eV]
    factor = 0.5 * MC2_EV_ELECTRON
    
    # テンソル回転: 平行成分 S_para = b . S . b
    S_para = (bx**2 * Sxx + by**2 * Syy + bz**2 * Szz +
              2*(bx*by*Sxy + bx*bz*Sxz + by*bz*Syz))
    
    S_trace = Sxx + Syy + Szz
    S_perp = (S_trace - S_para) / 2.0  # 垂直方向「1自由度あたり」の分散

    # エネルギーに変換
    E_para = S_para * factor
    E_perp = S_perp * factor
    
    # 異方性 (E_perp / E_para)
    with np.errstate(invalid='ignore', divide='ignore'):
        Aniso = E_perp / E_para
        Aniso[E_para < 1.0] = np.nan # 低エネルギー部はノイズになるので隠す

    # 磁力線
    Psi = cumtrapz(By, dx=P['delx'], axis=1, initial=0)
    
    # 4. プロット
    os.makedirs(os.path.join(OUT_DIR, ts), exist_ok=True)
    
    # Grid
    X, Y = np.linspace(0, P['nx']*P['delx'], P['nx']), np.linspace(0, P['ny']*P['delx'], P['ny'])
    Xm, Ym = np.meshgrid(X, Y)

    plot_list = [
        (E_para, r'Parallel Energy $\frac{1}{2}m\langle v_\parallel^2 \rangle$ (eV)', 'E_para', 'plasma'),
        (E_perp, r'Perp Energy $\frac{1}{2}m\langle v_\perp^2 \rangle$ (eV)',   'E_perp', 'plasma'),
        (Aniso,  r'Anisotropy $E_\perp / E_\parallel$', 'Anisotropy', 'bwr')
    ]

    for data, title, tag, cmap in plot_list:
        fig, ax = plt.subplots(figsize=(10,5))
        
        # 範囲設定
        if tag == 'Anisotropy':
            vmin, vmax = 0.5, 1.5
        else:
            vmin, vmax = 0, np.nanpercentile(data, 98)

        m = ax.pcolormesh(Xm, Ym, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.contour(Xm, Ym, Psi, levels=20, colors='k', alpha=0.4, linewidths=0.5)
        
        plt.colorbar(m, ax=ax, label=title)
        ax.set_title(f"TS: {ts} | {tag}")
        ax.set_aspect('equal')
        
        plt.savefig(os.path.join(OUT_DIR, ts, f'{tag}.png'), dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_energy.py [start] [end] [step]")
        sys.exit(1)
        
    s, e, st = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    with Pool(cpu_count()) as p:
        p.map(process_timestep, range(s, e+st, st))