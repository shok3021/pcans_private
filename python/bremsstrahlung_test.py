import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
from scipy.constants import m_e, c, elementary_charge 

# =======================================================
# ★ 設定パラメータ
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge

# 熱的/非熱的の分離しきい値 (局所温度の何倍か)
# マクスウェル分布の裾野(～10kTe)を超えるものを非熱的とする
ALPHA_THRESHOLD = 10.0 

# グリッド設定
GLOBAL_NX_GRID = 321
GLOBAL_NY_GRID = 640
NX = GLOBAL_NX_GRID - 1
NY = GLOBAL_NY_GRID - 1
DELX = 1.0
DI_PARAM = 100.0 # プロット軸用のスキンデプス長

# 座標範囲
X_MIN, X_MAX = 0.0, NX * DELX
Y_MIN, Y_MAX = 0.0, NY * DELX

# =======================================================
# 1. 計算エンジン: 制動放射強度マップの作成
# =======================================================
def calculate_bremsstrahlung_intensity(particle_data):
    """
    物理ベースの制動放射強度(Intensity)を計算し、
    熱的(Thermal)と非熱的(Non-Thermal)に分離してマップ化する。
    
    Intensity ∝ (イオン密度) * (電子流束)
              ∝ (N_total) * Sum(sqrt(E))
    """
    # --- データ展開 ---
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    vx = particle_data[:, 2]
    vy = particle_data[:, 3]
    vz = particle_data[:, 4]

    # --- エネルギー計算 (keV) ---
    print("  -> (1/5) エネルギー計算...")
    v_sq = vx**2 + vy**2 + vz**2
    v_sq = np.clip(v_sq, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v_sq)
    E_kin_keV = ((gamma - 1.0) * m_e * c**2) / KEV_TO_J

    # --- グリッド定義 ---
    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)

    # --- 局所温度・密度の計算 ---
    print("  -> (2/5) 局所温度・密度マップ作成...")
    # ヒストグラムでグリッドごとの粒子数(N)と総エネルギー(Sum E)を計算
    H_count, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges])
    H_sum_E, _, _ = np.histogram2d(Y_pos, X_pos, bins=[y_edges, x_edges], weights=E_kin_keV)

    # 平均エネルギー <E>
    with np.errstate(divide='ignore', invalid='ignore'):
        Mean_E_map = H_sum_E / H_count
        Mean_E_map[H_count == 0] = 0.0

    # 局所温度 Te ≈ (2/3)<E>
    T_local_map = (2.0 / 3.0) * Mean_E_map
    
    # 密度マップ (ターゲットとなるイオン密度と仮定)
    # ※シミュレーションの粒子重みに応じてスケールが変わりますが、相対比較には count で十分
    Density_map = H_count 

    # --- 粒子の選別 (Thermal vs Non-Thermal) ---
    print(f"  -> (3/5) 粒子振り分け (閾値: {ALPHA_THRESHOLD} * T_local)...")
    
    # 各粒子のグリッド座標を取得
    ix = np.clip(np.digitize(X_pos, x_edges) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(Y_pos, y_edges) - 1, 0, NY - 1)

    # 粒子の位置での温度を取得
    T_particle = T_local_map[iy, ix]

    # しきい値判定
    threshold = ALPHA_THRESHOLD * T_particle
    is_nonthermal = (E_kin_keV > threshold) & (T_particle > 1e-6)
    is_thermal    = ~is_nonthermal

    # --- 放射強度の積算 (Sum of sqrt(E)) ---
    print("  -> (4/5) 放射強度(Intensity)の積算...")
    
    # 制動放射パワー P ∝ v ∝ sqrt(E)
    Emission_weight = np.sqrt(E_kin_keV)

    # Thermal成分の放射源項 (Sum v)
    Source_Thermal = np.zeros((NY, NX))
    np.add.at(Source_Thermal, (iy[is_thermal], ix[is_thermal]), Emission_weight[is_thermal])
    
    # Non-Thermal成分の放射源項 (Sum v)
    Source_NonThermal = np.zeros((NY, NX))
    np.add.at(Source_NonThermal, (iy[is_nonthermal], ix[is_nonthermal]), Emission_weight[is_nonthermal])

    # --- 最終的なIntensityマップ ---
    # I = Density * Source (n * Σv ∝ n^2 * sqrt(T))
    print("  -> (5/5) マップ合成...")
    
    Intensity_Thermal = Density_map * Source_Thermal
    Intensity_NonThermal = Density_map * Source_NonThermal

    return Intensity_Thermal, Intensity_NonThermal

# =======================================================
# 2. 保存＆プロット関数
# =======================================================
def save_and_plot(map_data, label, timestep, output_base_dir):
    # --- 保存用ディレクトリ ---
    save_dir = os.path.join(output_base_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- TXT保存 ---
    txt_filename = os.path.join(save_dir, f'intensity_{label}_{timestep}.txt')
    header = (f'Bremsstrahlung Intensity Map ({label})\n'
              f'Timestep: {timestep}\n'
              f'Unit: Arbitrary (Density * Sum_sqrt_E)\n'
              f'Shape: ({NY}, {NX})')
    np.savetxt(txt_filename, map_data, header=header, fmt='%.6g')
    print(f"    Saved TXT: {txt_filename}")
    
    # --- PNGプロット (確認用) ---
    # 軸作成
    X_axis = np.linspace(X_MIN, X_MAX, NX) / DI_PARAM
    Y_axis = np.linspace(Y_MIN, Y_MAX, NY) / DI_PARAM
    XX, YY = np.meshgrid(X_axis, Y_axis)
    
    fig, ax = plt.subplots(figsize=(8, 10)) # 縦長レイアウトに合わせて調整
    
    # ログスケール表示 (0以下はマスク)
    data_plot = np.where(map_data > 1e-5, map_data, np.nan)
    
    cmap = 'Reds' if label == 'Thermal' else 'Blues'
    
    if np.nanmax(data_plot) > 0:
        vmin = np.nanpercentile(data_plot, 5) if np.nanpercentile(data_plot, 5) > 0 else 1e-2
        vmax = np.nanmax(data_plot)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(XX, YY, data_plot, cmap=cmap, norm=norm, shading='auto')
        plt.colorbar(pcm, ax=ax, label='Intensity [a.u.]')
    else:
        ax.text(0.5, 0.5, 'No Signal', ha='center', transform=ax.transAxes)
    
    ax.set_title(f'{label} Bremsstrahlung Intensity (TS: {timestep})')
    ax.set_xlabel('$x / d_i$')
    ax.set_ylabel('$y / d_i$')
    ax.set_aspect('equal')
    
    img_filename = os.path.join(save_dir, f'plot_{label}_{timestep}.png')
    plt.savefig(img_filename, dpi=150)
    plt.close()
    # print(f"    Saved IMG: {img_filename}")

# =======================================================
# 3. メイン実行部
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python bremsstrahlung_intensity.py [start] [end] [step]")
        sys.exit(1)
    
    start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # パス設定
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')
    
    # データ読み込み元
    DATA_DIR = '/home/shok/pcans/em2d_mpi/md_mrx/psd/'
    
    # 出力先
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_intensity_txt')
    
    print(f"--- Bremsstrahlung Intensity Calculation ---")
    print(f"Output Directory: {OUTPUT_DIR}")

    for t_int in range(start, end + step, step):
        ts = f"{t_int:06d}"
        filename = f'{ts}_0160-0320_psd_e.dat'
        filepath = os.path.join(DATA_DIR, filename)

        print(f"\n=== Processing TimeStep: {ts} ===")
        
        if not os.path.exists(filepath):
            print(f"  Skip: {filename} not found.")
            continue

        try:
            data = np.loadtxt(filepath)
            if data.size == 0: continue
            if data.ndim == 1: data = data.reshape(1, -1)
        except:
            print("  Error loading file.")
            continue

        # 計算実行
        Int_T, Int_NT = calculate_bremsstrahlung_intensity(data)

        # 保存とプロット
        save_and_plot(Int_T, 'Thermal', ts, OUTPUT_DIR)
        save_and_plot(Int_NT, 'NonThermal', ts, OUTPUT_DIR)

    print("\n--- All Finished ---")

if __name__ == "__main__":
    main()