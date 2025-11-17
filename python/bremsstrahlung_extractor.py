import numpy as np
import os
import sys
from scipy.constants import m_e, c, elementary_charge
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d # EEDFフィットのためにscipyをインポート

# =======================================================
# 物理定数 (計算用)
# =======================================================
KEV_TO_J = 1000.0 * elementary_charge
M_E_KEV = (m_e * c**2) / KEV_TO_J

# =======================================================
# ヘルパー関数 (変更なし)
# =======================================================

def load_raw_particle_data(filepath):
    """
    Fortranが出力した psd_*.dat (生の粒子リスト) を読み込む。
    """
    if not os.path.exists(filepath):
        print(f"    エラー: ファイルが見つかりません: {filepath}")
        return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            if data.size == 0:
                 print("    -> ファイルは空です。")
                 return np.array([])
            data = data.reshape(1, -1)
        if data.size == 0:
            print("    -> ファイルは空です。")
            return np.array([])
        return data
    except Exception as e:
        print(f"    エラー: {filepath} のテキスト読み込みに失敗: {e}")
        return None

def calculate_EEDF_from_particles(particle_data, n_bins=200, E_max_keV=50.0):
    """
    生の粒子データから電子エネルギー分布関数 (EEDF) を計算する。
    v(3:5) は 速度/c (cは光速) と仮定する。
    """
    v_norm_x = particle_data[:, 2] # vx/c
    v_norm_y = particle_data[:, 3] # vy/c
    v_norm_z = particle_data[:, 4] # vz/c
    
    v_norm_sq = v_norm_x**2 + v_norm_y**2 + v_norm_z**2
    v_norm_sq = v_norm_sq[v_norm_sq < 1.0] # 光速超過を除外
    
    if v_norm_sq.size == 0:
        print("    警告: 有効な速度データを持つ粒子が見つかりません。")
        return None, None

    gamma = 1.0 / np.sqrt(1.0 - v_norm_sq)
    E_kin_J = (gamma - 1.0) * m_e * (c**2)
    E_kin_keV = E_kin_J / KEV_TO_J
    
    bin_edges = np.linspace(0, E_max_keV, n_bins + 1)
    eedf, _ = np.histogram(E_kin_keV, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    dE = bin_edges[1] - bin_edges[0]
    eedf_normalized = eedf / dE
    
    return bin_centers, eedf_normalized

def fit_thermal_EEDF(E_keV, eedf):
    """
    EEDFの低エネルギー部分にマックスウェル分布をフィッティングする。
    """
    def maxwellian_E(E, A, T_keV):
        if T_keV < 0: return np.inf
        return A * np.sqrt(E) * np.exp(-E / T_keV)

    try:
        eedf_smooth = gaussian_filter1d(eedf, sigma=2)
        peak_index = np.argmax(eedf_smooth)
        
        E_fit = E_keV[:peak_index+1]
        eedf_fit = eedf[:peak_index+1]
        
        mask = E_fit > 0.1
        if np.sum(mask) < 2: mask = E_fit > 0.01
        
        if np.sum(mask) < 2:
            print("    警告: 熱的成分のフィッティングに失敗 (データ不足)。")
            return None, None, None

        E_fit_masked = E_fit[mask]
        eedf_fit_masked = eedf_fit[mask]

        T_guess = E_keV[peak_index] * 0.5 
        A_guess = np.max(eedf_fit_masked) / (np.sqrt(T_guess) * np.exp(-1))
        
        popt, pcov = curve_fit(maxwellian_E, E_fit_masked, eedf_fit_masked, p0=[A_guess, T_guess])
        
        A_fit, T_keV_fit = popt
        fit_line = maxwellian_E(E_keV, A_fit, T_keV_fit)
        
        print(f"    -> 熱的成分フィット: T_e = {T_keV_fit:.3f} keV, A = {A_fit:.2e}")
        
        return T_keV_fit, A_fit, fit_line

    except Exception as e:
        print(f"    警告: 熱的成分のフィッティング中にエラー: {e}")
        return None, None, None

def calculate_thermal_spectrum(A_fit, T_keV_fit, energy_bins_keV):
    """
    フィッティングした熱的パラメータから制動放射スペクトルを計算 (Kramers' law 近似)
    """
    if T_keV_fit is None or A_fit is None:
        return np.zeros_like(energy_bins_keV)
        
    pre_factor = A_fit * T_keV_fit
    spectrum = pre_factor * np.exp(-energy_bins_keV / T_keV_fit)
    
    return spectrum

def calculate_non_thermal_spectrum(E_keV_raw, eedf_raw, plot_energy_bins_keV):
    """
    EEDF全体 (非熱的成分) から制動放射スペクトルを計算 (Thin-target approx)
    """
    E_keV = E_keV_raw
    eedf = eedf_raw
    
    integrand = np.zeros_like(E_keV)
    valid_mask = E_keV > 1e-6
    integrand[valid_mask] = eedf[valid_mask] / np.sqrt(E_keV[valid_mask])
    
    dE = E_keV[1] - E_keV[0]
    
    cumulative_integral = np.cumsum(integrand[::-1])[::-1] * dE
    
    spectrum_raw = np.zeros_like(E_keV)
    spectrum_raw[valid_mask] = (1.0 / E_keV[valid_mask]) * cumulative_integral[valid_mask]
    
    interpolated_spectrum = np.interp(
        plot_energy_bins_keV, 
        E_keV, 
        spectrum_raw, 
        left=spectrum_raw[0],
        right=0
    )
    
    return interpolated_spectrum

# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python bremsstrahlung_save_txt.py [開始] [終了] [間隔]")
        print("例: python bremsstrahlung_save_txt.py 0 14000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: 引数は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")
    
    # ★ 入力: Fortran が psd_*.dat を出力するディレクトリ
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    # ★ 出力: .txt データを保存するディレクトリ
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_txt') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 生の粒子データ入力元: {data_dir} ---")
    print(f"--- TXTデータ出力先: {OUTPUT_DIR} ---")
    
    species_suffix = 'e'
    species_label = 'electron'
    
    # (プロット用のエネルギー範囲: 0.1 keV ～ 100 keV)
    plot_energy_bins = np.logspace(np.log10(0.1), np.log10(100.0), 50)
    # (EEDF計算用のエネルギー範囲: 0 keV ～ 50 keV)
    EEDF_MAX_KEV = 50.0
    EEDF_N_BINS = 200

    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        filename = f'{timestep}_0160-0320_psd_{species_suffix}.dat'
        filepath = os.path.join(data_dir, filename)
        
        print(f"--- {species_label} の生粒子データ ({filename}) を処理中 ---")

        particle_data = load_raw_particle_data(filepath)
        
        if particle_data is None or particle_data.size == 0:
            print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
            continue
        
        print(f"  -> {len(particle_data)} 個の粒子を読み込みました。EEDFを計算中...")

        # --- 1. EEDFの計算 ---
        E_keV_raw, eedf_raw = calculate_EEDF_from_particles(
            particle_data, n_bins=EEDF_N_BINS, E_max_keV=EEDF_MAX_KEV
        )
        if E_keV_raw is None:
            continue

        # --- 2. EEDFから熱的成分をフィッティング ---
        T_keV_fit, A_fit, thermal_fit_line = fit_thermal_EEDF(E_keV_raw, eedf_raw)
        
        # --- 3. 熱的スペクトルの計算 ---
        thermal_spec = calculate_thermal_spectrum(A_fit, T_keV_fit, plot_energy_bins)
        
        # --- 4. 非熱的スペクトルの計算 (EEDF全体を使用) ---
        non_thermal_spec = calculate_non_thermal_spectrum(E_keV_raw, eedf_raw, plot_energy_bins)
        
        print("  -> スペクトル計算完了。TXTファイルに保存中...")

        # --- 5. TXTファイルへの保存 ---
        
        # (A) EEDF データの保存 (3列: E_keV, EEDF_raw, Thermal_Fit)
        if thermal_fit_line is None:
            # フィット失敗時は 0 の配列
            thermal_fit_line = np.zeros_like(E_keV_raw)
            
        eedf_data_to_save = np.stack((E_keV_raw, eedf_raw, thermal_fit_line), axis=-1)
        eedf_filename = os.path.join(OUTPUT_DIR, f'eedf_{timestep}.txt')
        np.savetxt(eedf_filename, eedf_data_to_save, 
                   header='Column 1: E_keV (bin centers)\nColumn 2: EEDF_raw (dN/dE)\nColumn 3: Thermal_Fit_Line',
                   fmt='%.6e')
        print(f"  -> EEDFデータを {eedf_filename} に保存しました。")

        # (B) スペクトルデータの保存 (3列: E_keV, Thermal_Spec, NonThermal_Spec)
        spectrum_data_to_save = np.stack((plot_energy_bins, thermal_spec, non_thermal_spec), axis=-1)
        spectrum_filename = os.path.join(OUTPUT_DIR, f'spectrum_{timestep}.txt')
        np.savetxt(spectrum_filename, spectrum_data_to_save,
                   header='Column 1: Photon_Energy_keV (bin centers)\nColumn 2: Thermal_Spectrum_Intensity\nColumn 3: NonThermal_Spectrum_Intensity',
                   fmt='%.6e')
        print(f"  -> スペクトルデータを {spectrum_filename} に保存しました。")

        print(f"--- タイムステップ {timestep} のTXTデータ保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")

if __name__ == "__main__":
    main()