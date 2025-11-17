import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =======================================================
# メイン処理 (プロット)
# =======================================================
def plot_data(timestep):
    """
    指定されたタイムステップのTXTデータを読み込んでプロットする
    """
    
    # (Jupyterなどでの実行にも対応)
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    # ★ 入力: TXTデータが保存されているディレクトリ
    DATA_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_txt')
    # ★ 出力: プロット画像 (.png) を保存するディレクトリ
    PLOT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots')
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # --- ファイルパスの定義 ---
    eedf_filepath = os.path.join(DATA_DIR, f'eedf_{timestep}.txt')
    spectrum_filepath = os.path.join(DATA_DIR, f'spectrum_{timestep}.txt')

    # --- データの読み込み ---
    try:
        # (E_keV, EEDF_raw, Thermal_Fit)
        eedf_data = np.loadtxt(eedf_filepath)
        E_keV_raw = eedf_data[:, 0]
        eedf_raw = eedf_data[:, 1]
        thermal_fit_line = eedf_data[:, 2]
        print(f"-> EEDFデータ ({eedf_filepath}) を読み込みました。")
    except Exception as e:
        print(f"エラー: EEDFデータ ({eedf_filepath}) の読み込みに失敗: {e}")
        return

    try:
        # (Photon_Energy_keV, Thermal_Spec, NonThermal_Spec)
        spectrum_data = np.loadtxt(spectrum_filepath)
        plot_energy_bins = spectrum_data[:, 0]
        thermal_spec = spectrum_data[:, 1]
        non_thermal_spec = spectrum_data[:, 2]
        print(f"-> スペクトルデータ ({spectrum_filepath}) を読み込みました。")
    except Exception as e:
        print(f"エラー: スペクトルデータ ({spectrum_filepath}) の読み込みに失敗: {e}")
        return

    # --- プロット (2パネル) ---
    print("-> プロットを作成中...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Bremsstrahlung Analysis (Timestep: {timestep})", fontsize=16)

    # パネル1: EEDF (電子エネルギー分布関数)
    ax1.plot(E_keV_raw, eedf_raw, 'b-', label='EEDF (Data)')
    
    # フィットデータがゼロでない場合のみプロット
    if np.any(thermal_fit_line > 0):
        # T_e の値をファイル名などから復元するのは難しいため、ラベルは簡略化
        ax1.plot(E_keV_raw, thermal_fit_line, 'r--', 
                 label=f'Thermal Fit')
    
    ax1.set_ylabel('Electron Count (dN/dE)')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1) # ゼロを避ける
    ax1.legend()
    ax1.set_title('Electron Energy Distribution Function (EEDF)')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    # パネル2: 制動放射スペクトル
    # データがゼロでない場合のみプロット
    if np.any(thermal_spec > 0):
        ax2.plot(plot_energy_bins, thermal_spec, 'r--', label='Thermal Spectrum (from Fit)')
    if np.any(non_thermal_spec > 0):
        ax2.plot(plot_energy_bins, non_thermal_spec, 'g-', label='Non-Thermal Spectrum (from EEDF)')
    
    ax2.set_xlabel('Photon Energy (keV)')
    ax2.set_ylabel('Intensity (arb. units)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 適切なY軸範囲を設定
    valid_min_spec = np.nanmin(non_thermal_spec[non_thermal_spec > 1e-30])
    if np.isfinite(valid_min_spec):
        ax2.set_ylim(bottom=valid_min_spec * 0.1)
    
    ax2.set_xlim(plot_energy_bins[0], E_keV_raw[-1]) # EEDFの最大値まで
    ax2.legend()
    ax2.set_title('Bremsstrahlung Spectrum (Soft X-ray)')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

    # 保存
    output_filename = os.path.join(PLOT_DIR, f'bremsstrahlung_plot_{timestep}.png')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)

    print(f"--- プロットを {output_filename} に保存しました ---")

# =======================================================
# スクリプト実行
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python plot_bremsstrahlung.py [timestep1] [timestep2] ...")
        print("   または: python plot_bremsstrahlung.py [start] [end] [step]")
        print("例 1 (個別): python plot_bremsstrahlung.py 000500 001000")
        print("例 2 (範囲): python plot_bremsstrahlung.py 0 14000 500")
        sys.exit(1)
        
    # Matplotlibのフォント設定 (オプション)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    # 引数の処理
    if len(sys.argv) == 4:
        # 例 2 (範囲): python plot_bremsstrahlung.py 0 14000 500
        try:
            start_step = int(sys.argv[1])
            end_step   = int(sys.argv[2])
            step_size  = int(sys.argv[3])
            
            print(f"--- 範囲指定でプロット (Start: {start_step}, End: {end_step}, Step: {step_size}) ---")
            for step in range(start_step, end_step + step_size, step_size):
                timestep_str = f"{step:06d}"
                print(f"\n--- プロット中: {timestep_str} ---")
                plot_data(timestep_str)
                
        except ValueError:
            print("エラー: 範囲指定の引数は3つの整数である必要があります。")
            sys.exit(1)
            
    else:
        # 例 1 (個別): python plot_bremsstrahlung.py 000500 001000
        print(f"--- 個別指定でプロット ---")
        timesteps_to_plot = sys.argv[1:]
        for ts in timesteps_to_plot:
             # 念のため6桁ゼロ埋め
             try:
                 ts_int = int(ts)
                 timestep_str = f"{ts_int:06d}"
             except ValueError:
                 timestep_str = ts # 既に "000500" の場合
                 
             print(f"\n--- プロット中: {timestep_str} ---")
             plot_data(timestep_str)

    print("\n--- 全てのプロット処理が完了しました ---")