import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =======================================================
# ★ ステップ1: init_param.dat 読み込み関数 (★ '=>' に修正)
# =======================================================
def load_simulation_parameters(param_filepath):
    """
    (Extractor に合わせて '=>' を使用)
    """
    params = {}
    print(f"パラメータファイルを読み込み中: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "=>" in line: # ★ '====>' から '=>' に変更
                    parts = line.split("=>") # ★ '====>' から '=>' に変更
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    value_part = value_part.replace('x', ' ')
                    values = value_part.split()
                    
                    if not values:
                        continue

                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0])
                        params['NY_GRID_POINTS'] = int(values[1])
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])
                        params['DT'] = float(values[1])
                        params['C_LIGHT'] = float(values[2])
                    elif key_part.startswith('Mi, Me'):
                        params['MI'] = float(values[0])
                    elif key_part.startswith('Qi, Qe'):
                        params['QI'] = float(values[0])
                    elif key_part.startswith('Fpe, Fge, Fpi Fgi'):
                        params['FPI'] = float(values[2])
                        params['FGI'] = float(values[3])
                    elif key_part.startswith('Va, Vi, Ve'):
                        params['VA0'] = float(values[0])
    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        sys.exit(1)
        
    required_keys = [
        'NX_GRID_POINTS', 'NY_GRID_POINTS', 'DELX', 'C_LIGHT', 'FPI',
        'MI', 'QI', 'FGI', 'VA0', 'DT'
    ]
    if not all(key in params for key in required_keys):
        print("★★ エラー: init_param.dat から必要なパラメータのいくつかを抽出できませんでした。")
        missing = [k for k in required_keys if k not in params]
        print(f"   不足しているキー: {missing}")
        sys.exit(1)
        
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
    params['DI'] = params['C_LIGHT'] / params['FPI']
    params['B0'] = (params['FGI'] * params['MI'] * params['C_LIGHT']) / params['QI']
    
    print(f"  -> グリッド: {params['NX_PHYS']} x {params['NY_PHYS']}")
    print(f"  -> d_i = {params['DI']:.4f}")
    print(f"  -> B0 = {params['B0']:.4f}")
        
    return params

# =======================================================
# ★ ステップ2: ヘルパー関数 (★ 座標系を修正)
# =======================================================

def load_2d_field_data(timestep, component, field_dir, ny_phys, nx_phys):
    # (変更なし)
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(field_dir, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (ny_phys, nx_phys):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({ny_phys}, {nx_phys}) と異なります。")
            return None
        return data 
    except Exception as e:
        print(f"情報: 磁場ファイル {filepath} が見つからないか、読み込めません ({e})。")
        return None

def create_coordinates(NX, NY, DELX, DI):
    """
    ★ X軸の座標系を [0, 320] に修正
    """
    # ★ X_MIN = 0.0, X_MAX = NX * DELX
    x_phys = np.linspace(0.0, NX * DELX, NX) 
    
    # Y_MIN = 0.0, Y_MAX = NY * DELX
    y_phys = np.linspace(0.0, NY * DELX, NY)
    
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    
    return np.meshgrid(x_norm, y_norm)

def load_xray_proxy_map(filepath, ny_phys, nx_phys):
    # (変更なし)
    try:
        data = np.loadtxt(filepath)
        if data.shape != (ny_phys, nx_phys):
             print(f"エラー: {filepath} の形状 ({data.shape}) が ({ny_phys}, {nx_phys}) と一致しません。")
             return None
        print(f"-> X線プロキシマップ ({os.path.basename(filepath)}) を読み込みました。")
        return data
    except Exception as e:
        print(f"エラー: X線プロキシマップ ({filepath}) の読み込みに失敗: {e}")
        return None

# =======================================================
# ★ ステップ3: メイン処理 (プロット) (変更なし)
# =======================================================
def plot_2d_map(timestep, energy_bin_label, params, field_data_dir, xray_base_dir, plot_base_dir):
    # (この関数内のロジックは、座標グリッド(X_norm, Y_norm)が
    #  create_coordinates から渡されるため、変更不要)
    
    NX = params['NX_PHYS']
    NY = params['NY_PHYS']
    
    xray_data_dir = os.path.join(xray_base_dir, energy_bin_label)
    map_filepath = os.path.join(xray_data_dir, f'xray_proxy_{timestep}_{energy_bin_label}.txt')
    
    Z_map = load_xray_proxy_map(map_filepath, NY, NX)
    
    if Z_map is None:
        print(f"警告: マップファイル {map_filepath} が見つからないか、読み込めません。スキップします。")
        return

    # ★ ここで渡される X_norm が [0, 320/DI] の範囲になる
    X_norm, Y_norm = create_coordinates(NX, NY, params['DELX'], params['DI'])

    Bx_raw = load_2d_field_data(timestep, 'Bx', field_data_dir, NY, NX)
    By_raw = load_2d_field_data(timestep, 'By', field_data_dir, NY, NX)
    use_streamplot = (Bx_raw is not None) and (By_raw is not None)
    
    if use_streamplot:
        Bx_norm = Bx_raw / params['B0']
        By_norm = By_raw / params['B0']
        print("-> 磁場データ Bx, By を読み込みました。磁力線を重ね描きします。")
    else:
        print("-> 磁場データが見つからないため、磁力線なしでプロットします。")
        Bx_norm = np.zeros((NY, NX))
        By_norm = np.zeros((NY, NX))

    print(f"-> プロットを作成中 (Bin: {energy_bin_label})...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = 'hot' 
    Z_map_plot = np.where(Z_map > 0, Z_map, np.nan)
    
    try:
        from matplotlib.colors import LogNorm
        min_val = 1 
        max_val = np.nanmax(Z_map_plot)
        
        if not np.isfinite(max_val) or max_val < min_val:
            norm = None
            cf = ax.contourf(X_norm, Y_norm, Z_map_plot, cmap=cmap, vmin=0, vmax=1)
            print("  -> プロットデータが空か、すべてゼロです。")
        else:
            norm = LogNorm(vmin=min_val, vmax=max_val)
            cf = ax.contourf(X_norm, Y_norm, Z_map_plot, 
                            levels=np.logspace(np.log10(min_val), np.log10(max_val), 50), 
                            cmap=cmap, norm=norm, extend='max')
        
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'High Energy Particle Count')
    except ValueError as e:
        print(f"  -> ログプロットに失敗 ({e})。リニアスケールで試行します。")
        max_val = np.nanmax(Z_map)
        cf = ax.contourf(X_norm, Y_norm, Z_map, levels=np.linspace(0, max_val, 50), cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'High Energy Particle Count')

    if use_streamplot:
        stride_x = max(1, NX // 30) 
        stride_y = max(1, NY // 30) 
        ax.streamplot(X_norm[::stride_y, ::stride_x], Y_norm[::stride_y, ::stride_x], 
                      Bx_norm[::stride_y, ::stride_x], By_norm[::stride_y, ::stride_x], 
                      color='white', linewidth=0.5, density=1.0, 
                      arrowstyle='-', minlength=0.1, zorder=1)
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(f'Soft X-ray Proxy (Energy Bin: {energy_bin_label}) at Timestep {timestep}')
    ax.tick_params(direction='in', top=True, right=True)

    output_plot_dir = os.path.join(plot_base_dir, energy_bin_label)
    os.makedirs(output_plot_dir, exist_ok=True)
    
    output_filename = os.path.join(output_plot_dir, f'soft_xray_map_{timestep}_{energy_bin_label}.png')
    fig.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)

    print(f"--- 2Dマッププロットを {output_filename} に保存しました ---")

# =======================================================
# ★ ステップ4: スクリプト実行 (★ エネルギービン名更新)
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python bremsstrahlung_plot_binned.py [timestep1] [timestep2] ...")
        print("   または: python bremsstrahlung_plot_binned.py [start] [end] [step]")
        sys.exit(1)
        
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    # --- 1. ディレクトリ設定 ---
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 
    FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
    XRAY_BASE_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_binned_txt')
    PLOT_BASE_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots_binned')
    os.makedirs(PLOT_BASE_DIR, exist_ok=True)
    
    print(f"--- 共通設定 ---")
    print(f"パラメータファイル: {PARAM_FILE_PATH}")
    print(f"X線マップ入力 (ベース): {XRAY_BASE_DIR}")
    print(f"磁場データ入力: {FIELD_DATA_DIR} (存在すれば使用)")
    print(f"プロット出力 (ベース): {PLOT_BASE_DIR}")

    # --- 2. 共通パラメータを一度だけ読み込む ---
    plot_params = load_simulation_parameters(PARAM_FILE_PATH)
    if plot_params is None:
        print("エラー: init_param.dat の読み込みに失敗したため、プロットを終了します。")
        sys.exit(1)

    # ★★★ どのエネルギービンをプロット対象とするか？ ★★★
    # (Extractor の新しいビン名に更新)
    ENERGY_BINS_TO_PLOT = [
        '01keV_05keV',
        '05keV_10keV',
        '10keV_15keV',
        '15keV_20keV',
        '20keV_25keV',
        '25keV_30keV',
        # (以下、プロットしたいビンを追加)
        '30keV_over'
    ]
    print(f"--- プロット対象エネルギービン: {ENERGY_BINS_TO_PLOT} ---")

    # --- 3. 引数の処理 ---
    timesteps_to_process = []
    if len(sys.argv) == 4:
        try:
            start_step = int(sys.argv[1])
            end_step   = int(sys.argv[2])
            step_size  = int(sys.argv[3])
            print(f"--- 範囲指定 (Start: {start_step}, End: {end_step}, Step: {step_size}) ---")
            timesteps_to_process = list(range(start_step, end_step + step_size, step_size))
        except ValueError:
            print("エラー: 範囲指定の引数は3つの整数である必要があります。")
            sys.exit(1)
    else:
        print(f"--- 個別指定 ---")
        for ts in sys.argv[1:]:
             try:
                 ts_int = int(ts)
                 timesteps_to_process.append(ts_int)
             except ValueError:
                 print(f"警告: {ts} は整数に変換できません。スキップします。")
                 
    # --- 4. プロット実行 (二重ループ) ---
    for step_int in timesteps_to_process:
        timestep_str = f"{step_int:06d}"
        print(f"\n=========================================")
        print(f"--- プロット中: タイムステップ {timestep_str} ---")
        
        for energy_bin_label in ENERGY_BINS_TO_PLOT:
            xray_data_dir_check = os.path.join(XRAY_BASE_DIR, energy_bin_label)
            if not os.path.isdir(xray_data_dir_check):
                print(f"  -> スキップ: ビン '{energy_bin_label}' の入力ディレクトリが見つかりません: {xray_data_dir_check}")
                continue
                
            print(f"  -> エネルギービン: {energy_bin_label}")
            plot_2d_map(
                timestep_str, 
                energy_bin_label, 
                plot_params, 
                FIELD_DATA_DIR, 
                XRAY_BASE_DIR, 
                PLOT_BASE_DIR
            )

    print("\n--- 全てのプロット処理が完了しました ---")