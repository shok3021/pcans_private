import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ヘルパー関数 (visual_fields.py / psd_extractor.py から移植)
# =======================================================

def load_simulation_parameters(param_filepath):
    """
    init_param.dat (key ====> value 形式) を読み込み、
    プロットに必要な全てのパラメータ (グリッド, DI, B0, VA0 など) を抽出する。
    """
    params = {}
    print(f"パラメータファイルを読み込み中: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "====>" in line:
                    parts = line.split("====>")
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    value_part = value_part.replace('x', ' ')
                    values = value_part.split()
                    
                    if not values: continue

                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0]) # 321
                        params['NY_GRID_POINTS'] = int(values[1]) # 640
                    
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
        
    # --- 必須パラメータのチェック ---
    required_keys = [
        'NX_GRID_POINTS', 'NY_GRID_POINTS', 'DELX', 'C_LIGHT', 'FPI',
        'MI', 'QI', 'FGI', 'VA0', 'DT'
    ]
    if not all(key in params for key in required_keys):
        print("★★ エラー: init_param.dat から必要なパラメータのいくつかを抽出できませんでした。")
        missing = [k for k in required_keys if k not in params]
        print(f"   不足しているキー: {missing}")
        sys.exit(1)
        
    # --- 派生パラメータの計算 ---
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
    
    # イオンスキンデプス (di = c / omega_pi)
    params['DI'] = params['C_LIGHT'] / params['FPI']
    
    # 規格化磁場 (B0 = fgi * mi * c / qi)
    params['B0'] = (params['FGI'] * params['MI'] * params['C_LIGHT']) / params['QI']
    
    print(f"  -> グリッド: {params['NX_PHYS']} x {params['NY_PHYS']}")
    print(f"  -> d_i = {params['DI']:.4f}")
    print(f"  -> B0 = {params['B0']:.4f}")
        
    return params

def load_2d_field_data(timestep, component, field_dir, ny_phys, nx_phys):
    """
    visual_fields.py と同じ電磁場データローダー
    (Bx, By などの .txt ファイルを読み込む)
    """
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(field_dir, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (ny_phys, nx_phys):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({ny_phys}, {nx_phys}) と異なります。")
            # 形状が違う場合は使わない
            return None
        return data 
    except Exception as e:
        # ファイルが存在しない場合なども含む
        print(f"情報: 磁場ファイル {filepath} が見つからないか、読み込めません ({e})。")
        return None

def create_coordinates(NX, NY, DELX, DI):
    """
    visual_fields.py と同じ X軸中心の座標グリッド ($x/d_i$, $y/d_i$) を作成
    """
    # X_MIN = -NX * DELX / 2.0
    # X_MAX = NX * DELX / 2.0
    x_phys = np.linspace(-NX * DELX / 2.0, NX * DELX / 2.0, NX)
    
    # Y_MIN = 0.0
    # Y_MAX = NY * DELX
    y_phys = np.linspace(0.0, NY * DELX, NY)
    
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    
    return np.meshgrid(x_norm, y_norm)

def load_xray_proxy_map(filepath, ny_phys, nx_phys):
    """
    bremsstrahlung_calc_2d_map.py が出力したTXTマップを読み込む
    """
    try:
        data = np.loadtxt(filepath)
        if data.shape != (ny_phys, nx_phys):
             print(f"エラー: {filepath} の形状 ({data.shape}) が ({ny_phys}, {nx_phys}) と一致しません。")
             return None
        print(f"-> X線プロキシマップ ({filepath}) を読み込みました。")
        return data
    except Exception as e:
        print(f"エラー: X線プロキシマップ ({filepath}) の読み込みに失敗: {e}")
        return None

# =======================================================
# メイン処理 (プロット)
# =======================================================
def plot_2d_map(timestep, params, field_data_dir, xray_data_dir, plot_dir):
    """
    指定されたタイムステップの2DマップTXTデータを読み込んでプロットする
    """
    
    NX = params['NX_PHYS']
    NY = params['NY_PHYS']
    
    # --- 1. X線プロキシマップの読み込み ---
    map_filepath = os.path.join(xray_data_dir, f'soft_xray_proxy_map_{timestep}.txt')
    Z_map = load_xray_proxy_map(map_filepath, NY, NX)
    
    if Z_map is None:
        # 読み込み失敗時はこのタイムステップをスキップ
        return

    # --- 2. 座標グリッドの作成 ---
    X_norm, Y_norm = create_coordinates(NX, NY, params['DELX'], params['DI'])

    # --- 3. (オプション) 磁力線の読み込み ---
    # Bx, By は規格化されていない .txt データ
    Bx_raw = load_2d_field_data(timestep, 'Bx', field_data_dir, NY, NX)
    By_raw = load_2d_field_data(timestep, 'By', field_data_dir, NY, NX)
    
    use_streamplot = (Bx_raw is not None) and (By_raw is not None)
    
    if use_streamplot:
        # visual_fields.py と同様に B0 で規格化
        Bx_norm = Bx_raw / params['B0']
        By_norm = By_raw / params['B0']
        print("-> 磁場データ Bx, By を読み込みました。磁力線を重ね描きします。")
    else:
        print("-> 磁場データが見つからないため、磁力線なしでプロットします。")
        # ストリームプロット用にダミー配列を作成
        Bx_norm = np.zeros((NY, NX))
        By_norm = np.zeros((NY, NX))


    # --- 4. プロット ---
    print("-> プロットを作成中...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # カラーマップ (強度なので 'hot' や 'inferno')
    cmap = 'hot' 
    
    # ゼロの粒子はログスケールで表示できないため、nan に設定
    Z_map_plot = np.where(Z_map > 0, Z_map, np.nan)
    
    try:
        # ログスケールでプロット (強度のダイナミクスが大きいため)
        from matplotlib.colors import LogNorm
        min_val = 1 # カウント 1 から
        max_val = np.nanmax(Z_map_plot)
        
        if not np.isfinite(max_val) or max_val < min_val:
            # データがまったくない場合
            norm = None
            cf = ax.contourf(X_norm, Y_norm, Z_map_plot, cmap=cmap, vmin=0, vmax=1)
            print("  -> プロットデータが空か、すべてゼロです。")
        else:
            norm = LogNorm(vmin=min_val, vmax=max_val)
            cf = ax.contourf(X_norm, Y_norm, Z_map_plot, 
                            levels=np.logspace(np.log10(min_val), np.log10(max_val), 50), 
                            cmap=cmap, norm=norm, extend='max')
        
        cbar = plt.colorbar(cf, ax=ax)
        # (ヘッダーから閾値を読むのは大変なので、ここではラベルを固定)
        cbar.set_label(f'High Energy Particle Count (Proxy for Soft X-ray)')

    except ValueError as e:
        # (max_val が min_val と同じ場合など)
        print(f"  -> ログプロットに失敗 ({e})。リニアスケールで試行します。")
        max_val = np.nanmax(Z_map)
        cf = ax.contourf(X_norm, Y_norm, Z_map, levels=np.linspace(0, max_val, 50), cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'High Energy Particle Count (Proxy for Soft X-ray)')

    # ストリームプロット (磁力線)
    if use_streamplot:
        stride_x = max(1, NX // 30) 
        stride_y = max(1, NY // 30) 
        ax.streamplot(X_norm[::stride_y, ::stride_x], Y_norm[::stride_y, ::stride_x], 
                      Bx_norm[::stride_y, ::stride_x], By_norm[::stride_y, ::stride_x], 
                      color='white', linewidth=0.5, density=1.0, 
                      arrowstyle='-', minlength=0.1, zorder=1)
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(f'Soft X-ray Proxy (High Energy $e^-$ Density) at Timestep {timestep}')
    ax.tick_params(direction='in', top=True, right=True)

    # 保存
    output_filename = os.path.join(plot_dir, f'soft_xray_map_{timestep}.png')
    fig.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)

    print(f"--- 2Dマッププロットを {output_filename} に保存しました ---")

# =======================================================
# スクリプト実行
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python plot_bremsstrahlung_2d_map.py [timestep1] [timestep2] ...")
        print("   または: python plot_bremsstrahlung_2d_map.py [start] [end] [step]")
        sys.exit(1)
        
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    # --- 1. ディレクトリ設定 ---
    # (Jupyterなどでの実行にも対応)
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    # ★ 共通の init_param.dat のパス
    PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

    # ★ (オプション) 磁場データ (visual_fields.py が使う)
    FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
    # ★ 入力: 2DマップTXTデータ
    XRAY_DATA_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_2dmap_txt')
    # ★ 出力: プロット画像 (.png)
    PLOT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots_2d')
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print(f"--- 共通設定 ---")
    print(f"パラメータファイル: {PARAM_FILE_PATH}")
    print(f"X線マップ入力: {XRAY_DATA_DIR}")
    print(f"磁場データ入力: {FIELD_DATA_DIR}")
    print(f"プロット出力: {PLOT_DIR}")

    # --- 2. 共通パラメータを一度だけ読み込む ---
    plot_params = load_simulation_parameters(PARAM_FILE_PATH)
    if plot_params is None:
        print("エラー: init_param.dat の読み込みに失敗したため、プロットを終了します。")
        sys.exit(1)

    # --- 3. 引数の処理 ---
    if len(sys.argv) == 4:
        # 範囲指定
        try:
            start_step = int(sys.argv[1])
            end_step   = int(sys.argv[2])
            step_size  = int(sys.argv[3])
            
            print(f"--- 範囲指定でプロット (Start: {start_step}, End: {end_step}, Step: {step_size}) ---")
            for step in range(start_step, end_step + step_size, step_size):
                timestep_str = f"{step:06d}"
                print(f"\n--- プロット中: {timestep_str} ---")
                plot_2d_map(timestep_str, plot_params, FIELD_DATA_DIR, XRAY_DATA_DIR, PLOT_DIR)
                
        except ValueError:
            print("エラー: 範囲指定の引数は3つの整数である必要があります。")
            sys.exit(1)
            
    else:
        # 個別指定
        print(f"--- 個別指定でプロット ---")
        timesteps_to_plot = sys.argv[1:]
        for ts in timesteps_to_plot:
             try:
                 ts_int = int(ts)
                 timestep_str = f"{ts_int:06d}"
             except ValueError:
                 timestep_str = ts # 既に "000500" の場合
                 
             print(f"\n--- プロット中: {timestep_str} ---")
             plot_2d_map(timestep_str, plot_params, FIELD_DATA_DIR, XRAY_DATA_DIR, PLOT_DIR)

    print("\n--- 全てのプロット処理が完了しました ---")