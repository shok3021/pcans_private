import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ★ ステップ1: init_param.dat からパラメータを読み込む
# (visual_fields.py と同じ関数群が必要)
# =======================================================

# --- (visual_fields.py から必要な関数をコピー) ---
# 1. load_simulation_parameters (init_param.dat のフルパーサー)
# 2. create_coordinates (X, Y グリッド作成)
# 3. load_2d_field_data (Bx, By の読み込み用)
#
# (簡略化のため、ここでは主要な関数を再定義・簡略化します)
# (実際には visual_fields.py からコピー＆ペーストするのが確実です)

def load_init_params_for_plotting(param_filepath):
    """
    プロットに必要なパラメータ (NX, NY, DELX, DI) のみを読み込む
    """
    params = {}
    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "====>" in line:
                    parts = line.split("====>")
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    values = value_part.split()
                    
                    if key_part.startswith('grid size'):
                        params['NX_GRID_POINTS'] = int(values[0])
                        params['NY_GRID_POINTS'] = int(values[1].replace('x',''))
                    elif key_part.startswith('dx, dt, c'):
                        params['DELX'] = float(values[0])
                        params['C_LIGHT'] = float(values[2])
                    elif key_part.startswith('Fpe, Fge, Fpi Fgi'):
                        params['FPI'] = float(values[2])
    except Exception as e:
        print(f"警告: init_param.dat の読み込みに失敗: {e}")
        return None

    if not all(k in params for k in ['NX_GRID_POINTS', 'NY_GRID_POINTS', 'DELX', 'C_LIGHT', 'FPI']):
        print("警告: 必要なパラメータが init_param.dat から見つかりません。")
        return None
        
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
    params['DI'] = params['C_LIGHT'] / params['FPI'] # イオンスキンデプス
    return params

def create_coordinates(NX, NY, DELX, DI):
    """
    visual_fields.py と同じ X軸中心の座標グリッド ($x/d_i$, $y/d_i$) を作成
    """
    x_phys = np.linspace(-NX * DELX / 2.0, NX * DELX / 2.0, NX)
    y_phys = np.linspace(0.0, NY * DELX, NY)
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    return np.meshgrid(x_norm, y_norm)

def load_2d_field_data_simple(filepath, NY, NX):
    """
    電磁場データ (Bx, By) を読み込む簡易ローダー
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape == (NY, NX):
            return data
        else:
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({NY}, {NX}) と異なります。")
            return None
    except Exception as e:
        print(f"警告: {filepath} の読み込み失敗: {e}")
        return None

# =======================================================
# メイン処理 (プロット)
# =======================================================
def plot_2d_map(timestep, params):
    """
    指定されたタイムステップの2DマップTXTデータを読み込んでプロットする
    """
    
    NX = params['NX_PHYS']
    NY = params['NY_PHYS']
    
    # (Jupyterなどでの実行にも対応)
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.abspath('.') 

    # ★ 入力: 2DマップTXTデータが保存されているディレクトリ
    DATA_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_data_2dmap_txt')
    # ★ (オプション) 電磁場データのディレクトリ
    FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
    # ★ 出力: プロット画像 (.png) を保存するディレクトリ
    PLOT_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots_2d')
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # --- 1. X線プロキシマップの読み込み ---
    map_filepath = os.path.join(DATA_DIR, f'soft_xray_proxy_map_{timestep}.txt')
    try:
        Z_map = np.loadtxt(map_filepath)
        if Z_map.shape != (NY, NX):
             print(f"エラー: {map_filepath} の形状 ({Z_map.shape}) が ({NY}, {NX}) と一致しません。")
             return
        print(f"-> X線プロキシマップ ({map_filepath}) を読み込みました。")
    except Exception as e:
        print(f"エラー: X線プロキシマップ ({map_filepath}) の読み込みに失敗: {e}")
        return

    # --- 2. 座標グリッドの作成 ---
    X_norm, Y_norm = create_coordinates(NX, NY, params['DELX'], params['DI'])

    # --- 3. (オプション) 磁力線の読み込み ---
    Bx_file = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_Bx.txt')
    By_file = os.path.join(FIELD_DATA_DIR, f'data_{timestep}_By.txt')
    
    Bx = load_2d_field_data_simple(Bx_file, NY, NX)
    By = load_2d_field_data_simple(By_file, NY, NX)
    
    use_streamplot = (Bx is not None) and (By is not None)
    if use_streamplot:
        print("-> 磁場データ Bx, By を読み込みました。磁力線を重ね描きします。")
    else:
        print("-> 磁場データが見つからないため、磁力線なしでプロットします。")
        # ストリームプロット用にダミー配列を作成
        Bx = np.zeros((NY, NX))
        By = np.zeros((NY, NX))


    # --- 4. プロット ---
    print("-> プロットを作成中...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # カラーマップ (強度なので 'viridis' や 'hot' などが適している)
    cmap = 'hot' 
    
    # ゼロの粒子はログスケールで表示できないため、最小値を設定
    Z_map_plot = np.where(Z_map > 0, Z_map, np.nan)
    
    try:
        # ログスケールでプロット (強度のダイナミクスが大きいため)
        from matplotlib.colors import LogNorm
        min_val = 1 
        max_val = np.nanmax(Z_map_plot)
        
        if not np.isfinite(max_val) or max_val < min_val:
            # データがまったくない場合
            norm = None
            print("  -> プロットデータが空です。")
        else:
            norm = LogNorm(vmin=min_val, vmax=max_val)
            
        cf = ax.contourf(X_norm, Y_norm, Z_map_plot, 
                        levels=np.logspace(np.log10(min_val), np.log10(max_val), 50), 
                        cmap=cmap, norm=norm, extend='max')
        
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'High Energy Particle Count (E >= {ENERGY_THRESHOLD_KEV} keV)')

    except Exception as e:
        print(f"  -> ログプロットに失敗 ({e})。リニアスケールで試行します。")
        cf = ax.contourf(X_norm, Y_norm, Z_map, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f'High Energy Particle Count (E >= {ENERGY_THRESHOLD_KEV} keV)')

    # ストリームプロット (磁力線)
    if use_streamplot:
        stride_x = max(1, NX // 30) 
        stride_y = max(1, NY // 30) 
        ax.streamplot(X_norm[::stride_y, ::stride_x], Y_norm[::stride_y, ::stride_x], 
                      Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                      color='white', linewidth=0.5, density=1.0, 
                      arrowstyle='-', minlength=0.1, zorder=1)
            
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(f'Soft X-ray Proxy (High Energy $e^-$ Density) at Timestep {timestep}')
    ax.tick_params(direction='in', top=True, right=True)

    # 保存
    output_filename = os.path.join(PLOT_DIR, f'soft_xray_map_{timestep}.png')
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
    
    # --- init_param.dat から共通パラメータを一度だけ読み込む ---
    PARAM_FILE_PATH = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 
    plot_params = load_init_params_for_plotting(PARAM_FILE_PATH)
    if plot_params is None:
        print("エラー: init_param.dat の読み込みに失敗したため、プロットを終了します。")
        sys.exit(1)

    # --- 引数の処理 ---
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
                plot_2d_map(timestep_str, plot_params)
                
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
                 timestep_str = ts
                 
             print(f"\n--- プロット中: {timestep_str} ---")
             plot_2d_map(timestep_str, plot_params)

    print("\n--- 全てのプロット処理が完了しました ---")