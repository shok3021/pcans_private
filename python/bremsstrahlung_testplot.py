import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
import glob

# =======================================================
# 1. シミュレーションパラメータの読み込み
# =======================================================
def load_simulation_parameters(param_filepath):
    params = {}
    print(f"Loading parameters from: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                if "=>" in line:
                    parts = line.split("=>")
                    key_part = parts[0].strip()
                    value_part = parts[1].strip().replace('x', ' ')
                    values = value_part.split()
                    
                    if not values: continue

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
    except FileNotFoundError:
        print(f"Error: Parameter file not found at {param_filepath}")
        # デフォルト値（読み込み失敗時用）
        return {'NX_PHYS': 320, 'NY_PHYS': 639, 'DELX': 1.0, 'DI': 100.0, 'B0': 1.0}

    # 物理グリッド数
    params['NX_PHYS'] = params['NX_GRID_POINTS'] - 1
    params['NY_PHYS'] = params['NY_GRID_POINTS'] - 1
    
    # 物理長 (d_i) と磁場 (B0) の計算
    if params.get('FPI', 0) > 0:
        params['DI'] = params['C_LIGHT'] / params['FPI']
    else:
        params['DI'] = 100.0 # Default fallback

    if params.get('QI', 0) != 0:
        params['B0'] = (params['FGI'] * params['MI'] * params['C_LIGHT']) / params['QI']
    else:
        params['B0'] = 1.0

    print(f"  -> Grid: {params['NX_PHYS']} x {params['NY_PHYS']}")
    print(f"  -> d_i: {params['DI']:.2f}")
    
    return params

# =======================================================
# 2. データ読み込みヘルパー
# =======================================================
def load_2d_map(filepath):
    try:
        data = np.loadtxt(filepath)
        # 1次元配列になってしまった場合の処置
        if data.ndim == 1 and data.size > 0:
             # 正方形に近い形状を推測するか、パラメータ情報が必要
             pass 
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_field_data(timestep, component, field_dir, ny, nx):
    """磁場データの読み込み (streamplot用)"""
    filename = f'data_{timestep}_{component}.txt'
    path = os.path.join(field_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.shape == (ny, nx):
            return data
    except:
        pass
    return None

# =======================================================
# 3. プロット実行関数
# =======================================================
def plot_intensity_map(txt_path, particle_type, bin_label, timestep, params, field_dir, output_dir):
    
    # --- データ読み込み ---
    intensity_map = load_2d_map(txt_path)
    if intensity_map is None: return

    NX = params['NX_PHYS']
    NY = params['NY_PHYS']
    
    # 形状チェック (読み込んだデータとパラメータが一致するか)
    if intensity_map.shape != (NY, NX):
        print(f"Warning: Shape mismatch. Data: {intensity_map.shape}, Params: ({NY}, {NX}). Skipping.")
        return

    # --- 座標作成 (d_i単位) ---
    x_axis = np.linspace(0, NX * params['DELX'], NX) / params['DI']
    y_axis = np.linspace(0, NY * params['DELX'], NY) / params['DI']
    XX, YY = np.meshgrid(x_axis, y_axis)

    # --- 磁場データ (あれば読み込み) ---
    Bx = load_field_data(timestep, 'Bx', field_dir, NY, NX)
    By = load_field_data(timestep, 'By', field_dir, NY, NX)

    # --- プロット設定 ---
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # カラーマップ: Thermal=Red, NonThermal=Blue
    cmap_name = 'Reds' if particle_type == 'Thermal' else 'Blues'
    
    # ログスケール表示の前処理 (0以下をNaNにしてマスク)
    data_plot = np.where(intensity_map > 1e-8, intensity_map, np.nan)
    
    # 何もデータがない(真っ黒)場合の処理
    if np.all(np.isnan(data_plot)):
        print(f"  -> Empty map (No signal). Generating blank plot.")
        ax.text(0.5, 0.5, 'No Signal', ha='center', va='center', transform=ax.transAxes)
        # 便宜上の空プロット
        norm = colors.Normalize(vmin=0, vmax=1)
        pcm = ax.pcolormesh(XX, YY, np.zeros_like(XX), cmap=cmap_name, norm=norm, shading='auto')
    else:
        # ダイナミックレンジの自動調整
        max_val = np.nanmax(data_plot)
        min_val = np.nanmin(data_plot)
        
        # あまりに範囲が狭い、あるいは小さすぎる場合は調整
        if min_val <= 0 or not np.isfinite(min_val): min_val = 1e-4
        if max_val <= min_val: max_val = min_val * 10.0
        
        # 下位側を少し切ってノイズを消す (percentile)
        robust_min = np.nanpercentile(data_plot, 1)
        if robust_min > min_val: min_val = robust_min

        norm = colors.LogNorm(vmin=min_val, vmax=max_val)
        
        # メインプロット
        pcm = ax.pcolormesh(XX, YY, data_plot, cmap=cmap_name, norm=norm, shading='auto')
        
        # カラーバー
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(f'{particle_type} Intensity [a.u.]')

    # --- 磁力線の重ね書き (オプション) ---
    if Bx is not None and By is not None:
        # 間引きして表示
        st = max(1, NX // 40)
        B_norm = np.sqrt(Bx**2 + By**2)
        # 磁場が弱すぎる場所は描かない等の処理も可能だが、ここではそのまま
        ax.streamplot(XX[::st, ::st], YY[::st, ::st], 
                      Bx[::st, ::st], By[::st, ::st], 
                      color='black', linewidth=0.5, arrowsize=0.6, density=1.0)

    # --- 装飾 ---
    ax.set_title(f'{particle_type} Bremsstrahlung ({bin_label})\nTime Step: {timestep}')
    ax.set_xlabel('$x / d_i$')
    ax.set_ylabel('$y / d_i$')
    ax.set_aspect('equal')
    
    # --- 保存 ---
    # 保存先: bremsstrahlung_plots_detailed/Thermal/100-200keV/plot_...
    save_dir = os.path.join(output_dir, particle_type, bin_label)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'plot_{particle_type}_{bin_label}_{timestep}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {save_path}")

# =======================================================
# 4. メインループ
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python bremsstrahlung_plotter_detailed.py [start] [end] [step]")
        sys.exit(1)

    start_step, end_step, step_size = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # --- パス設定 ---
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except: SCRIPT_DIR = os.path.abspath('.')

    # パラメータファイル
    PARAM_FILE = '/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat'
    
    # 入力ディレクトリ (さっき作ったTXTデータの場所)
    DATA_BASE_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_detailed_intensity')
    
    # 磁場データディレクトリ (streamplot用、もしあれば)
    FIELD_DIR = os.path.join(SCRIPT_DIR, 'extracted_data')
    
    # 出力ディレクトリ (画像の保存先)
    PLOT_BASE_DIR = os.path.join(SCRIPT_DIR, 'bremsstrahlung_plots_detailed')

    print(f"--- Bremsstrahlung Plotter (Detailed) ---")
    print(f"Input Data: {DATA_BASE_DIR}")
    print(f"Output Plots: {PLOT_BASE_DIR}")

    # パラメータ読み込み
    params = load_simulation_parameters(PARAM_FILE)

    # タイムステップループ
    for t_int in range(start_step, end_step + step_size, step_size):
        timestep = f"{t_int:06d}"
        print(f"\n=== Processing TS: {timestep} ===")
        
        # フォルダ探索: Thermal / NonThermal
        for p_type in ['Thermal', 'NonThermal']:
            type_dir = os.path.join(DATA_BASE_DIR, p_type)
            if not os.path.exists(type_dir): continue
            
            # その中のエネルギービンフォルダを探索
            # 構造: .../Thermal/001-100keV/*.txt
            bin_dirs = glob.glob(os.path.join(type_dir, '*'))
            
            for b_dir in bin_dirs:
                if not os.path.isdir(b_dir): continue
                
                bin_label = os.path.basename(b_dir)
                
                # 該当するタイムステップのTXTファイルを探す
                # ファイル名規則: intensity_{p_type}_{bin_label}_{timestep}.txt
                target_file = os.path.join(b_dir, f'intensity_{p_type}_{bin_label}_{timestep}.txt')
                
                if os.path.exists(target_file):
                    plot_intensity_map(
                        target_file, p_type, bin_label, timestep,
                        params, FIELD_DIR, PLOT_BASE_DIR
                    )

    print("\n--- All Done ---")

if __name__ == "__main__":
    main()