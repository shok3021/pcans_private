import pandas as pd
import matplotlib.pyplot as plt
import os
import re # 正規表現モジュールをインポート

# --- 1. ファイルパスの指定 ---
base_dir = os.path.expanduser('/data/shok/dat/')
energy_file_path = os.path.join(base_dir, 'energy.dat')
param_file_path = os.path.join(base_dir, 'init_param.dat')

# --- 2. init_param.dat から Omega_ci (Fgi) を読み込む ---
omega_ci = None
try:
    with open(param_file_path, 'r') as f:
        for line in f:
            # 'Fpe, Fge, Fpi Fgi' で始まる行を検索
            if line.strip().startswith('Fpe, Fge, Fpi Fgi'):
                # 正規表現で行末の4つの浮動小数点数を探す
                match = re.search(r'([\d\.E\+\-]+)\s*$', line) # 最後の数値 (Fgi) を探す
                if match:
                    omega_ci = float(match.group(1))
                    print(f"'{param_file_path}' から Fgi (Omega_ci) = {omega_ci} を読み込みました。")
                    break
    if omega_ci is None:
        print(f"エラー: '{param_file_path}' で 'Fgi' の値が見つかりませんでした。")
        omega_ci = 4.94E-03 
        print(f"警告: Fgiを自動検出できませんでした。手動値 {omega_ci} を使用します。")

except FileNotFoundError:
    print(f"エラー: パラメータファイル '{param_file_path}' が見つかりません。")
    omega_ci = 4.94E-03 # フォールバックとして提示された値を使用
    print(f"警告: ファイルが見つからないため、既知の値 Omega_ci = {omega_ci} を使用します。")
except Exception as e:
    print(f"パラメータファイルの読み込み中にエラーが発生しました: {e}")
    omega_ci = 4.94E-03
    print(f"警告: エラーのため、既知の値 Omega_ci = {omega_ci} を使用します。")


# --- 3. energy.dat の読み込みとX軸の変換 ---
if omega_ci is not None:
    try:
        # データの読み込み
        data = pd.read_csv(energy_file_path, sep=r'\s+', header=None, engine='python')
        
        # 列に名前を付ける
        data.columns = ['Time', 'KE_ion', 'KE_electron', 'E_field', 'B_field', 'Total_Energy']

        # 横軸の変換
        data['Omega_ci_t'] = data['Time'] * omega_ci

        print(f"ファイル '{energy_file_path}' の読み込みと $\Omega_cit$ への変換に成功しました。")
        print("--- データ (先頭5行、Omega_ci_t列を追加) ---")
        print(data.head())
        print("------------------------------------------")

        # --- 4. グラフの描画 ---
        plt.figure(figsize=(12, 7))

        # X軸に 'Omega_ci_t' を使用
        x_axis = 'Omega_ci_t'
        
        # 各エネルギー成分をプロット
        plt.plot(data[x_axis], data['KE_ion'], label='Ion Kinetic Energy (KE_ion)')
        plt.plot(data[x_axis], data['KE_electron'], label='Electron Kinetic Energy (KE_e)')
        plt.plot(data[x_axis], data['E_field'], label='Electric Field Energy (U_E)')
        plt.plot(data[x_axis], data['B_field'], label='Magnetic Field Energy (U_B)')
        
        # 全エネルギー (Total_Energy) を点線でプロット
        plt.plot(data[x_axis], data['Total_Energy'], label='Total Energy (Conserved)', 
                 linestyle='--', linewidth=2.5, color='black')

        # --- 5. グラフの装飾 ---
        plt.title('Energy History ($\Omega_{ci} t$)', fontsize=16)
        plt.xlabel('Normalized Time ($\Omega_{ci} t$)', fontsize=12) 
        plt.ylabel('Energy (Arbitrary Units)', fontsize=12)
        
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # --- 6. グラフの保存 ---
        # スクリプトと同じディレクトリに保存
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_filename = 'final_plots/energy_history.png'
        output_path = os.path.join(script_dir, output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # 高解像度で余白を詰めて保存
        print(f"グラフを '{output_path}' に保存しました。")

        # --- 7. グラフの表示 (オプション) ---
        # 保存した後に画面に表示するかどうかは任意です。
        # コメントアウトすると表示されず、ファイル保存のみになります。
        # plt.show()

    except FileNotFoundError:
        print(f"エラー: エネルギーファイル '{energy_file_path}' が見つかりませんでした。")
    except Exception as e:
        print(f"データの読み込みまたは描画中に予期せぬエラーが発生しました: {e}")
else:
    print("Omega_ci が設定されていないため、グラフ描画をスキップしました。")