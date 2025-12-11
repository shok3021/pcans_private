import numpy as np
import os
import glob
import sys
import time

# --- Scipy をインポート ---
try:
    from scipy.io import FortranFile
except ImportError:
    print("エラー: 'scipy' ライブラリが見つかりません。")
    print("ターミナルで 'pip install scipy' を実行してください。")
    sys.exit(1)

# =======================================================
# Fortranバイナリ読み込み関数 (変更なし)
# =======================================================
def load_and_stitch_fortran_binary(pattern):
    """
    fio__output (Fortran) によって書かれたバイナリファイルを読み込み、
    電磁場データ (uf) を結合（スティッチ）します。
    """
    
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"エラー: ファイルが見つかりません: {pattern}")
        return None
        
    print(f"{len(file_list)} 個のファイルを読み込みます: {pattern}")
    
    all_headers = {}
    global_nx_full = 0
    global_ny_full = 0
    
    header_dtype = np.dtype([
        ('it', 'i4'), ('np', 'i4'),
        ('nxgs', 'i4'), ('nxge', 'i4'), ('nygs', 'i4'), ('nyge', 'i4'),
        ('nxs', 'i4'), ('nxe', 'i4'), ('nys', 'i4'), ('nye', 'i4'),
        ('nsp', 'i4'), ('nproc', 'i4'), ('bc', 'i4'),
        ('delt', 'f8'), ('delx', 'f8'), ('c', 'f8')
    ])

    print("  ... パス 1/2: ヘッダをスキャン中 ...")
    for f in file_list:
        try:
            ff = FortranFile(f, 'r')
            header_data = ff.read_record(dtype=header_dtype)[0]
            all_headers[f] = header_data
            
            # グローバルグリッドサイズを決定 (Ghostセルを含む)
            global_nx_full = header_data['nxge'] - header_data['nxgs'] + 1 + 2 
            global_ny_full = header_data['nyge'] - header_data['nygs'] + 1 + 2
            
            ff.close()
            
        except Exception as e:
            print(f"    エラー: {f} のヘッダ読み込みに失敗: {e}")
            return None

    if global_nx_full == 0 or global_ny_full == 0:
        print("エラー: グローバルグリッドサイズを決定できませんでした。")
        return None
        
    print(f"  -> グローバルグリッドサイズ (Ghost含む) を (NX={global_nx_full}, NY={global_ny_full}) と決定しました。")

    # フィールドは 6成分 (Bx, By, Bz, Ex, Ey, Ez)
    global_fields = np.zeros((6, global_ny_full, global_nx_full))
    
    print("  ... パス 2/2: データを読み込み・結合中 ...")
    
    for f, header in all_headers.items():
        try:
            ff = FortranFile(f, 'r')
            
            # ヘッダとパーティクル関連レコードをスキップ
            ff.read_record(dtype=header_dtype) 
            ff.read_ints('i4')   # Record 2: np2 (スキップ)
            ff.read_reals('f8')  # Record 3: q (スキップ)
            ff.read_reals('f8')  # Record 4: r (スキップ)

            # ローカルデータサイズを計算
            nx_local_written = header['nxe'] - header['nxs'] + 3
            ny_local_written = header['nye'] - header['nys'] + 3
            data_flat = ff.read_reals('f8')
            
            if data_flat.size != (6 * nx_local_written * ny_local_written):
                raise ValueError(f"Field (uf): データサイズ不一致。 期待値={6 * nx_local_written * ny_local_written}, 実際={data_flat.size}")
            
            # Fortranの列優先 (order='F') で読み込み、 (6, NX_local, NY_local) の形にする
            field_data_local = data_flat.reshape((6, nx_local_written, ny_local_written), order='F')
            
            # グローバル配列への貼り付け位置を計算 (Ghostセルを含むインデックス)
            g_start_x = header['nxs'] - header['nxgs']
            g_end_x   = g_start_x + nx_local_written
            g_start_y = header['nys'] - header['nygs']
            g_end_y   = g_start_y + ny_local_written

            # Python/NumPyの標準形式 (6, NY, NX) にするために転置 (0, 2, 1) を行う
            global_fields[:, g_start_y:g_end_y, g_start_x:g_end_x] = field_data_local.transpose(0, 2, 1)

            ff.close() 

        except Exception as e:
            print(f"    エラー: {f} のデータ読み込みまたは変形に失敗: {e}")
            return None

    print("  ... 全ファイルの結合が完了しました。")
    return global_fields, all_headers[file_list[0]]


# =======================================================
# データ抽出・保存関数
# =======================================================

def get_physical_region(global_fields, header):
    """Ghostセルを除いた物理領域を切り出す (規格化された値)"""
    nxgs, nxge = header['nxgs'], header['nxge']
    nygs, nyge = header['nygs'], header['nyge']
    
    # Fortranのインデックスは 1 から始まるが、NumPyは 0 から始まる
    # global_fields は (6, NY_full, NX_full)
    
    # ★★★ 修正が必要な箇所 ★★★
    # グリッド点数ではなく、セル数を取得する:
    NX_CELLS = header['nxge'] - header['nxgs'] + 1
    NY_CELLS = header['nyge'] - header['nygs'] + 1
    
    # Ghostセルは各辺に1つずつあり、それらがインデックス 0 と -1 に対応すると仮定
    phys_start_x = 1 # Ghostセルをスキップ
    # 物理領域の終了インデックスは、開始インデックス + セル数
    # しかし、読み込まれたデータが (640, 321) の形状で、これはグリッド点数 (NY=640, NX=321) であるため、
    # 物理セル (NY=639, NX=320) を抽出するには、各軸で終端を -1 する必要があります。
    phys_end_x   = phys_start_x + (NX_CELLS - 1) # 物理セル数 NX_CELLS-1 = 320
    phys_start_y = 1 
    phys_end_y   = phys_start_y + (NY_CELLS - 1) # 物理セル数 NY_CELLS-1 = 639
    
    # 読み込まれたデータが (NY_full, NX_full) = (642, 323) のサイズであると仮定すると、
    # 物理領域は [1:640, 1:321] のサイズ (640, 321) です。
    # 必要なのは (639, 320) のサイズです。

    # Fortranの流儀に従い、グリッド点数ではなく、セル数で切り出す:
    NX_CELLS_PHYS = (header['nxge'] - header['nxgs'] + 1) - 1 # 321 - 1 = 320
    NY_CELLS_PHYS = (header['nyge'] - header['nygs'] + 1) - 1 # 640 - 1 = 639
    
    # 切り出す NumPy のスライスは [1: NY_CELLS_PHYS + 1, 1: NX_CELLS_PHYS + 1]
    phys_start_x = 1
    phys_end_x   = phys_start_x + NX_CELLS_PHYS # 1 + 320 = 321
    phys_start_y = 1
    phys_end_y   = phys_start_y + NY_CELLS_PHYS # 1 + 639 = 640
    
    # (6, NY_phys, NX_phys) の配列を返す (サイズ 6, 639, 320)
    return global_fields[:, phys_start_y:phys_end_y, phys_start_x:phys_end_x]

def save_data_to_txt(data_2d, label, timestep, out_dir, filename):
    """
    2Dデータをテキストファイルに保存する。
    """
    output_file = os.path.join(out_dir, f'data_{timestep}_{filename}.txt')
    
    # データをCSV形式で保存 (区切り文字: カンマ)
    # numpy.savetxt はデータのみを書き出す
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    
    print(f"-> 規格化された {label} データを {output_file} に保存しました。")


# =======================================================
# メイン処理 (変更後)
# =======================================================
def main():
    # ★★★ 変更点 1: コマンドライン引数から 3 つの引数 (start, end, step) を取得 ★★★
    if len(sys.argv) < 4:
        print("使用方法: python data_extractor.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python data_extractor.py 000000 014000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: すべての引数 (開始、終了、間隔) は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")
    
    # 環境に応じてこのパスを調整してください
    data_dir = os.path.join('/data/shok/dat/')
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 出力先ディレクトリ: {OUTPUT_DIR} ---")
    
    # ★★★ 変更点 2: タイムステップを反復処理するループ ★★★
    # NumPyの arange を使ってステップのリストを生成
    # end_step も含むように + step_size
    for current_step in range(start_step, end_step + step_size, step_size):
        
        # タイムステップの文字列を '000500' のようにゼロ埋め6桁でフォーマット
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        file_pattern = os.path.join(data_dir, f'{timestep}_rank=*.dat') 
        
        # --- Fortranバイナリの読み込みと結合 ---
        start_time = time.time()
        result = load_and_stitch_fortran_binary(file_pattern)
        end_time = time.time()
        
        if result is None:
            print(f"警告: タイムステップ {timestep} のファイルが見つからないか、読み込みに失敗しました。スキップします。")
            continue # 次のステップへ
            
        global_fields, header = result
        print(f"  -> 処理時間 (読み込み/結合): {end_time - start_time:.2f} 秒")

        # --- 1. 物理領域を切り出す (規格化された値) ---
        phys_fields = get_physical_region(global_fields, header)
        
        # 各成分を切り出し
        bx_phys = phys_fields[0, :, :]
        by_phys = phys_fields[1, :, :]
        bz_phys = phys_fields[2, :, :]
        ex_phys = phys_fields[3, :, :]
        ey_phys = phys_fields[4, :, :]
        ez_phys = phys_fields[5, :, :]

        # --- 2. 各物理量をテキストファイルに保存 ---
        # 処理を関数内にまとめても良いが、ここでは元の構造を維持しつつループ内に配置
        
        save_data_to_txt(bx_phys, 'Magnetic Field (Bx)', timestep, OUTPUT_DIR, 'Bx')
        save_data_to_txt(by_phys, 'Magnetic Field (By)', timestep, OUTPUT_DIR, 'By')
        save_data_to_txt(bz_phys, 'Magnetic Field (Bz)', timestep, OUTPUT_DIR, 'Bz')
        save_data_to_txt(ex_phys, 'Electric Field (Ex)', timestep, OUTPUT_DIR, 'Ex')
        save_data_to_txt(ey_phys, 'Electric Field (Ey)', timestep, OUTPUT_DIR, 'Ey')
        save_data_to_txt(ez_phys, 'Electric Field (Ez)', timestep, OUTPUT_DIR, 'Ez')
        
        print(f"--- タイムステップ {timestep} の処理が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()