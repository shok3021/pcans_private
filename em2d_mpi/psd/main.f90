program main

  use boundary, only: boundary__particle
  use particle, only: particle__solv
  ! constモジュール (nx, ny, delx, nproc, np, c 等の定数)
  use const
  
  ! fioモジュールから、サブルーチンだけでなくデータ変数もインポートする
  ! ※ np は const と衝突するので import しない
  use fio, only: fio__input, fio__psd, &
                 up, uf, q, r, delt, np2

  implicit none

  logical           :: lflag=.true.
  integer           :: ndata, idata, irank
  character(len=64) :: ifile
  real(8)           :: x0, y0
  real(8)           :: phys_w, phys_h 
  character(len=64) :: xpos, ypos

  ! ローカル変数の宣言 (fioにもconstにもないもの)
  integer           :: nys, nye 

  ! ------------------------------------------------
  ! 1. 引数取得 (中心座標)
  ! ------------------------------------------------
  ndata = iargc()
  call getarg(1,xpos)
  call getarg(2,ypos)
  
  read(xpos,*) x0
  read(ypos,*) y0

  ! ------------------------------------------------
  ! 2. 出力範囲の決定
  ! ------------------------------------------------
  ! constモジュールの定数を使用
  phys_w = real(nx-1, 8) * delx  ! 1600 * 0.2 = 320.0
  phys_h = real(ny, 8)   * delx  ! 640  * 0.2 = 128.0

  ! nproc は const モジュールの parameter なので自動的に設定されています

  ! ------------------------------------------------
  ! 3. ループ処理
  ! ------------------------------------------------
  do idata=4,ndata,nproc
     do irank=0,nproc-1

        call getarg(idata+irank,ifile)
        write(*,'(a)')'reading.....  '//trim(ifile)

        ! ファイルを読み込む (fio内の up, uf, np2 等にデータが入る)
        call fio__input(nproc,ifile)

        ! nys, nye (Y方向の担当範囲) は fio__input で設定されない場合、
        ! ここで計算する必要がありますが、通常は fio モジュール内か
        ! input データから定まることが多いです。
        ! もしここで未定義エラーが出る場合、fio から nys, nye も import する必要があります。
        ! とりあえず nys=nygs, nye=nyge で初期化するか、
        ! fio__input が内部で nys/nye をセットしているなら import 追加が必要です。
        ! ここでは一般的な並列計算の範囲として仮置きします：
        nys = nygs
        nye = nyge
        ! ※もし fio モジュールに nys, nye があるなら、上の宣言を消して
        !   use fio, only: ..., nys, nye  を追加してください。

        call particle__solv(up,uf,c,q,r,0.5*delt,np,nxgs,nxge,nygs,nyge,nys,nye,nsp,np2)
        call boundary__particle(up,np,nys,nye,nxgs,nxge,nygs,nyge,nsp,np2,bc)
        
        call fio__psd(up, x0, y0, phys_w, phys_h, np, nys, nye, nsp, np2, it0, '/data/shok/psd/')

        ! fioモジュールの allocatable 変数を解放
        if (allocated(np2)) deallocate(np2)
        if (allocated(up))  deallocate(up)
     enddo
  enddo

end program main