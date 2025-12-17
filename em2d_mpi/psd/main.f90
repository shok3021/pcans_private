program main

  use boundary, only: boundary__particle
  use particle, only: particle__solv
  ! constモジュールを「正」として全変数をインポート (nx, ny, delx, nproc, np, c, q...等)
  use const
  ! fioからはサブルーチンだけをインポート (np等の変数衝突を防ぐため)
  use fio, only: fio__input, fio__psd

  implicit none

  logical           :: lflag=.true.
  ! nproc は const から来るので、ここでの宣言(integer)からは削除
  integer           :: ndata, idata, irank
  character(len=64) :: ifile
  real(8)           :: x0, y0
  real(8)           :: phys_w, phys_h 
  character(len=64) :: xpos, ypos

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

  ! nproc = 160  <-- 削除 (constのparameterなので代入不可。自動的に160になります)

  ! ------------------------------------------------
  ! 3. ループ処理
  ! ------------------------------------------------
  do idata=4,ndata,nproc
     do irank=0,nproc-1

        call getarg(idata+irank,ifile)
        write(*,'(a)')'reading.....  '//trim(ifile)

        call fio__input(nproc,ifile)

        ! constの変数 (c, q, r, delt, np 等) がそのまま使われます
        call particle__solv(up,uf,c,q,r,0.5*delt,np,nxgs,nxge,nygs,nyge,nys,nye,nsp,np2)
        call boundary__particle(up,np,nys,nye,nxgs,nxge,nygs,nyge,nsp,np2,bc)
        
        ! 物理サイズ (phys_w, phys_h) を渡す
        call fio__psd(up, x0, y0, phys_w, phys_h, np, nys, nye, nsp, np2, it0, '/data/shok/psd/')

        deallocate(np2)
        deallocate(up)
     enddo
  enddo

end program main