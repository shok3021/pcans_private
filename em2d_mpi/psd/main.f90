program main

  use boundary
  use fio
  use particle
  use const  ! ← constモジュールにある nx, ny, delx を正として使う

  implicit none

  logical           :: lflag=.true.
  integer           :: nproc, ndata, idata, irank
  character(len=64) :: ifile
  real(8)           :: x0, y0
  ! dx, dy 変数は混乱の元なので廃止、あるいは計算用として明確化
  real(8)           :: phys_w, phys_h 
  character(len=64) :: xpos, ypos

  ! ------------------------------------------------
  ! 1. 引数取得 (中心座標)
  ! ------------------------------------------------
  ndata = iargc()
  call getarg(1,xpos)
  call getarg(2,ypos)
  
  read(xpos,*) x0 ! Makefileからは 160 (物理座標) が来る想定
  read(ypos,*) y0 ! Makefileからは 64  (物理座標) が来る想定

  ! ------------------------------------------------
  ! 2. 出力範囲の決定 (ここをCleanにする)
  ! ------------------------------------------------
  ! constモジュールの nx, ny (全グリッド数) を信じる。
  ! fio__psd には「物理的な幅 (Physical Width/Height)」を渡す必要がある。
  
  phys_w = real(nx-1, 8) * delx  ! 1600 * 0.2 = 320.0
  phys_h = real(ny, 8)   * delx  ! 640  * 0.2 = 128.0

  ! ※ nx は 1600+1 と定義されている場合があるので、実セル数なら nx-1 を使うのが物理的に正しいです。
  !   (境界条件によりますが、1600セルなら nx-1 が安全圏です)

  ! プロセス数は固定 (constの値を使う)
  nproc = 160 

  ! ------------------------------------------------
  ! 3. ループ処理
  ! ------------------------------------------------
  do idata=4,ndata,nproc
     do irank=0,nproc-1

        call getarg(idata+irank,ifile)
        write(*,'(a)')'reading.....  '//trim(ifile)

        call fio__input(nproc,ifile)

        call particle__solv(up,uf,c,q,r,0.5*delt,np,nxgs,nxge,nygs,nyge,nys,nye,nsp,np2)
        call boundary__particle(up,np,nys,nye,nxgs,nxge,nygs,nyge,nsp,np2,bc)
        
        ! ここで計算済みの「物理サイズ」を直接渡す
        ! 第4,5引数は "Physical Size" を期待しているため、
        ! ここでさらに *delx をしてはいけない (phys_w は既に物理サイズ)
        
        call fio__psd(up, x0, y0, phys_w, phys_h, np, nys, nye, nsp, np2, it0, '/data/shok/psd/')

        deallocate(np2)
        deallocate(up)
     enddo
  enddo

end program main