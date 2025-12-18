program main

  use const, only : nx, ny
  use boundary
  use fio
  use particle

  implicit none

  logical           :: lflag=.true.
  integer           :: ndata, nproc, idata, irank
  ! character(len=64) :: dir  ! 不要になった場合は削除可能
  character(len=64) :: ifile
  real(8)           :: x0, y0, dx, dy
  character(len=64) :: xpos, ypos

  ndata = iargc()
  call getarg(1,xpos)
  call getarg(2,ypos)
  read(xpos,*)x0
  read(ypos,*)y0
  ! call getarg(3,dir) ! パスを固定するため、引数からの読み込みはコメントアウトまたは削除

  write(*,*)'No. of processes?'
  read(*,*)nproc

  dx = real(nx, 8) !sampling area in the x direction
  dy = real(ny, 8) !sampling area in the y direction


  do idata=4,ndata,nproc
     do irank=0,nproc-1

        call getarg(idata+irank,ifile)
        write(*,'(a)')'reading.....  '//trim(ifile)

        call fio__input(nproc,ifile)

        call particle__solv(up,uf,c,q,r,0.5*delt,np,nxgs,nxge,nygs,nyge,nys,nye,nsp,np2)
        call boundary__particle(up,np,nys,nye,nxgs,nxge,nygs,nyge,nsp,np2,bc)
        call fio__psd(up,x0,y0,dx*delx,dy*delx,np,nys,nye,nsp,np2,it0,'/data/shok/psd/')

        deallocate(np2)
        deallocate(up)
     enddo
  enddo

end program main