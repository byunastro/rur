MODULE io_dice

    REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: pos, vel
    REAL(KIND=4), DIMENSION(:), ALLOCATABLE :: mm
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: id
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: type
    !! HEAD VARIABLES
    TYPE htype
    	INTEGER*4, DIMENSION(6) :: npart
    	REAL*8, DIMENSION(6) :: mass
    	REAL*8 :: time
    	REAL*8 :: redshift
    	INTEGER*4 :: flag_sfr
    	INTEGER*4 :: flag_feedback
    	INTEGER*4, DIMENSION(6) :: nparttotal
    	INTEGER*4 :: flag_cooling
    	INTEGER*4 :: numfiles
    	REAL*8 :: boxsize
    	REAL*8 :: omega0
    	REAL*8 :: omegalambda
    	REAL*8 :: hubbleparam
    	INTEGER*4 :: flag_stellarage
    	INTEGER*4 :: flag_metals
    	INTEGER*4, DIMENSION(6)  :: totalhighword
    	INTEGER*4 :: flag_entropy_instead_u
    	INTEGER*4 :: flag_doubleprecision
    	INTEGER*4 :: flag_ic_info
    	REAL*4 :: lpt_scalingfactor
    	CHARACTER, DIMENSION(48) :: unused
    END TYPE htype

    !! GET snapshots basic info
    INTEGER(KIND=4) ncpu, nsnap, npart_tot, ncell_tot, ndim, ngridmax
    CHARACTER(LEN=1000) repo

    !! GET AMR & GRAV
    REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: g_pos, g_vel, g_force
    REAL(KIND=4), DIMENSION(:), ALLOCATABLE :: g_mm, g_phi
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: g_id, g_type 
CONTAINS
!#####################################################################
    SUBROUTINE read_gadget(fname)
!#####################################################################
    IMPLICIT NONE
    CHARACTER(1000), INTENT(IN) :: fname

    !! LOCAL VARIABLES
    logical :: ok
    INTEGER :: jump_blck, stat, dummy_int, blck_size, np, i0, i1, i
    INTEGER :: head_blck,pos_blck,vel_blck,id_blck,mass_blck,u_blck,metal_blck,age_blck
    CHARACTER(LEN=4) :: blck_name
    character(len=4)::ic_head_name  = 'HEAD'
    character(len=4)::ic_pos_name   = 'POS '
    character(len=4)::ic_vel_name   = 'VEL '
    character(len=4)::ic_id_name    = 'ID  '
    character(len=4)::ic_mass_name  = 'MASS'
    character(len=4)::ic_u_name     = 'U   '
    character(len=4)::ic_metal_name = 'Z   '
    character(len=4)::ic_age_name   = 'AGE '


    TYPE(htype) :: header
    INQUIRE(file=fname, exist=ok)
    if(.not.ok) then 
        write(*,*) 'No file '//fname
        stop
    end if
    OPEN(unit=10, file=TRIM(fname), status='old', form='unformatted', action='read', access='stream')
    head_blck  = -1
    pos_blck   = -1
    vel_blck   = -1
    id_blck    = -1
    mass_blck  = -1
    jump_blck = 1
    do while(.true.)
    !! READ HEADER
        read(10,POS=jump_blck,iostat=stat) dummy_int
        if(stat /= 0) exit
        read(10,POS=jump_blck+sizeof(dummy_int),iostat=stat) blck_name
        if(stat /= 0) exit
        read(10,POS=jump_blck+sizeof(dummy_int)+sizeof(blck_name),iostat=stat) dummy_int
        if(stat /= 0) exit
        read(10,POS=jump_blck+2*sizeof(dummy_int)+sizeof(blck_name),iostat=stat) dummy_int
        if(stat /= 0) exit
        read(10,POS=jump_blck+3*sizeof(dummy_int)+sizeof(blck_name),iostat=stat) blck_size
        if(stat /= 0) exit

        if(blck_name .eq. ic_head_name) then
            head_blck  = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_pos_name) then
            pos_blck   = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_vel_name) then
            vel_blck  = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_id_name) then
            id_blck    = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_mass_name) then
            mass_blck  = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_u_name) then
            u_blck     = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_metal_name) then
            metal_blck = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        if(blck_name .eq. ic_age_name) then
            age_blck   = int(jump_blck+sizeof(blck_name)+4*sizeof(dummy_int))
        endif
        jump_blck = int(jump_blck+blck_size+sizeof(blck_name)+5*sizeof(dummy_int))
    ENDDO

    READ(10,POS=head_blck) header%npart,header%mass,header%time,header%redshift, &
         header%flag_sfr,header%flag_feedback,header%nparttotal, &
         header%flag_cooling,header%numfiles,header%boxsize, &
         header%omega0,header%omegalambda,header%hubbleparam, &
         header%flag_stellarage,header%flag_metals,header%totalhighword, &
         header%flag_entropy_instead_u, header%flag_doubleprecision, &
         header%flag_ic_info, header%lpt_scalingfactor

    !! READ PART
    np = sum(header%npart)

    CALL read_gadget_allocate(np)

    READ(10,POS=pos_blck) pos
    READ(10,POS=vel_blck) vel
    READ(10,POS=id_blck) id
    READ(10,POS=mass_blck) mm
    CLOSE(10)

    CLOSE(10)

    i0 = 1
    i1 = 1
    DO i=1, SIZE(header%npart)
    	IF(header%npart(i) .EQ. 0) CYCLE
    	i1 = i0 + header%npart(i)-1
    	type(i0:i1) = i-1

    	i0 = i1 + 1
    ENDDO

    END SUBROUTINE read_gadget

!#####################################################################
    SUBROUTINE read_gadget_allocate(np)
!#####################################################################
    IMPLICIT NONE
    INTEGER(KIND=4) np

    IF(ALLOCATED(pos)) DEALLOCATE(pos)
    IF(ALLOCATED(vel)) DEALLOCATE(vel)
    IF(ALLOCATED(id)) DEALLOCATE(id)
    IF(ALLOCATED(mm)) DEALLOCATE(mm)
    IF(ALLOCATED(type)) DEALLOCATE(type)

    ALLOCATE(pos(1:3,1:np))
    ALLOCATE(vel(1:3,1:np))
    ALLOCATE(id(1:np))
    ALLOCATE(mm(1:np))
    ALLOCATE(type(1:np))
    END SUBROUTINE
!!!!!
    SUBROUTINE read_gadget_deallocate()
    IMPLICIT NONE
    IF(ALLOCATED(pos)) DEALLOCATE(pos)
    IF(ALLOCATED(vel)) DEALLOCATE(vel)
    IF(ALLOCATED(id)) DEALLOCATE(id)
    IF(ALLOCATED(mm)) DEALLOCATE(mm)
    IF(ALLOCATED(type)) DEALLOCATE(type)
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_part()
!#####################################################################
    IMPLICIT NONE
    !!----- LOCAL
    INTEGER(KIND=4) i, j, k, n, npdum
    CHARACTER*(1000) fdum, domnum, snum

    REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: tmp_dbl
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: tmp_int

    CALL get_part_tot()

    CALL get_part_allocate()

    WRITE(snum, '(I5.5)') nsnap

    j = 1
    DO i=1, ncpu
      WRITE(domnum, '(I5.5)') i
      fdum = TRIM(repo)//'/output_'//TRIM(snum)//'/part_'//TRIM(snum)//'.out'//TRIM(domnum)

      OPEN(UNIT=10, FILE=fdum, FORM='unformatted', STATUS='OLD')
      READ(10); READ(10); READ(10) npdum; READ(10)
      READ(10); READ(10); READ(10); READ(10);

      k = j + npdum-1

      ALLOCATE(tmp_dbl(1:npdum))
      ALLOCATE(tmp_int(1:npdum))

      !x, y, z
      DO n=1, 3
        READ(10) tmp_dbl
        g_pos(j:k,n) = tmp_dbl
      ENDDO

      !vx, vy, vz
      DO n=1, 3
        READ(10) tmp_dbl
        g_vel(j:k,n) = tmp_dbl
      ENDDO

      !mass
      READ(10) tmp_dbl
      g_mm(j:k) = tmp_dbl

      !ID
      READ(10) tmp_int
      g_id(j:k) = tmp_int

      !TAG (TODO)
         

      DEALLOCATE(tmp_dbl)
      DEALLOCATE(tmp_int)
      j = k+1
    ENDDO
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_part_allocate()
!#####################################################################
    IMPLICIT NONE
        IF(ALLOCATED(g_pos)) DEALLOCATE(g_pos)
        IF(ALLOCATED(g_vel)) DEALLOCATE(g_vel)
        IF(ALLOCATED(g_id)) DEALLOCATE(g_id)
        IF(ALLOCATED(g_mm)) DEALLOCATE(g_mm)
        IF(ALLOCATED(g_type)) DEALLOCATE(g_type)


        ALLOCATE(g_pos(1:npart_tot,3))
        ALLOCATE(g_vel(1:npart_tot,3))
        ALLOCATE(g_id(1:npart_tot))
        ALLOCATE(g_mm(1:npart_tot))
        ALLOCATE(g_type(1:npart_tot))
    END SUBROUTINE

    SUBROUTINE get_part_deallocate()
    IMPLICIT NONE
        IF(ALLOCATED(g_pos)) DEALLOCATE(g_pos)
        IF(ALLOCATED(g_vel)) DEALLOCATE(g_vel)
        IF(ALLOCATED(g_id)) DEALLOCATE(g_id)
        IF(ALLOCATED(g_mm)) DEALLOCATE(g_mm)
        IF(ALLOCATED(g_type)) DEALLOCATE(g_type)
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_part_tot()
!#####################################################################
    IMPLICIT NONE

    !!----- LOCAL
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: parts
    INTEGER(KIND=4) i, j, k, np

    CHARACTER*(1000) fdum, domnum, snum

    ALLOCATE(parts(1:ncpu))
    parts = 0
    WRITE(snum, '(I5.5)') nsnap
    DO i=1, ncpu
      WRITE(domnum, '(I5.5)') i
      fdum = TRIM(repo)//'/output_'//TRIM(snum)//'/part_'//TRIM(snum)//'.out'//TRIM(domnum)

      OPEN(UNIT=10, FILE=fdum, FORM='unformatted', STATUS='OLD')
      READ(10); READ(10); READ(10) np
      parts(i) = np
    ENDDO

    npart_tot = SUM(parts)
    DEALLOCATE(parts)
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_grav_allocate()
!#####################################################################
    IMPLICIT NONE
        IF(ALLOCATED(g_phi)) DEALLOCATE(g_phi)
        IF(ALLOCATED(g_force)) DEALLOCATE(g_force)

        ALLOCATE(g_phi(1:ncell_tot))
        ALLOCATE(g_force(1:ncell_tot,3))
    END SUBROUTINE

    SUBROUTINE get_grav_deallocate()
    IMPLICIT NONE
        IF(ALLOCATED(g_phi)) DEALLOCATE(g_phi)
        IF(ALLOCATED(g_force)) DEALLOCATE(g_force)
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_grav_tot()
!#####################################################################
    IMPLICIT NONE

    !!----- LOCAL
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: ncell
    INTEGER(KIND=4) i, grav_n,amr_n, cpunum, nboundary, nlevelmax, ncoarse, nx,ny,nz
    INTEGER(KIND=4) twotondim
    logical :: ok
    CHARACTER*(1000) fdum_grav,fdum_amr, domnum, snum
    twotondim = 2**ndim

    ALLOCATE(ncell(1:ncpu))
    ncell = 0
    WRITE(snum, '(I5.5)') nsnap
    DO i=1, ncpu
        WRITE(domnum, '(I5.5)') i
        fdum_grav = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/grav_'//TRIM(snum)//'.out'//TRIM(domnum)
        fdum_amr = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/amr_'//TRIM(snum)//'.out'//TRIM(domnum)
        inquire(file=fdum_grav, exist=ok)
        if ( .not. ok ) then
            print *,'File not found in repo: '//fdum_grav
            stop
        endif
        OPEN(UNIT=amr_n, FILE=fdum_amr, FORM='unformatted', STATUS='OLD')
        read(amr_n); read(amr_n); read(amr_n) nx,ny,nz
        close(amr_n)
        ncoarse = nx*ny*nz
        OPEN(UNIT=grav_n, FILE=fdum_grav, FORM='unformatted', STATUS='OLD')
        read(grav_n)cpunum; read(grav_n); read(grav_n) nlevelmax
        read(grav_n) nboundary
        close(grav_n)
        ncell(i) = ncoarse+twotondim*ngridmax
    ENDDO
    ncell_tot = SUM(ncell)
    DEALLOCATE(ncell)
    END SUBROUTINE

!#####################################################################
    SUBROUTINE get_grav()
!#####################################################################
    IMPLICIT NONE

    !!----- LOCAL
    INTEGER(KIND=4) i,j, grav_n,amr_n, cpunum, nboundary,nboundary2, nlevelmax,nlevelmax2, ncoarse, nx,ny,nz
    INTEGER(KIND=4) twotondim, ncache,iskip,igrid,ilevel,ind,ivar,ibound,istart
    integer,allocatable,dimension(:)::ind_grid, xdp
    integer ,allocatable,dimension(:)  ::next
    integer,allocatable,dimension(:,:) :: headl, headb
    CHARACTER*(1000) fdum_grav,fdum_amr, domnum, snum
    logical :: ok
    twotondim = 2**ndim

    CALL get_grav_tot()

    CALL get_grav_allocate()

    WRITE(snum, '(I5.5)') nsnap
    allocate(next  (1:ngridmax))
    do i=1,ngridmax-1
        next(i)=i+1
    end do
    next(ngridmax) = 0
    DO j=1, ncpu
        WRITE(domnum, '(I5.5)') j
        fdum_grav = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/grav_'//TRIM(snum)//'.out'//TRIM(domnum)
        fdum_amr = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/amr_'//TRIM(snum)//'.out'//TRIM(domnum)
        inquire(file=fdum_grav, exist=ok)
        if ( .not. ok ) then
            print *,'File not found in repo: '//fdum_grav
            stop
        end if
        OPEN(UNIT=amr_n, FILE=fdum_amr, FORM='unformatted', STATUS='OLD')
        read(amr_n); read(amr_n); read(amr_n) nx,ny,nz
        read(amr_n) nlevelmax2
        read(amr_n); read(amr_n) nboundary2
        allocate(headl(1:ncpu+nboundary2, 1:nlevelmax2))
        allocate(headb(1:100, 1:nlevelmax2))
        close(amr_n)
        OPEN(UNIT=amr_n, FILE=fdum_amr, FORM='unformatted', STATUS='OLD')
        call skip_read(amr_n, 21)
        read(amr_n) headl(1:ncpu, 1:nlevelmax2)
        call skip_read(amr_n, 3)
        read(amr_n) headb(1:nboundary2,1:nlevelmax2)
        close(amr_n)
        
        ncoarse = nx*ny*nz
        OPEN(UNIT=grav_n, FILE=fdum_grav, FORM='unformatted', STATUS='OLD')
        read(grav_n)cpunum; read(grav_n); read(grav_n) nlevelmax
        read(grav_n) nboundary
        do ilevel=1,nlevelmax
            do ibound=1,nboundary+cpunum
                read(grav_n); read(grav_n) ncache
                if(ibound<=cpunum)then
                    istart=headl(ibound,ilevel)
                else
                    istart=headb(ibound-cpunum,ilevel)
                end if

                if(ncache>0)then
                    allocate(ind_grid(1:ncache), xdp(1:ncache))
                    ! Loop over level grids
                    igrid=istart
                    do i=1,ncache
                        ind_grid(i)=igrid
                        igrid=next(igrid)
                    end do
                    ! Loop over cells

                    do ind=1,twotondim
                        iskip=ncoarse+(ind-1)*ngridmax
                        ! Read potential 
                        read(grav_n)xdp
                        do i=1,ncache
                           g_phi(ind_grid(i)+iskip)=xdp(i)
                        end do
                        ! Read force
                        do ivar=1,ndim
                           read(grav_n)xdp
                            do i=1,ncache
                                g_force(ind_grid(i)+iskip,ivar)=xdp(i)
                            end do
                        end do
                    end do
                    deallocate(ind_grid, xdp)
                endif
            end do
        end do
        close(grav_n)
        deallocate(headl,headb)
    ENDDO

    END SUBROUTINE

!#####################################################################
    subroutine skip_read(unit,nskip)
!#####################################################################
    implicit none
    integer,intent(in) :: unit, nskip
    integer :: i
    do i=1,nskip
        read(unit)
    end do
    end subroutine skip_read

END MODULE io_dice
