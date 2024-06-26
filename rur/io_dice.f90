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

    !! GET PART
    INTEGER(KIND=4) ncpu, nsnap, npart_tot
    CHARACTER(LEN=1000) repo

    REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: g_pos, g_vel
    REAL(KIND=4), DIMENSION(:), ALLOCATABLE :: g_mm
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: g_id, g_type 
CONTAINS
    SUBROUTINE read_gadget(fname)

    IMPLICIT NONE
    CHARACTER(1000), INTENT(IN) :: fname

    !! LOCAL VARIABLES
    logical :: ok
    INTEGER :: dummy_int, blck_size, head_blck, np, ntype, i0, i1, i
    CHARACTER(LEN=4) :: blck_name


    TYPE(htype) :: header

    OPEN(unit=10, file=TRIM(fname), status='old', form='unformatted', action='read', access='stream')

    !! READ HEADER

    READ(10,POS=1+sizeof(dummy_int)) blck_name
    READ(10,POS=1+sizeof(dummy_int)+sizeof(blck_name)) dummy_int
    READ(10,POS=1+2*sizeof(dummy_int)+sizeof(blck_name)) dummy_int
    READ(10,POS=1+3*sizeof(dummy_int)+sizeof(blck_name)) blck_size
    head_blck = 1+sizeof(blck_name)+4*sizeof(dummy_int)

    READ(10,POS=head_blck) header%npart,header%mass,header%time,header%redshift, &
         header%flag_sfr,header%flag_feedback,header%nparttotal, &
         header%flag_cooling,header%numfiles,header%boxsize, &
         header%omega0,header%omegalambda,header%hubbleparam, &
         header%flag_stellarage,header%flag_metals,header%totalhighword, &
         header%flag_entropy_instead_u, header%flag_doubleprecision, &
         header%flag_ic_info, header%lpt_scalingfactor

    !PRINT *, header%mass

    !! READ PART
    np = sum(header%npart)

    CALL read_gadget_allocate(np)

    READ(10) pos
    READ(10) vel
    READ(10) id
    READ(10) mm
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

!!!!!
    SUBROUTINE read_gadget_allocate(np)
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

!!!!!
    SUBROUTINE get_part()
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
      fdum = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/part_'//TRIM(snum)//'.out'//TRIM(domnum)

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

    SUBROUTINE get_part_allocate()
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

    SUBROUTINE get_part_tot()
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
      fdum = TRIM(repo)//'/snapshots/output_'//TRIM(snum)//'/part_'//TRIM(snum)//'.out'//TRIM(domnum)

      OPEN(UNIT=10, FILE=fdum, FORM='unformatted', STATUS='OLD')
      READ(10); READ(10); READ(10) np
      parts(i) = np
    ENDDO

    npart_tot = SUM(parts)
    DEALLOCATE(parts)
    END SUBROUTINE


END MODULE io_dice
