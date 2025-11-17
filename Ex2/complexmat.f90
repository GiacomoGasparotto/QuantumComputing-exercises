!> @file complexmat.f90
!> @brief Module for handling complex single-precision matrices.
!>
!> This module defines a custom type 'complex8_matrix' and provides
!> several operations such as initialization, random filling,
!> trace computation, adjoint, writing to file, and deallocation.

module complexmat
    
    !***********************************************************************
    !> @brief Custom derived type to represent a complex matrix.
    !>
    !> The 'complex8_matrix' type stores both matrix dimensions and
    !> the actual data in a 2D allocatable array of complex(real32).
    !***********************************************************************

    use iso_fortran_env, only: real32
    use debugger
    implicit none

    type complex8_matrix
        integer, dimension(2) :: size  ! Matrix dimension
        complex(real32), dimension(:,:), allocatable :: elem  ! Matrix elements
    end type

    interface operator(.Tr.)
        module procedure Trace
    end interface

    interface operator(.Adj.)
        module procedure Adjoint
    end interface

contains

    !***********************************************************************
    !> @brief Initialize a complex matrix of given dimensions.
    !>
    !> Allocates memory for the complex matrix elements and sets its size.
    !>
    !> @param[out] M       Complex matrix to initialize.
    !> @param[in]  nrows   Number of rows.
    !> @param[in]  ncols   Number of columns.
    !>
    !> @throws Stops execution if invalid dimensions are provided.
    !***********************************************************************
    subroutine initMatrix(M, nrows, ncols)
        type(complex8_matrix), intent(out) :: M
        integer, intent(in) :: nrows, ncols

        if (nrows <= 0 .or. ncols <= 0) then
            print *, "ERROR: invalid matrix dimensions!"
            stop
        end if

        M%size = [nrows, ncols]
        allocate(M%elem(nrows, ncols))
    end subroutine initMatrix

    !***********************************************************************
    !> @brief Fill a complex matrix with random complex numbers.
    !>
    !> Each element is assigned a random real and imaginary part
    !> in the interval (0,1).
    !>
    !> @param[inout] M  Matrix to be filled.
    !***********************************************************************
    subroutine fillRandomComplexMatrix(M)
        use iso_fortran_env, only: real32
        implicit none

        type(complex8_matrix), intent(inout) :: M
        integer :: i, j
        real(real32) :: Re, Im

        if (.not. allocated(M%elem)) then
            print *, "ERROR: matrix not allocated!"
            stop
        end if

        call random_seed()

        ! Allocate the real and imaginary parts of the matrix
        do i = 1, M%size(1)
            do j = 1, M%size(2)
                call random_number(Re)
                call random_number(Im)
                M%elem(i,j) = cmplx(Re, Im, kind=real32)
            end do
        end do
    end subroutine fillRandomComplexMatrix

    !***********************************************************************
    !> @brief Compute the trace of a complex matrix.
    !>
    !> @param[in] M   Complex matrix (square).
    !> @return tr_M   Complex trace of M.
    !***********************************************************************
    function Trace(M) result(tr_M)
        type(complex8_matrix), intent(in) :: M
        complex(real32) :: tr_M
        integer :: i

        tr_M = complex(0.0_real32, 0.0_real32)
        do i=1, M%size(1)
            tr_M = tr_M + M%elem(i, i)
        end do
    end function Trace

    !***********************************************************************
    !> @brief Compute the adjoint (conjugate transpose) of a complex matrix.
    !>
    !> @param[in] M     Input complex matrix.
    !> @return  adj_M   The adjoint matrix (conjugate transpose of M).
    !***********************************************************************
    function Adjoint(M) result(adj_M)
        type(complex8_matrix), intent(in) :: M
        type(complex8_matrix) :: adj_M

        adj_M%size(1) = M%size(2)
        adj_M%size(2) = M%size(1)

        allocate(adj_M%elem(adj_M%size(1), adj_M%size(2)))
        adj_M%elem = conjg(transpose(M%elem))
    end function Adjoint

    !***********************************************************************
    !> @brief Write a complex matrix to a text file and optionally print it.
    !>
    !> @param[in] M         Matrix to be written.
    !> @param[in] filename  Output file name.
    !> @param[in] do_print  Logical flag to print matrix on screen.
    !***********************************************************************
    subroutine writeMatrix(M, filename, do_print)
        use, intrinsic :: iso_fortran_env, only: real32
        implicit none

        type(complex8_matrix), intent(in) :: M
        character(len=*), intent(in) :: filename
        logical, intent(in) :: do_print
        integer :: u
        integer :: i, j

        ! Print to screen if condition is true
        if (do_print) then
            print *, "Matrix size:", M%size(1), "x", M%size(2)
            do i = 1, M%size(1)
                do j = 1, M%size(2)
                    write(*,"(A,F8.4,A,F8.4,A)", advance="no") "(", real(M%elem(i,j)), ",", aimag(M%elem(i,j)), ") "
                end do
                print *  
            end do
            print *, "" 
        end if

        ! Write to file
        open(newunit=u, file=filename, status="replace", action="write")
        write(u,"(A,I0,A,I0)") "! Complex matrix of size ", M%size(1), "x", M%size(2)
        do i = 1, M%size(1)
            do j = 1, M%size(2)
                write(u, "(A,F8.4,A,F8.4,A)", advance="no") "(", real(M%elem(i,j)), ",", aimag(M%elem(i,j)), ") "
            end do
            write(u,*)
        end do
        close(u)
    end subroutine writeMatrix

    !***********************************************************************
    !> @brief Deallocate the memory of a complex matrix.
    !>
    !> @param[inout] M  Complex matrix to be deallocated.
    !***********************************************************************
    subroutine deallocateMatrix(M)
        type(complex8_matrix), intent(inout) :: M

        if(allocated(M%elem)) then
            deallocate(M%elem)
        else
            print *, "Matrix not allocated!"
        end if
    end subroutine deallocateMatrix

end module complexmat