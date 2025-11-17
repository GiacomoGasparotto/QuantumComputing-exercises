!> @file matmul_module.f90
!> @brief Matrix multiplication utilities.
!>
!> This module provides subroutines to perform matrix multiplication
!> using different memory access orders (row-by-column and column-by-row).
!> Both routines include optional debug printing and dimension checks.

module matmul_module 

    !***********************************************************************
    !> @brief Perform matrix multiplication \( C = A \times B \) using row-by-column order.
    !>
    !> This subroutine computes the product of two matrices using the
    !> row-by-column algorithm.
    !>
    !> @param[in]  A(m,n)           Left-hand matrix operand.
    !> @param[in]  B(n,p)           Right-hand matrix operand.
    !> @param[out] C(m,p)           Result matrix.
    !> @param[in]  m                Number of rows in A.
    !> @param[in]  n                Number of columns in A and rows in B.
    !> @param[in]  p                Number of columns in B.
    !> @param[in,optional] do_print Logical flag to print the resulting matrix.
    !>
    !> @note Calls 'checkpoint' from the 'debugger' module to ensure
    !>       the dimension match of the two matrices.
    !***********************************************************************

    use iso_fortran_env, only: real32   
    implicit none
    contains   

    subroutine matmul_rowbycol(A, B, C, m, n, p, do_print)
        use iso_fortran_env, only: real32
        use debugger
        implicit none

        ! Allocate variables
        integer, intent(in) :: m, n, p
        real(real32), intent(in) :: A(m,n), B(n,p)
        real(real32), intent(out) :: C(m,p)
        logical, intent(in), optional :: do_print
        logical :: print_res
        integer :: i, j, k

        ! Default printing behavior
        if (present(do_print)) then
            print_res = do_print
        else
            print_res = .true.
        end if

        ! Check matrix dimension compatibility
        call checkpoint(.true., 1, "", A, B)

        ! Matrix multiplication (row-by-column)
        do i = 1, m
            do j = 1, p
                C(i,j) = 0.0_real32
                do k = 1, n
                    C(i,j) = C(i,j) + A(i,k) * B(k,j)
                end do
            end do
        end do

        if (print_res) then
            print *, "Result of matmul rowbycol:"
            print *, C
        end if
    end subroutine matmul_rowbycol


    !***********************************************************************
    !> @brief Perform matrix multiplication \( C = A \times B \) using column-by-row order.
    !>
    !> This subroutine computes the product of two matrices using
    !> the row-by-col algorithm.
    !>
    !> @param[in]  A(m,n)           Left-hand matrix operand.
    !> @param[in]  B(n,p)           Right-hand matrix operand.
    !> @param[out] C(m,p)           Result matrix.
    !> @param[in]  m                Number of rows in A.
    !> @param[in]  n                Number of columns in A and rows in B.
    !> @param[in]  p                Number of columns in B.
    !> @param[in,optional] do_print Logical flag to print the resulting matrix.
    !>
    !> @note Calls 'checkpoint' from the 'debugger' module to ensure
    !>       the dimension match of the two matrices.
    !***********************************************************************
    subroutine matmul_colbyrow(A, B, C, m, n, p, do_print)
        use iso_fortran_env, only: real32
        use debugger
        implicit none

        ! Allocate variables
        integer, intent(in) :: m, n, p
        real(real32), intent(in) :: A(m,n), B(n,p)
        real(real32), intent(out) :: C(m,p)
        logical, intent(in), optional :: do_print
        logical :: print_res
        integer :: i, j, k

        ! Default printing behavior
        if (present(do_print)) then
            print_res = do_print
        else
            print_res = .true.
        end if

        ! Check matrix dimension compatibility
        call checkpoint(.true., 1, "", A, B)

        ! Matrix multiplication (column-by-row)
        do j = 1, p
            do i = 1, m
                C(i,j) = 0.0_real32
                do k = 1, n
                    C(i,j) = C(i,j) + A(i,k) * B(k,j)
                end do
            end do
        end do

        if (print_res) then
            print *, "Result of matmul colbyrow:"
            print *, C
        end if
    end subroutine matmul_colbyrow

end module matmul_module