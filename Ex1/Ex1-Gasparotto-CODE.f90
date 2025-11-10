!=================================================================================================================
! Assignment 1   
! Author: Giacomo Gasparotto (2156362)  
! ================================================================================================================
!  - Compile this program with: gfortran Ex1-Gasparotto-CODE.f90 -o ex1 -fno-range-check
!    and then execute it with: ./ex1
!    (You can choose the name you prefer insteasd of 'ex1')
!
!  - In order to see the effect of different optimization levels, execute the bash file: ./optimization.sh 
!    it will automatize the optimization process.
!
!  - Plots are generated with the notebook: 'plots.ipynb'
! ================================================================================================================
! This program performs the following tasks:
! Ex.2a) Demonstrates integer overflow with 16-bit and 32-bit integer precision.
! Ex.2b) Demonstrates rounding errors in floating-point arithmetic with single and double precision.
! Ex. 3) Implements matrix-matrix multiplication using two different algorithms (row-by-column and column-by-row)
!        and compares their performance with the built-in matmul function. 
!        - Run 'plots.ipynb' file to plot the results of this section.
!        - Compile this program with ./optimization.sh to see the effect of different optimization flags and then
!          run 'plots.ipynb' to plot the results.
! ================================================================================================================

program main
    use iso_fortran_env, only: int16, int32, real32, real64
    use matmul_module
    implicit none

    ! ==== Ex. 2a integer precision ====
    integer(int16) :: n1_short, n2_short
    integer(int32) :: n1_long, n2_long

    ! ==== Ex. 2b real precision ====
    real(real32) :: pi_single, n1_single, n2_single, single_sum
    real(real64) :: pi_double, n1_double, n2_double, double_sum

    ! ==== Ex. 3 matrix-matrix multiplication ====
    ! allocate matrices
    real(real32), allocatable :: M1(:,:), M2(:,:), prod1(:,:), prod2(:,:), prod3(:,:) 

    ! allocate matrices dimensions
    integer :: m, n, p 
    integer :: i, dim, dimensions(10)

    ! time variables
    real(real32) :: init_time, end_time

    ! ===== 2.a =====
    n1_short = 2000000
    n2_short = 1
    n1_long = 2000000
    n2_long = 1

    print *, "Sum of 2000000 and 1 using 16-bit integers:"
    print *, n1_short, "+", n2_short, "=", n1_short + n2_short
    print *, "Max interval for 16-bit: [", -huge(0_int16)-1, ",", huge(0_int16), "]"
    print *, ""

    print *, "Sum of 2000000 and 1 using 32-bit integers:"
    print *, n1_long, "+", n2_long, "=", n1_long + n2_long
    print *, "Max interval for 32-bit: [", -huge(0_int32)-1, ",", huge(0_int32), "]"
    print *, ""

    ! ===== 2.b =====
    ! Define numbers in single precision
    pi_single = acos(-1.0_real32) ! Pi in single precision, no built-in constant present
    n1_single = pi_single * 1.0e32_real32
    n2_single = sqrt(2.0_real32) * 1.0e21_real32
    single_sum = n1_single + n2_single

    print *, "Single precision real numbers to sum:"
    print *, n1_single
    print *, n2_single
    print *, 'Single precision sum: ', single_sum
    print *, ""

    ! Define numbers in double precision
    pi_double = acos(-1.0_real64) ! Pi in double precision, no built-in constant present
    n1_double = pi_double * 1.0e32_real64
    n2_double = sqrt(2.0_real64) * 1.0e21_real64 
    double_sum = n1_double + n2_double
    
    print *, "Double precision real numbers to sum:"
    print *, n1_double
    print *, n2_double
    print *, 'Double precision sum: ', double_sum
    print *, 'Rounding error occurs in single precision due to limited significant digits.'
    print *, ""

    ! ===== 3 =====
    ! Define matrix dimensions
    m = 2
    n = 3
    p = 5

    ! Allocate matrices
    allocate(M1(m,n), M2(n,p), prod1(m,p), prod2(m,p), prod3(m,p))

    call random_seed()
    call random_number(M1)
    call random_number(M2)  

    call matmul_rowbycol(M1, M2, prod1, m, n, p, .true.)
    call matmul_colbyrow(M1, M2, prod2, m, n, p, .true.)
    print *, "Result of builtin matmul:"
    print *, matmul(M1, M2)  
    print *, ""

    deallocate(M1, M2, prod1, prod2, prod3)

    ! Now test performance for increasing matrix sizes  
    ! For simplicity, let's consider square matrices only
    ! Note: you can decrease matrices dimension for a lower compilation time
    dimensions = [2, 10, 100, 200, 400, 600, 800, 1000, 2000, 5000]
    
    do i=1, size(dimensions)
        dim = dimensions(i)
        print *, "Performing multiplication for matrix dimension: ", dim, "x", dim, "..."
        allocate(M1(dim,dim), M2(dim,dim), prod1(dim,dim), prod2(dim,dim), prod3(dim,dim))
        
        call random_seed() 
        call random_number(M1)
        call random_number(M2)

        ! Overwrite data the first time to create/clear files
        if (i==1) then
            open(unit=10, file='row_col.dat', status="replace")
            close(10)
            open(unit=10, file='col_row.dat', status="replace")
            close(10)
            open(unit=10, file='matmul.dat', status="replace")
            close(10)
        end if

        open(unit=10, file='row_col.dat', status="unknown", position="append")
        call cpu_time(init_time)
        call matmul_rowbycol(M1, M2, prod1, dim, dim, dim, .false.)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        open(unit=10, file='col_row.dat', status="unknown", position="append")
        call cpu_time(init_time)
        call matmul_colbyrow(M1, M2, prod2, dim, dim, dim, .false.)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        open(unit=10, file='matmul.dat', status="unknown", position="append")
        call cpu_time(init_time)
        prod3 = matmul(M1, M2)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        deallocate(M1, M2, prod1, prod2, prod3)
    end do

end program main


! module to handle two subroutines for matrix-matrix multiplication algorithms
module matmul_module
    use iso_fortran_env, only: real32   
    implicit none
contains    

    !====================================================================
    !> Perform matrix multiplication C = A * B using row-by-column order
    !!  A: matrix of size (m,n)
    !!  B: matrix of size (n,p)
    !!  C: result matrix of size (m,p)
    !!  Optional argument "do_print" controls whether to print the result
    !====================================================================
    subroutine matmul_rowbycol(A, B, C, m, n, p, do_print)
        use iso_fortran_env, only: real32
        implicit none

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
        if (size(A,2) /= size(B,1)) then
            print *, "Error: incompatible matrix dimensions!"
            stop
        end if

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


    !====================================================================
    !> Perform matrix multiplication C = A * B using column-by-row order
    !!  A: matrix of size (m,n)
    !!  B: matrix of size (n,p)
    !!  C: result matrix of size (m,p)
    !!  Optional argument "do_print" controls whether to print the result
    !====================================================================
    subroutine matmul_colbyrow(A, B, C, m, n, p, do_print)
        use iso_fortran_env, only: real32
        implicit none

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
        if (size(A,2) /= size(B,1)) then
            print *, "Error: incompatible matrix dimensions!"
            stop
        end if

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