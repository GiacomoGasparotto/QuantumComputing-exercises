!> @file main.f90
!> @brief Main program for testing real and complex matrix operations.
!>
!> This program demonstrates:
!> - Real matrix multiplication using different algorithms (row-by-col, col-by-row, built-in matmul)
!> - Performance benchmarking for increasing matrix dimensions
!> - Operations on complex matrices (trace, adjoint)
!> - Use of the 'checkpoint' subroutine from the 'debugger' module for error checking
!>
!> @note Output data for timing tests are written to:
!>       - 'row_col.dat'
!>       - 'col_row.dat'
!>       - 'matmul.dat'
!>
!> @note To read the Doxygen file execute: 'firefox docs/html/index.html'
!>
!> @author Giacomo Gasparotto
!> @date 18/11/2025

program main
    use iso_fortran_env, only: real32
    use debugger 
    use matmul_module
    use complexmat
    implicit none

    ! Allocate matrices (real type)
    real(real32), allocatable :: M1(:,:), M2(:,:), prod1(:,:), prod2(:,:), prod3(:,:) 

    ! Allocate matrices (complex type)
    type(complex8_matrix) :: A, adj_A
    complex(real32) :: tr_A
    integer :: nrows, ncols

    ! Allocate matrices dimensions
    integer :: m, n, p 
    integer :: i, dim, dimensions(10)

    ! Time variables
    real(real32) :: init_time, end_time

    ! ======================================         
    !       REAL MATRIX EXERCISE
    ! ======================================

    ! Define matrix dimensions
    m = 2
    n = 3
    p = 5

    ! Allocate (real) matrices
    allocate(M1(m,n), M2(n,p), prod1(m,p), prod2(m,p), prod3(m,p))

    call random_seed()
    call random_number(M1)
    call random_number(M2)  

    ! Compute the matrix-matrix product
    call matmul_rowbycol(M1, M2, prod1, m, n, p, .true.)
    call matmul_colbyrow(M1, M2, prod2, m, n, p, .true.)
    print *, "Result of builtin matmul:"
    print *, matmul(M1, M2)  
    print *, ""

    deallocate(M1, M2, prod1, prod2, prod3)

    ! Now test performance for increasing matrix sizes  
    ! For simplicity, let's consider square matrices only
    ! Note: you can decrease matrices dimension for a lower compilation time
    dimensions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    ! loop over dimensions to check CPU perfomance for increasing matrix size
    call checkpoint(.true., 0, "Computing matrix-matrix multiplication for different dimensions...")
    do i=1, size(dimensions)
        dim = dimensions(i)
        print *, "Performing multiplication for matrix dimension: ", dim, "x", dim, "..."
        allocate(M1(dim,dim), M2(dim,dim), prod1(dim,dim), prod2(dim,dim), prod3(dim,dim))
        
        call random_seed() 
        call random_number(M1)
        call random_number(M2)

        ! Overwrite data the first time to create/clear files
        if (i==1) then
            open(unit=10, file="row_col.dat", status="replace")
            close(10)
            open(unit=10, file="col_row.dat", status="replace")
            close(10)
            open(unit=10, file="matmul.dat", status="replace")
            close(10)
        end if

        ! compute row-by-col multiplication and CPU time and write the results into a file
        open(unit=10, file="row_col.dat", status="unknown", position="append")
        call cpu_time(init_time)
        call matmul_rowbycol(M1, M2, prod1, dim, dim, dim, .false.)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        ! compute col-by-row multiplication and CPU time and write the results into a file
        open(unit=10, file="col_row.dat", status="unknown", position="append")
        call cpu_time(init_time)
        call matmul_colbyrow(M1, M2, prod2, dim, dim, dim, .false.)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        ! compute 'matmul' multiplication and CPU time and write the results into a file
        open(unit=10, file="matmul.dat", status="unknown", position="append")
        call cpu_time(init_time)
        prod3 = matmul(M1, M2)
        call cpu_time(end_time)
        write(10,*) dim, end_time - init_time
        close(10)

        ! deallocate matrices
        deallocate(M1, M2, prod1, prod2, prod3)   
    end do
    call checkpoint(.true., 0, "Calculation done!")

    ! ======================================         
    !       COMPLEX MATRIX EXERCISE
    ! ======================================

    ! initialize and fill matrix A
    nrows = 5
    ncols = 5
    call initMatrix(A, nrows, ncols)
    call fillRandomComplexMatrix(A)
    print *, "Matrix A:"
    call writeMatrix(A, "A.txt", .true.)

    ! Compute the adjoint of A
    call checkpoint(.true., 2, "adjoint", nrows=nrows, ncols=ncols)
    call initMatrix(adj_A, nrows, ncols)
    adj_A = .Adj.A
    print *, "Adjoint of A:"
    call writeMatrix(Adj_A, "adj_A.txt", .true.)

    ! Compute the trace of A
    call checkpoint(.true., 2, "trace", nrows=nrows, ncols=ncols)
    tr_A = .Tr.A
    print *, "Trace of A:", tr_A

    ! Deallocate matrices
    call deallocateMatrix(A)
    call deallocateMatrix(adj_A)

end program main