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

program main
    use iso_fortran_env, only: int16, int32, real32, real64
    use matmul_module
    implicit none

    ! ===============================================
    !       Ex. 3 matrix-matrix multiplication 
    ! ===============================================

    ! allocate matrices
    real(real32), allocatable :: M1(:,:), M2(:,:), prod1(:,:), prod2(:,:), prod3(:,:) 
    integer :: dim, i
    ! time variables
    real(real32) :: init_time, end_time

    
    ! Test the performance for increasing matrix sizes  
    ! For simplicity, let's consider square matrices only
    ! Note: you can decrease matrices dimension for a lower compilation time
    
    ! read the dimension from input
    read(*,*) dim

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


end program main