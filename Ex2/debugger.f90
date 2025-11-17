!> @file debugger.f90
!> @brief Debugging and error-checking utilities for matrix operations.
!>
!> This module provides the subroutine 'checkpoint', which is useful for
!> runtime debugging, error handling, and code flow control.
!> It supports different verbosity levels to perform optional checks
!> (e.g., printing messages, dimension compatibility, square-matrix verification).

module debugger

    !***********************************************************************
    !> @brief Perform runtime debugging and validation checkpoints.
    !>
    !> The 'checkpoint' subroutine provides an interface for handling
    !> debugging messages and validity checks for matrices during execution.
    !> Depending on the selected verbosity level, it can:
    !>  - 0: print a debug message
    !>  - 1: check matrix dimension compatibility
    !>  - 2: verify that a matrix is square
    !>
    !> @param[in] debug      Logical flag to enable or disable debugging.
    !> @param[in] verbosity  Integer verbosity level:
    !>                       - **0**: print a debug message
    !>                       - **1**: check matrix dimension compatibility
    !>                       - **2**: verify if a matrix is square or not
    !> @param[in,optional] message  Custom message to print (verbosity 0 or 2).
    !> @param[in,optional] A(:,:)   First matrix to check (verbosity 1).
    !> @param[in,optional] B(:,:)   Second matrix to check (verbosity 1).
    !> @param[in,optional] nrows    Number of rows (verbosity 2).
    !> @param[in,optional] ncols    Number of columns (verbosity 2).
    !>
    !> @throws Stops execution if:
    !>         - matrix dimensions are incompatible (verbosity=1)
    !>         - matrix is not square (verbosity=2)
    !>         - an invalid verbosity level is provided
    !>
    !> @note The subroutine safely exits if 'debug' is '.false.'
    !***********************************************************************

    implicit none

contains
    subroutine checkpoint(debug, verbosity, message, A, B, nrows, ncols)
        use iso_fortran_env, only: real32
        implicit none

        ! Allocate variables
        logical, intent(in) :: debug
        integer, intent(in) :: verbosity
        character(len=*), intent(in), optional :: message
        real(real32), intent(in), optional :: A(:,:), B(:,:)
        integer, intent(in), optional :: nrows, ncols

        ! Early exit if debugging is disabled
        if (.not. debug) return

        select case(verbosity)
        case (0) ! Print a custom debug message
            if (present(message)) then
                print *, ""
                print *, trim(message)
                print *, ""
            end if

        case (1) ! Check matrix compatibility for multiplication
            if (present(A) .and. present(B)) then
                if (size(A,2) /= size(B,1)) then
                    print *, "ERROR: incompatible matrix dimensions!"
                    stop
                end if
            else
                print *, "WARNING: Missing A or B in verbosity level 1"
            end if

        case (2) ! Check if matrix is square
            if (present(nrows) .and. present(ncols)) then
                if (nrows /= ncols) then
                    if (present(message)) then
                        print *, "ERROR: nrows /= ncols — the ", trim(message), " is only defined for square matrices."
                    else
                        print *, "ERROR: nrows /= ncols — matrix is not square."
                    end if
                    stop
                end if
            else
                print *, "WARNING: Missing nrows/ncols in verbosity level 2"
            end if

        case default
            print *, "ERROR: Invalid verbosity level. Choose 0, 1, or 2."
            stop
        end select
    end subroutine checkpoint

end module debugger