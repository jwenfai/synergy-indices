subroutine is_subset(n_coalitions, n_obsv, n_agents, coalitions, N_bool, subset_bool)
    implicit none
    integer, intent(in) :: n_coalitions, n_obsv, n_agents
    integer, intent(in), dimension(0:n_coalitions-1, 0:n_agents-1) :: coalitions
    integer, intent(in), dimension(0:n_obsv-1, 0:n_agents-1) :: N_bool
    integer, intent(out), dimension(0:n_coalitions-1, 0:n_obsv-1) :: subset_bool
    integer :: i, j, k
    do i = 0, n_obsv-1
        do j = 0, n_coalitions-1
            subset_bool(j, i) = 1
            do k = 0, n_agents-1
                if (coalitions(j, k) - N_bool(i, k) < 0) then
                    subset_bool(j, i) = 0
                    exit
                end if
            end do
        end do
    end do
end subroutine is_subset

subroutine is_set(n_coalitions, n_obsv, n_agents, coalitions, N_bool, subset_bool)
    implicit none
    integer, intent(in) :: n_coalitions, n_obsv, n_agents
    integer, intent(in), dimension(0:n_coalitions-1, 0:n_agents-1) :: coalitions
    integer, intent(in), dimension(0:n_obsv-1, 0:n_agents-1) :: N_bool
    integer, intent(out), dimension(0:n_coalitions-1, 0:n_obsv-1) :: subset_bool
    integer :: i, j, k
    do i = 0, n_obsv-1
        do j = 0, n_coalitions-1
            subset_bool(j, i) = 1
            do k = 0, n_agents-1
                if (coalitions(j, k) - N_bool(i, k) /= 0) then
                    subset_bool(j, i) = 0
                    exit
                end if
            end do
        end do
    end do
end subroutine is_set

! This is a Fortran implementation of the issubset function written in Julia. No erroneous result in Julia version.
subroutine issubset_nonbool(n_supersets, n_subsets, n_agents, supersets, subsets, supersets_sizes, subsets_sizes, result_bool)
    implicit none
    integer, intent(in) :: n_supersets, n_subsets, n_agents
    integer, intent(in), dimension(1:n_supersets, 1:n_agents) :: supersets
    integer, intent(in), dimension(1:n_subsets, 1:n_agents) :: subsets
    integer, intent(in), dimension(1:n_supersets) :: supersets_sizes
    integer, intent(in), dimension(1:n_subsets) :: subsets_sizes
    integer, intent(out), dimension(1:n_supersets, 1:n_subsets) :: result_bool
    integer :: i, j, k, m
    do i = 1, n_supersets
        do j = 1, n_subsets
!            print *, i, j
            result_bool(i, j) = 1
            if (supersets_sizes(i) < subsets_sizes(j)) then
!                print *, 'a'
                result_bool(i, j) = 0
            else if ((supersets_sizes(i) == 1) .and. (supersets(i, 1) /= subsets(j, 1))) then
!                print *, 'b'
                result_bool(i, j) = 0
            else
!                print *, 'c'
                k = 1
                m = 1
                do while ((k <= supersets_sizes(i)) .and. (m <= subsets_sizes(j)))
                    if (supersets(i, k) == subsets(j, m)) then
!                        print *, 'd', ' ', k, ' ', m, ' ', supersets_sizes(i), ' ', subsets_sizes(j)
                        m = m+1
                        if (k < supersets_sizes(i)) then
                            k = k+1
                        else if (m > subsets_sizes(j)) then
                            k = k+1
                        end if
                    else if (supersets(i, k) < subsets(j, m)) then
!                        print *, 'e'
                        if (k == supersets_sizes(i)) then
                            result_bool(i, j) = 0
                            exit
                        end if
                        k = k+1
                    else if (supersets(i, k) > subsets(j, m)) then
!                        print *, 'f'
                        result_bool(i, j) = 0
                        exit
                    end if
                end do
            end if
        end do
    end do
end subroutine issubset_nonbool


! This works now
!gfortran-mp-4.8 -c ~/Dropbox/masdar/Thesis/Code/fortransetops.f90 -o ~/Dropbox/masdar/Thesis/Code/fortransetops.o
!f2py-2.7 -c -m --f90exec=/opt/local/bin/gfortran-mp-4.8 fortransetops fortransetops.f90

! Worked before, but not now
!gfortran-mp-4.8 -o ~/Dropbox/masdar/Thesis/Code/fortransetops ~/Dropbox/masdar/Thesis/Code/fortransetops.f90
!~/Dropbox/masdar/Thesis/Code/fortransetops
!f2py-2.7 -c -m --f90exec=/opt/local/bin/gfortran-mp-4.8 fortransetops fortransetops.f90

subroutine is_subset_old_v1(n_coalitions, n_obsv, n_agents, coalitions, N_bool, subset_bool)
    implicit none
    integer, intent(in) :: n_coalitions, n_obsv, n_agents
    integer, intent(in), dimension(0:n_coalitions-1, 0:n_agents-1) :: coalitions
    integer, intent(in), dimension(0:n_obsv-1, 0:n_agents-1) :: N_bool
    integer, intent(out), dimension(0:n_coalitions-1, 0:n_obsv-1, 0:n_agents-1) :: subset_bool
    integer :: i, j, k
    do i = 0, n_agents-1
        do j = 0, n_obsv-1
            do k = 0, n_coalitions-1
                subset_bool(k, j, i) = coalitions(k, i) - N_bool(j, i)
            end do
        end do
    end do
end subroutine is_subset_old_v1

! This implementation gives incorrect results for edge cases
subroutine is_subset_nonbool_old(n_coalitions, n_obsv, n_agents, coalitions, N, coalitions_sizes, N_sizes, subset_bool)
    implicit none
    integer, intent(in) :: n_coalitions, n_obsv, n_agents
    integer, intent(in), dimension(0:n_coalitions-1, 0:n_agents-1) :: coalitions
    integer, intent(in), dimension(0:n_obsv-1, 0:n_agents-1) :: N
    integer, intent(in), dimension(0:n_coalitions-1) :: coalitions_sizes
    integer, intent(in), dimension(0:n_obsv-1) :: N_sizes
    integer, intent(out), dimension(0:n_coalitions-1, 0:n_obsv-1) :: subset_bool
    integer :: i, j, k, m
    do i = 0, n_obsv-1
        do j = 0, n_coalitions-1
            subset_bool(j, i) = 1
            if (N_sizes(i) > coalitions_sizes(j)) then
                subset_bool(j, i) = 0
            else
                k = 0
                m = 0
                do while ((k < coalitions_sizes(j)) .or. (m < N_sizes(i)))
                    if (coalitions(j, k) == N(i, m)) then
                        k = k+1
                        m = m+1
                    else if (coalitions(j, k) > N(i, m)) then
                        k = k+1
                    else if (coalitions(j, k) < N(i, m)) then
                        subset_bool(j, i) = 0
                        exit
                    end if   
                end do
            end if
        end do
    end do
end subroutine is_subset_nonbool_old