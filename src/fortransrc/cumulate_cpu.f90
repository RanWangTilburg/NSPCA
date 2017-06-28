subroutine cumulate_cpu(data, count, N, P)
use iso_c_binding, only: c_int
implicit none
integer(c_int), value :: N, P
integer, dimension(N, P) :: data
integer, dimension(3, P) :: count

integer :: row, col

!#omp do
do col=1, P
    count(1, col)=0
    count(2, col)=0
    count(3, col)=0

    do row = 1,N
        if (data(row, col)==-1) then
            count(1, col) = count(1,col)+1
        else if (data(row,col)==0) then
            count(2, col) = count(2, col)+1
        else
            count(3, col) = count(3, col)+1
        end if
    end do
end do
!#end omp do

end subroutine cumulate_cpu
