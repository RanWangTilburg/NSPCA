subroutine GramSchmidt(out, nrow, ncol)
integer :: nrow, ncol
real*8, dimension(nrow, ncol) :: out
INTEGER :: i, j
REAL*8    :: norma

norma = sqrt(dot_product(out(:,1), out(:,1)))


do i=1, nrow
	out(i,1) = out(i,1)/norma
end do

do i=1, ncol
	do j=1,i-1
		norma = 0.0
		norma = norma + dot_product(out(:,i), out(:, j))
		out(:,i) = out(:,i)-norma*out(:,j)
		norma = sqrt(dot_product(out(:, i), out(:,i)))
		out(:,i) = out(:,i)/norma	
	end do 
end do

end subroutine GramSchmidt

