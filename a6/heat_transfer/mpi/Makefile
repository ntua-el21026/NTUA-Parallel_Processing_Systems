all: jacobi gauss redblack

jacobi: mpi_jacobi.c utils.c
	mpicc -O3 mpi_jacobi.c utils.c -o jacobi_mpi

gauss: mpi_gauss.c utils.c
	mpicc -O3 mpi_gauss.c utils.c -o gauss_mpi

redblack: mpi_redblack.c utils.c
	mpicc -O3 mpi_redblack.c utils.c -o redblack_mpi

clean:
	rm -f jacobi_mpi gauss_mpi redblack_mpi