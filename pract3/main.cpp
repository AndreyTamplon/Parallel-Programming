#include <iostream>
#include "mpi.h"

void DistributeMatrices(double* A,
						double* part_of_A,
						double* B,
						double* part_of_B,
						int B_x_dim,
						int B_y_dim,
						int part_of_A_x_dim,
						int part_of_A_y_dim,
						int part_of_B_x_dim,
						int part_of_B_y_dim,
						int* coords,
						MPI_Comm column_comm,
						MPI_Comm row_comm)
{
	int send_counts_A = part_of_A_x_dim * part_of_A_y_dim;
	int send_counts_B = 1;
	int recv_counts_A = part_of_A_x_dim * part_of_A_y_dim;
	int recv_counts_B = part_of_B_x_dim * part_of_B_y_dim;
	if (coords[1] == 0)
	{
		MPI_Scatter(A, send_counts_A, MPI_DOUBLE, part_of_A, recv_counts_A,
					MPI_DOUBLE, 0, column_comm);
	}
	MPI_Bcast(part_of_A, send_counts_A, MPI_DOUBLE, 0, row_comm);
	MPI_Datatype column, column_type;
	MPI_Type_vector(B_y_dim, part_of_B_x_dim, B_x_dim, MPI_DOUBLE, &column);
	MPI_Type_commit(&column);
	MPI_Type_create_resized(column, 0, part_of_B_x_dim * sizeof(double), &column_type);
	MPI_Type_commit(&column_type);
	if (coords[0] == 0)
	{
		MPI_Scatter(B, send_counts_B, column_type, part_of_B, recv_counts_B,
					MPI_DOUBLE, 0, row_comm);
	}
	MPI_Bcast(part_of_B, part_of_B_x_dim * part_of_B_y_dim, MPI_DOUBLE, 0, column_comm);
	MPI_Type_free(&column);
	MPI_Type_free(&column_type);
}

double* CreateMatrix(int x, int y)
{
	return new double[x * y];
}

void FillMatrix(int x, int y, double* a)
{
	for (int i = 0; i < y; ++i)
	{
		for (int j = 0; j < x; ++j)
		{
			a[i * x + j] = rand() % 100 + 1;
		}
	}
}

void MatrixMultiplication(const double* A, double* B, double* AB, int A_x_dim, int A_y_dim, int B_x_dim)
{
	for (int i = 0; i < A_y_dim; ++i)
	{
		double* ab = AB + i * B_x_dim;
		for (int j = 0; j < B_x_dim; ++j)
		{
			ab[j] = 0;
		}
		for (int k = 0; k < A_x_dim; ++k)
		{
			double a = A[i * A_x_dim + k];
			double* b = B + k * B_x_dim;
			for (int j = 0; j < B_x_dim; ++j)
			{
				ab[j] += a * b[j];
			}
		}
	}
}

void MatrixBlocksAssembly(double* C, double* part_of_C, int C_x_dim, int C_y_dim,
						  int part_of_C_x_dim, int part_of_C_y_dim, int size, MPI_Comm grid_comm)
{
	MPI_Datatype matrix_block, matrix_block_type;
	MPI_Type_vector(part_of_C_y_dim, part_of_C_x_dim, C_x_dim, MPI_DOUBLE, &matrix_block);
	MPI_Type_commit(&matrix_block);
	MPI_Type_create_resized(matrix_block, 0, part_of_C_x_dim * sizeof(double), &matrix_block_type);
	MPI_Type_commit(&matrix_block_type);
	auto displs = new int[size];
	auto recv_counts = new int[size];
	int blocks_fits_x = C_x_dim / part_of_C_x_dim;
	int blocks_fits_y = C_y_dim / part_of_C_y_dim;
	for (int i = 0, process = 0; i < blocks_fits_y; ++i)
	{
		for (int j = 0; j < blocks_fits_x; ++j)
		{
			recv_counts[process] = 1;
			displs[process] = j + (i * blocks_fits_x) * part_of_C_y_dim;
			process++;
		}
	}
	MPI_Gatherv(part_of_C, part_of_C_x_dim * part_of_C_y_dim, MPI_DOUBLE,
				C, recv_counts, displs, matrix_block_type,
				0, grid_comm);
	MPI_Type_free(&matrix_block);
	MPI_Type_free(&matrix_block_type);
	delete[] displs;
	delete[] recv_counts;
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "3 arguments are required for correct operation " << argc << " are provided" << "\n";
		return EXIT_FAILURE;
	}
	MPI_Init(&argc, &argv);
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm column_comm;
	int ndims = 2, reorder = 0, size, rank, x_dim, y_dim, A_x_dim, A_y_dim, B_x_dim, B_y_dim, C_x_dim, C_y_dim;
	int part_of_A_x_dim, part_of_A_y_dim, part_of_B_x_dim, part_of_B_y_dim, part_of_C_x_dim, part_of_C_y_dim;
	y_dim = atoi(argv[1]);
	x_dim = atoi(argv[2]);
	A_x_dim = 480;
	A_y_dim = 240;
	B_x_dim = 240;
	B_y_dim = A_x_dim;
	C_x_dim = A_y_dim;
	C_y_dim = B_x_dim;
	part_of_A_x_dim = A_x_dim;
	part_of_A_y_dim = A_y_dim / y_dim;
	part_of_B_x_dim = B_x_dim / x_dim;
	part_of_B_y_dim = B_y_dim;
	part_of_C_x_dim = part_of_B_x_dim;
	part_of_C_y_dim = part_of_A_y_dim;
	auto A = CreateMatrix(A_y_dim, A_x_dim);
	auto B = CreateMatrix(B_x_dim, B_y_dim);
	auto C = CreateMatrix(C_x_dim, C_y_dim);
	auto part_of_A = CreateMatrix(part_of_A_x_dim, part_of_A_y_dim);
	auto part_of_B = CreateMatrix(part_of_B_x_dim, part_of_B_y_dim);
	auto part_of_C = CreateMatrix(part_of_C_x_dim, part_of_C_y_dim);
	int dims[2];
	int persisting_dimension[2] = {0, 1};
	int coords[2];
	int periods[2] = {0, 0};
	dims[0] = y_dim;
	dims[1] = x_dim;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0)
	{
		FillMatrix(A_x_dim, A_y_dim, A);
		FillMatrix(B_x_dim, B_y_dim, B);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	auto start = MPI_Wtime();
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
	MPI_Cart_sub(grid_comm, persisting_dimension, &row_comm);
	persisting_dimension[0] = 1;
	persisting_dimension[1] = 0;
	MPI_Cart_sub(grid_comm, persisting_dimension, &column_comm);
	MPI_Comm_rank(grid_comm, &rank);
	MPI_Cart_coords(grid_comm, rank, 2, coords);
	DistributeMatrices(A, part_of_A, B, part_of_B, B_x_dim, B_y_dim,
					   part_of_A_x_dim, part_of_A_y_dim, part_of_B_x_dim, part_of_B_y_dim,
					   coords, column_comm, row_comm);
	MatrixMultiplication(part_of_A, part_of_B, part_of_C,
						 part_of_A_x_dim, part_of_A_y_dim, part_of_B_x_dim);
	MatrixBlocksAssembly(C, part_of_C, C_x_dim, C_y_dim, part_of_C_x_dim, part_of_C_y_dim, size, grid_comm);
	auto end = MPI_Wtime();
	if (rank == 0)
	{
		std::cout << end - start << "\n";
	}
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] part_of_A;
	delete[] part_of_B;
	delete[] part_of_C;
	MPI_Finalize();
	return 0;
}