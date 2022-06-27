#include <cmath>
#include <iostream>
#include "mpi.h"

void MVMultiplication(int N, const double* A, const double* x, double* Ax, int slice_size)
{
	for (int i = 0; i < slice_size; ++i)
	{
		Ax[i] = 0;
		for (int j = 0; j < N; ++j)
		{
			Ax[i] += (*(A + i * N + j) * x[j]);
		}
	}
}

void VCMultiplication(int N, double* x, double tau)
{
	for (int i = 0; i < N; ++i)
	{
		x[i] *= tau;
	}
}

double* CalcSLAU(int N,
				 const double* A,
				 const double* b,
				 double* x0,
				 int slice_size,
				 MPI_Comm communicator,
				 int rank,
				 int* recvcounts,
				 int* displacement,
				 int size,
				 double tau = 0.00001,
				 double epsilon = 0.001)
{
	auto next_x = new double[N];
	auto Ax = new double[slice_size];
	double* swap;
	double partial_b_norm = 0, b_norm = 0, partial_Ax_b_norm = 0, Ax_b_norm = 0;
	displacement[0] = 0;
	recvcounts[0] = recvcounts[0] / N;
	for (int j = 1; j < size; ++j)
	{
		recvcounts[j] /= N;
		displacement[j] = displacement[j - 1] + recvcounts[j - 1];
	}
	for (int i = 0; i < slice_size; ++i)
	{
		partial_b_norm += (b[displacement[rank] + i] * b[displacement[rank] + i]);
	}
	MPI_Allreduce(&partial_b_norm, &b_norm, 1, MPI_DOUBLE, MPI_SUM, communicator);
	epsilon *= epsilon;
	epsilon *= b_norm;
	int i = 0;
	double* last_x = x0;
	int triple_check = 0;
	do
	{
		MVMultiplication(N, A, last_x, Ax, slice_size);
		for (int j = 0; j < slice_size; ++j)
		{
			Ax[j] = Ax[j] - b[displacement[rank] + j];
		}
		partial_Ax_b_norm = 0;
		for (int j = 0; j < slice_size; ++j)
		{
			partial_Ax_b_norm += (Ax[j] * Ax[j]);
		}
		MPI_Allreduce(&partial_Ax_b_norm, &Ax_b_norm, 1, MPI_DOUBLE, MPI_SUM, communicator);
		VCMultiplication(slice_size, Ax, tau);
		for (int j = 0; j < slice_size; ++j)
		{
			Ax[j] = last_x[displacement[rank] + j] - Ax[j];
		}
		MPI_Allgatherv(Ax, slice_size, MPI_DOUBLE, next_x, recvcounts, displacement, MPI_DOUBLE, communicator);
		if (Ax_b_norm < epsilon)
		{
			if (triple_check == 3)
			{
				delete[] Ax;
				delete[] last_x;
				return next_x;
			}
			triple_check++;
		}
		else
		{
			triple_check = 0;

		}
		swap = last_x;
		last_x = next_x;
		next_x = swap;
		++i;
		if (i >= 100000 || Ax_b_norm == INFINITY)
		{
			abort();
		}
	} while (true);
}

void FillMatrix(int N, double* mat)
{

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if (j >= i)
			{
				*(mat + i * N + j) = rand() % 10;
			}
			else
			{
				*(mat + i * N + j) = *(mat + j * N + i);
			}
			if (i == j)
			{
				*(mat + i * N + j) += 500;
			}

		}
	}
}

void FillVector(int N, double* b)
{
	for (int j = 0; j < N; ++j)
	{
		b[j] = N + 1;
	}
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int N = 6000;
	int slice_size = N / size;
	if (rank >= size - N % size && size != 1)
	{
		slice_size++;
	}
	auto* A = new double[N * N];
	auto b = new double[N];
	auto x0 = new double[N];
	int* displacement = new int[size];
	int* sendcounts = new int[size];
	auto piece_of_A = new double[slice_size * N];
	if (rank == 0)
	{
		FillVector(N, b);
		FillMatrix(N, A);
		for (int i = 0; i < N; ++i)
		{
			x0[i] = 0;
		}
	}
	auto start = MPI_Wtime();
	displacement[0] = 0;
	for (int i = 0; i < size; ++i)
	{
		sendcounts[i] = N / size;
		if (i >= size - N % size && size != 1)
		{
			sendcounts[i]++;
		}
		sendcounts[i] *= N;
		if (i != 0)
		{
			displacement[i] = displacement[i - 1] + sendcounts[i - 1];
		}
	}
	MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x0, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(A,
				 sendcounts,
				 displacement,
				 MPI_DOUBLE,
				 piece_of_A,
				 slice_size * N,
				 MPI_DOUBLE,
				 0,
				 MPI_COMM_WORLD);
	double* result = CalcSLAU(N, piece_of_A, b, x0, slice_size, MPI_COMM_WORLD, rank, sendcounts, displacement, size);
	double time = MPI_Wtime() - start;
	std::cout << "rank = " << rank << " time taken: " << time << "\n";
	delete[] displacement;
	delete[] sendcounts;
	delete[] A;
	delete[] b;
	delete[] piece_of_A;
	MPI_Finalize();
	return 0;
}

