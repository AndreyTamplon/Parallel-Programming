#include <cmath>
#include <iostream>
#include "mpi.h"

double CalcVectorLength(int N, const double* a)
{
	double length = 0;
	for (int i = 0; i < N; ++i)
	{
		length += a[i] * a[i];
	}
	return length;
}

double* CalcSLAU(int N,
				 const double* A,
				 double* b,
				 double* x0,
				 int slice_size,
				 MPI_Comm communicator,
				 int* recvcounts,
				 int* displacement,
				 int size,
				 double tau = 0.00001,
				 double epsilon = 0.001)
{
	auto next_x = new double[slice_size];
	double* swap;
	double partial_b_norm, b_norm, partial_Ax_b_norm = 0, Ax_b_norm = 0;
	partial_b_norm = CalcVectorLength(slice_size, b);
	MPI_Allreduce(&partial_b_norm, &b_norm, 1, MPI_DOUBLE, MPI_SUM, communicator);
	epsilon *= epsilon;
	epsilon *= b_norm;
	double* last_x = x0;
	displacement[0] = 0;
	recvcounts[0] = recvcounts[0] / N;
	for (int i = 1; i < size; ++i)
	{
		recvcounts[i] /= N;
		displacement[i] = displacement[i - 1] + recvcounts[i - 1];
	}
	int i = 0;
	int triple_check = 0;
	auto partial_Ax = new double[N];
	auto Ax = new double[N];
	do
	{
		partial_Ax_b_norm = 0;
		for (int l = 0; l < slice_size; ++l)
		{
			for (int k = 0; k < N; ++k)
			{
				if (l == 0)
				{
					partial_Ax[k] = last_x[l] * A[l * N + k];
				}
				else
				{
					partial_Ax[k] += last_x[l] * A[l * N + k];
				}
			}
		}
		MPI_Allreduce(partial_Ax, Ax, N, MPI_DOUBLE, MPI_SUM, communicator);
		MPI_Scatterv(Ax,
					 recvcounts,
					 displacement,
					 MPI_DOUBLE,
					 next_x,
					 slice_size,
					 MPI_DOUBLE,
					 0,
					 MPI_COMM_WORLD);
		for (int j = 0; j < slice_size; ++j)
		{
			next_x[j] -= b[j];
			partial_Ax_b_norm += (next_x[j] * next_x[j]);
			next_x[j] = last_x[j] - tau * next_x[j];
		}
		MPI_Allreduce(&partial_Ax_b_norm, &Ax_b_norm, 1, MPI_DOUBLE, MPI_SUM, communicator);
		if (Ax_b_norm < epsilon)
		{
			if (triple_check == 3)
			{
				delete[] Ax;
				delete[] partial_Ax;
				delete[] last_x;
				return next_x;
			}
			else
			{
				triple_check++;
			}
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

void FillVector(int N, double* b, double value)
{
	for (int j = 0; j < N; ++j)
	{
		b[j] = value;
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
	auto b = new double[slice_size];
	auto x0 = new double[slice_size];
	int* displacement = new int[size];
	int* sendcounts = new int[size];
	auto piece_of_A = new double[slice_size * N];
	if (rank == 0)
	{
		FillMatrix(N, A);
	}
	FillVector(slice_size, b, N + 1);
	FillVector(slice_size, x0, 0);
	auto result = new double[N];
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
	MPI_Scatterv(A, sendcounts, displacement, MPI_DOUBLE, piece_of_A, slice_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	double
		* partial_result = CalcSLAU(N, piece_of_A, b, x0, slice_size, MPI_COMM_WORLD, sendcounts, displacement, size);
	MPI_Allgatherv(partial_result,
				   slice_size,
				   MPI_DOUBLE,
				   result,
				   sendcounts,
				   displacement,
				   MPI_DOUBLE,
				   MPI_COMM_WORLD);
	double time = MPI_Wtime() - start;
	std::cout << "rank = " << rank << " time taken: " << time << "\n";
	delete[] displacement;
	delete[] result;
	delete[] sendcounts;
	delete[] A;
	delete[] b;
	delete[] piece_of_A;
	MPI_Finalize();
	return 0;
}


