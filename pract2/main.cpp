#include <cmath>
#include <iostream>
#include <chrono>

void MVMultiplication(int N, const double* A, const double* x, double* Ax)
{
#pragma omp for
	for (int i = 0; i < N; ++i)
	{
		Ax[i] = 0;
		for (int j = 0; j < N; ++j)
		{
			Ax[i] = Ax[i] + (*(A + i * N + j) * x[j]);
		}
	}
}

void VCMultiplication(int N, double* x, double tau)
{
#pragma omp for
	for (int i = 0; i < N; ++i)
	{
		x[i] *= tau;
	}
}

void VVSubtraction(int N, const double* a, const double* b, double* result)
{
#pragma omp for
	for (int i = 0; i < N; ++i)
	{
		result[i] = a[i] - b[i];
	}
}

double CalcVectorLength(int N, const double* a)
{
	double length = 0;
#pragma omp parallel for reduction(+: length)
	for (int i = 0; i < N; ++i)
	{
		length += a[i] * a[i];
	}
	return length;
}

double* CalcSLAU(int N,  double* A, double* b, double tau, double* x0, double epsilon)
{
	auto next_x = new double[N];
	double* buffer;
	double b_norm = CalcVectorLength(N, b);
	double* last_x = x0;
	int i = 0, triple_check = 0;
	double Ax_b_norm;
	epsilon = epsilon * epsilon * b_norm;
#pragma omp parallel
	do
	{
#pragma omp for
		for (int k = 0; k < N; k++)
		{
			last_x[k] = next_x[k];
		}
#pragma omp barrier
		MVMultiplication(N, A, last_x, next_x);
		VVSubtraction(N, next_x, b, next_x);
		Ax_b_norm = CalcVectorLength(N, next_x);
#pragma omp barrier
		VCMultiplication(N, next_x, tau);
		VVSubtraction(N, last_x, next_x, next_x);
#pragma omp single
		{
			++i;
		}
		if (i >= 10000 || Ax_b_norm == INFINITY)
		{
			abort();
		}
	} while (Ax_b_norm >= epsilon);
	return next_x;
}

double* GenerateMatrix(int N)
{
	auto mat = new double[N * N];
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
	return mat;
}

double* GenerateVector(int N)
{
	auto b = new double[N];
	for (int j = 0; j < N; ++j)
	{
		b[j] = N + 1;
	}
	return b;
}
int main()
{
	int N = 2900;
	double* A = GenerateMatrix(N);
	double* b = GenerateVector(N);
	auto x0 = new double[N];
	for (int i = 0; i < N; ++i)
	{
		x0[i] = 0;
	}
	auto start = std::chrono::steady_clock::now();
	double* result = CalcSLAU(N, A, b, 0.00001, x0, 0.001);
	double time =
		(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)).count();
	std::cout << "time taken: " << time << "\n";
	delete[] result;
	delete[] A;
	delete[] b;
	return 0;
}