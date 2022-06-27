#include <cmath>
#include <iostream>
#include <chrono>

void MVMultiplication(int N, const double* A, const double* x, double* Ax)
{
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
	for (int i = 0; i < N; ++i)
	{
		x[i] *= tau;
	}
}

void VVAddition(int N, const double* a, const double* b, double* result, int sign)
{
	if (sign > 0)
	{
		for (int i = 0; i < N; ++i)
		{
			result[i] = a[i] + b[i];
		}
	}
	else
	{
		for (int i = 0; i < N; ++i)
		{
			result[i] = a[i] - b[i];
		}
	}
}

double CalcVectorLength(int N, const double* a)
{
	double length = 0;
	for (int i = 0; i < N; ++i)
	{
		length += a[i] * a[i];
	}
	return length;
}

double* CalcSLAU(int N, double* A, double* b, double tau, double* x0, double epsilon)
{
	auto next_x = new double[N];
	double* buffer;
	double b_norm = CalcVectorLength(N, b);
	double* last_x = x0;
	int i = 0, triple_check = 0;
	double Ax_b_norm;
	epsilon = epsilon * epsilon * b_norm;
	do
	{
		MVMultiplication(N, A, last_x, next_x);
		VVAddition(N, next_x, b, next_x, -1);
		Ax_b_norm = CalcVectorLength(N, next_x);
		VCMultiplication(N, next_x, tau);
		VVAddition(N, last_x, next_x, next_x, -1);
		if (Ax_b_norm < epsilon)
		{
			if (triple_check == 3)
			{
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
		buffer = last_x;
		last_x = next_x;
		next_x = buffer;
		++i;
		if (i >= 100000 || Ax_b_norm == INFINITY)
		{
			abort();
		}
	} while (true);
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
	int N = 6000;
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

