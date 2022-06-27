# Fixed-point iteration method

1. 3 programs (one sequential and 2 parallel) in C or C++, which implement an iterative algorithm for solving a system of linear algebraic equations of the form Ax=b. Here A is a matrix of size N×N, x and b are vectors of length N. The type of elements is double.
2. 2 parallel programs are implemented using MPI with cutting the matrix A by rows (in the first program) or by columns (in the second) into parts that are close in size, possibly not the same. Program 1: Vectors x and b are duplicated in each MPI process, Program 2: vectors x and b are split between MPI processes.
3. The operation time of the serial version of the program and 2 parallel ones was measured when using a different number of processor cores. (2, 4, 8, 16, 24).

| Number of processes | Sequential version | Shared vectors | Distributed vectors |
| ------------------- | ------------------ | -------------- | ------------------- |
| 2                   | 30,57              | 16,429         | 20,123              |
| 4                   | 30,57              | 10,478         | 11,762              |
| 8                   | 30,57              | 8,131          | 7,539               |
| 16                  | 30,57              | 5,859          | 5,933               |
| 24                  | 30,57              | 4,33           | 4,593               |
