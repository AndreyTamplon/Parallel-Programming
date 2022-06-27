#include "iostream"
#include "mpi.h"
#include "cassert"
#include "cstring"
#include "vector"

bool* CreateField(int rows, int columns)
{
	bool* field = new bool[rows * columns];
	assert(field);
	memset(field, false, rows * columns);
	return field;
}

void PutGliderOnField(bool* field, int columns)
{
	field[0 * columns + 1] = true;
	field[1 * columns + 2] = true;
	field[2 * columns + 0] = true;
	field[2 * columns + 1] = true;
	field[2 * columns + 2] = true;
}

void FillSendcounts(int* sendcounts, int size, int rows, int columns)
{
	int base_slice = rows / size;
	for (int i = 0; i < size; ++i)
	{
		sendcounts[i] = base_slice;
		if (i < rows % size && size != 1)
		{
			sendcounts[i]++;
		}
	}
	for (int i = 0; i < size; ++i)
	{
		sendcounts[i] *= columns;
	}
}

void FillDispls(int* displs, const int* sendcounts, int size)
{
	displs[0] = 0;
	for (int i = 1; i < size; ++i)
	{
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}
}

void SetStopFlags(bool* stop_flags,
				  const std::vector<bool*> &previous_states,
				  const bool* subfield,
				  int rows,
				  int columns)
{
	int i = 0;
	for (auto state: previous_states)
	{
		if (i >= previous_states.size() - 1)
		{
			break;
		}
		stop_flags[i] = true;
		for (int j = columns; j < (columns * (rows + 1)); ++j)
		{
			if (state[j] != subfield[j])
			{
				stop_flags[i] = false;
				break;
			}
		}
		++i;
	}
}

bool CheckForColumnOfOne(const bool* stop_flags_matrix, int rows, int columns)
{
	bool column_has_zero = false;
	for (int i = 0; i < columns; ++i)
	{
		for (int j = 0; j < rows; ++j)
		{
			if (stop_flags_matrix[j * columns + i] == 0)
			{
				column_has_zero = true;
				break;
			}
		}
		if (!column_has_zero)
		{
			return true;
		}
	}
	return false;
}

void FreePreviousStatesVector(const std::vector<bool*> &previous_states)
{
	for (bool* state: previous_states)
	{
		delete[] state;
	}
}

void NextLifeIteration(const bool* previous_step, bool* next_step, int rows, int columns)
{
	for (int i = 1; i < rows - 1; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			int alive_neighbors = 0;
			if (previous_step[(i - 1) * columns + j])
			{
				alive_neighbors++;
			}
			if (previous_step[(i + 1) * columns + j])
			{
				alive_neighbors++;
			}
			if (previous_step[(i - 1) * columns + (j + columns - 1) % columns])
			{
				alive_neighbors++;
			}
			if (previous_step[(i - 1) * columns + (j + 1) % columns])
			{
				alive_neighbors++;
			}
			if (previous_step[(i + 1) * columns + (j + columns - 1) % columns])
			{
				alive_neighbors++;
			}
			if (previous_step[(i + 1) * columns + (j + 1) % columns])
			{
				alive_neighbors++;
			}
			if (previous_step[i * columns + (j + columns - 1) % columns])
			{
				alive_neighbors++;
			}
			if (previous_step[i * columns + (j + 1) % columns])
			{
				alive_neighbors++;
			}
			next_step[i * columns + j] = true;
			if (previous_step[i * columns + j])
			{
				if (alive_neighbors < 2 || alive_neighbors > 3)
				{
					next_step[i * columns + j] = false;
				}
			}
			else
			{
				if (alive_neighbors != 3)
				{
					next_step[i * columns + j] = false;
				}
			}
		}
	}
}
void GameOfLifeSimulation(bool* subfield, int* sendcounts, int size, int rank, int columns)
{
	int neighbor_from_above_rank = (rank + size - 1) % size;
	int neighbor_from_below_rank = (rank + 1) % size;
	std::vector<bool*> previous_states;
	int subfield_rows = sendcounts[rank] / columns;
	int i = 0;
	bool is_looped = false;

	while (true)
	{
		MPI_Request send_to_above, send_to_below, receive_from_above, receive_from_below;
		MPI_Isend(subfield + columns, columns, MPI_C_BOOL, neighbor_from_above_rank, 1,
				  MPI_COMM_WORLD, &send_to_above);
		MPI_Isend(subfield + sendcounts[rank],
				  columns, MPI_C_BOOL, neighbor_from_below_rank,
				  0, MPI_COMM_WORLD, &send_to_below);
		MPI_Irecv(subfield, columns, MPI_C_BOOL, neighbor_from_above_rank, 0,
				  MPI_COMM_WORLD, &receive_from_above);
		MPI_Irecv(subfield + sendcounts[rank] + columns, columns, MPI_C_BOOL, neighbor_from_below_rank, 1,
				  MPI_COMM_WORLD, &receive_from_below);
		bool* auxiliary_subfield = new bool[sendcounts[rank] + 2 * columns];
		previous_states.push_back(subfield);
		if (i > 0)
		{
			bool* stop_flags = new bool[i];
			assert(stop_flags);
			SetStopFlags(stop_flags, previous_states, subfield, subfield_rows, columns);
			bool* stop_flags_matrix = new bool[i * size];
			assert(stop_flags_matrix);
			MPI_Allgather(stop_flags, i, MPI_C_BOOL,
						  stop_flags_matrix, i, MPI_C_BOOL, MPI_COMM_WORLD);
			is_looped = CheckForColumnOfOne(stop_flags_matrix, size, i);
			delete[] stop_flags;
			delete[] stop_flags_matrix;
		}
		if (is_looped)
		{
			break;
		}
		NextLifeIteration(subfield + columns, auxiliary_subfield + columns, subfield_rows, columns);
		MPI_Wait(&send_to_above, MPI_STATUSES_IGNORE);
		MPI_Wait(&receive_from_above, MPI_STATUSES_IGNORE);
		NextLifeIteration(subfield, auxiliary_subfield, 3, columns);
		MPI_Wait(&send_to_below, MPI_STATUSES_IGNORE);
		MPI_Wait(&receive_from_below, MPI_STATUSES_IGNORE);
		NextLifeIteration(subfield + (subfield_rows - 1) * columns,
						  auxiliary_subfield + (subfield_rows - 1) * columns, 3, columns);
		subfield = auxiliary_subfield;
		++i;
	}
	FreePreviousStatesVector(previous_states);
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "3 arguments are required for correct operation " << argc << " are provided" << "\n";
		std::cerr << "Usage: ./a.out <number_of_rows> <number_of_columns>\n";
		return EXIT_FAILURE;
	}
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int rows, columns;
	rows = atoi(argv[1]);
	columns = atoi(argv[1]);
	if (rows < 3 || columns < 3)
	{
		std::cerr << "Rows and columns should be at least 3\n";
		return EXIT_FAILURE;
	}
	bool* field;
	auto start = MPI_Wtime();
	if (rank == 0)
	{
		field = CreateField(rows, columns);
		PutGliderOnField(field, columns);
	}

	int* sendcounts = new int[size];
	assert(sendcounts);
	int* displs = new int[size];
	assert(displs);
	FillSendcounts(sendcounts, size, rows, columns);
	FillDispls(displs, sendcounts, size);
	bool* subfield = new bool[sendcounts[rank] + columns * 2];
	MPI_Scatterv(field,
				 sendcounts,
				 displs,
				 MPI_C_BOOL,
				 subfield + columns,
				 sendcounts[rank],
				 MPI_C_BOOL,
				 0,
				 MPI_COMM_WORLD);
	GameOfLifeSimulation(subfield, sendcounts, size, rank, columns);
	auto end = MPI_Wtime();
	if (rank == 0)
	{
		std::cout << end - start << "\n";
	}
	delete[] sendcounts;
	delete[] displs;
	if (rank == 0)
	{
		delete[] field;
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}

