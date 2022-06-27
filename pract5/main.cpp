#include <iostream>
#include <cmath>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>
#include "string.h"

using namespace std;

double global_res = 0;
double L = 4000;
double mean_disbalance;
int tasks_left;
int tasks_completed;
pthread_mutex_t tasks_vector_mutex;
pthread_mutex_t tasks_left_mutex;
pthread_mutex_t current_task_mutex;
int* tasks_vector;
int my_rank;
int size;
int number_of_tasks = 2000;
int number_of_iterations;
int current_task;

int getCurrentTask()
{
	pthread_mutex_lock(&current_task_mutex);
	int ret_value = current_task;
	pthread_mutex_unlock(&current_task_mutex);
	return ret_value;
}

void setCurrentTask(int value)
{
	pthread_mutex_lock(&current_task_mutex);
	current_task = value;
	pthread_mutex_unlock(&current_task_mutex);
}

void increaseCurrentTask(int value)
{
	pthread_mutex_lock(&current_task_mutex);
	current_task += value;
	pthread_mutex_unlock(&current_task_mutex);
}

int getTasksLeft()
{
	pthread_mutex_lock(&tasks_left_mutex);
	int ret_value = tasks_left;
	pthread_mutex_unlock(&tasks_left_mutex);
	return ret_value;
}

void setTasksLeft(int value)
{
	pthread_mutex_lock(&tasks_left_mutex);
	tasks_left = value;
	pthread_mutex_unlock(&tasks_left_mutex);
}

void increaseTasksLeft(int value)
{
	pthread_mutex_lock(&tasks_left_mutex);
	tasks_left += value;
	pthread_mutex_unlock(&tasks_left_mutex);
}

int* getTasksVector()
{
	pthread_mutex_lock(&tasks_vector_mutex);
	int* ret_value = tasks_vector;
	pthread_mutex_unlock(&tasks_vector_mutex);
	return ret_value;
}

void setTasksVector(int* vector)
{
	pthread_mutex_lock(&tasks_vector_mutex);
	tasks_vector = vector;
	pthread_mutex_unlock(&tasks_vector_mutex);
}

void deleteTaskVector()
{
	pthread_mutex_lock(&tasks_vector_mutex);
	delete[] tasks_vector;
	pthread_mutex_unlock(&tasks_vector_mutex);
}

int getTasksVectorValue(int index)
{
	pthread_mutex_lock(&tasks_vector_mutex);
	int ret_value = tasks_vector[index];
	pthread_mutex_unlock(&tasks_vector_mutex);
	return ret_value;
}

void setTasksVectorValue(int value, int index)
{
	pthread_mutex_lock(&tasks_vector_mutex);
	tasks_vector[index] = value;
	pthread_mutex_unlock(&tasks_vector_mutex);
}

void zeroTakenTasks(int index, int amount)
{
	pthread_mutex_lock(&tasks_vector_mutex);
	memset(tasks_vector + index, 0, amount);
	pthread_mutex_unlock(&tasks_vector_mutex);
}

void CalcTask(int repeatNum)
{
	for(int i = 0; i < repeatNum; ++i)
	{
		global_res += sin(i);
	}
}

void FillTaskVector(int iteration_number)
{
	for(int i = 0; i < number_of_tasks; ++i)
	{
		setTasksVectorValue(abs(50 - i % 100) * abs(my_rank - (iteration_number % size)) * L + 1, i);
	}
}

void* CalculatorThread(void* args)
{
	double start, finish, time_of_one_iteration, max_iteration_time, min_iteration_time;
	double maximum_delta, disbalance;
	int sharing_opportunities;
	setTasksVector(new int[number_of_tasks]);
	setTasksLeft(number_of_tasks);
	for(int i = 0; i < number_of_iterations; ++i)
	{
		start = MPI_Wtime();
		FillTaskVector(i);
		tasks_completed = 0;
		setTasksLeft(number_of_tasks);


		for (int j = 0; j <= size; ++j)
		{
			int k;
			setCurrentTask(0);
			while(getTasksLeft() > 0)
			{
				k = getCurrentTask();
				int repeat_num = getTasksVectorValue(k);
				CalcTask(repeat_num);
				setTasksVectorValue(0, k);
				increaseTasksLeft(-1);
				tasks_completed++;
				increaseCurrentTask(1);
			}
			if (j != my_rank && j < size)
			{
				MPI_Send(&my_rank, 1, MPI_INT, j, 1, MPI_COMM_WORLD);
				MPI_Recv(&sharing_opportunities, 1, MPI_INT, j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (sharing_opportunities > 0)
				{
					MPI_Recv(getTasksVector(), sharing_opportunities, MPI_INT, j, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					setTasksLeft(sharing_opportunities);
				}
			}
		}
		finish = MPI_Wtime();
		time_of_one_iteration = finish - start;
		MPI_Allreduce(&time_of_one_iteration, &max_iteration_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&time_of_one_iteration, &min_iteration_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		maximum_delta = max_iteration_time - min_iteration_time;
		disbalance = maximum_delta / max_iteration_time;
		mean_disbalance += disbalance;
		cout << "\nProcess " << my_rank << " completed " << tasks_completed << " with result = "
			 << global_res << " and spent " << time_of_one_iteration << " for it\n";
		cout << "Disbalance = " << disbalance * 100 << "% maximum delta = " << maximum_delta << endl;
	}
	int end_signal = -1;
	MPI_Send(&end_signal, 1, MPI_INT, my_rank, 1, MPI_COMM_WORLD);
	deleteTaskVector();
	pthread_exit(nullptr);
}

void* DistributorThread(void* args)
{
	int request, response;
	int sharing_opportunities;
	double multiplier = 0.5;
	while(true)
	{
		MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		sharing_opportunities = getTasksLeft();
		if(request == -1)
		{
			break;
		}
		if(sharing_opportunities < 2)
		{
			response= 0;
			MPI_Send(&response, 1, MPI_INT, request, 2, MPI_COMM_WORLD);
		}
		else
		{
			pthread_mutex_lock(&tasks_vector_mutex);
			response= (int) (tasks_left * multiplier);
			if(response < 1)
			{
				response= tasks_left;
			}
			MPI_Send(&response, 1, MPI_INT, request, 2, MPI_COMM_WORLD);
			if(response > 0)
			{
				MPI_Send((tasks_vector + current_task + tasks_left - response), response,
						 MPI_INT,
						 request,
						 3,
						 MPI_COMM_WORLD);
			}
			pthread_mutex_unlock(&tasks_vector_mutex);
			//zeroTakenTasks(current_task + tasks_left - response, response);
			increaseTasksLeft(-response);
		}
	}
}

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if(provided != MPI_THREAD_MULTIPLE)
	{
		perror("This system does not support MPI_THREAD_MULTIPLE");
		MPI_Finalize();
		return 1;
	}
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	number_of_iterations = size;
	pthread_mutex_init(&tasks_vector_mutex, nullptr);
	pthread_mutex_init(&tasks_left_mutex, nullptr);
	pthread_mutex_init(&current_task_mutex, nullptr);
	pthread_attr_t attrs;
	if(pthread_attr_init(&attrs) != 0)
	{
		perror("Cannot initialize attributes");
		abort();
	}
	if(pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0)
	{
		perror("Error in setting attributes");
		abort();
	}
	tasks_completed = 0;
	mean_disbalance = 0;
	pthread_t pthread;
	double start = MPI_Wtime();
	if(pthread_create(&pthread, &attrs, CalculatorThread, nullptr) != 0)
	{
		perror("Cannot create a thread");
		abort();
	}
	DistributorThread(nullptr);
	pthread_attr_destroy(&attrs);
	pthread_mutex_destroy(&tasks_vector_mutex);
	pthread_mutex_destroy(&tasks_left_mutex);
	pthread_mutex_destroy(&current_task_mutex);
	if(pthread_join(pthread, nullptr) != 0)
	{
		perror("Cannot join a thread");
		abort();
	}
	if(my_rank == 0)
	{
		mean_disbalance = mean_disbalance / number_of_iterations;
		cout << "\nMean disbalance is " << mean_disbalance * 100 << "%\n Time taken = " << MPI_Wtime() - start << endl;
	}
	MPI_Finalize();
	return 0;
}

