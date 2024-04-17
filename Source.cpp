#include <iostream>
#include "omp.h"
#include <limits>

struct MinSumInfo {
    long long minSum;
    int rowIndex;
};

using namespace std;

const int arr_size = 20000;

int arr[arr_size][arr_size];

void init_arr();
long long sum(int);
long long min_sum_row(int);

int main()
{

   init_arr();

   omp_set_nested(1);
   double t1 = omp_get_wtime();
   #pragma omp parallel sections
   {
      #pragma omp section
      {
         min_sum_row(1);
         min_sum_row(2);
         min_sum_row(3);
         min_sum_row(4);
         min_sum_row(8);
         min_sum_row(10);
         min_sum_row(16);
         min_sum_row(32);
      }

      #pragma omp section
      {
         sum(1);
         sum(2);
         sum(3);
         sum(4);
         sum(8);
         sum(10);
         sum(16);
         sum(32);
      }
   }
   double t2 = omp_get_wtime();

   cout << "Total time - " << t2 - t1 << " seconds" << endl;
   return 0;
}

void init_arr()
{
   for (int i = 0; i < arr_size; i++)
   {
      for(int j = 0; j < arr_size; j++) {
         arr[i][j] = j * i - arr_size;
      }
   }
}

long long sum(int num_threads)
{
   long long sum = 0;
   double t1 = omp_get_wtime();
   #pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
   for (int i = 0; i < arr_size; i++)
   {
      for(int j = 0; j < arr_size; j++) {
         sum += arr[i][j];
      }
   }

   double t2 = omp_get_wtime();

   cout << "sum " << num_threads << " threads worked - " << t2 - t1 << " seconds result:" << sum << endl;

   return sum;
}

#pragma omp declare reduction(minSumReduction:MinSumInfo: \
   omp_out.rowIndex = omp_in.minSum < omp_out.minSum ? omp_in.rowIndex : omp_out.rowIndex, \
   omp_out.minSum = omp_in.minSum < omp_out.minSum ? omp_in.minSum : omp_out.minSum) \
   initializer(omp_priv=omp_orig)


long long min_sum_row(int num_threads)
{
   MinSumInfo minSumInfo;
   minSumInfo.minSum = INT64_MAX;
   minSumInfo.rowIndex = -1;
   int minSum = 0;
   int rowIndex = -1;
   double t1 = omp_get_wtime();

   #pragma omp parallel for num_threads(num_threads) reduction (minSumReduction:minSumInfo)
   for (int i = 0; i < arr_size; i++)
   {
      int sum = 0;

      for(int j = 0; j < arr_size; j++) {
         sum += arr[i][j];
      }

      if (minSumInfo.minSum > sum)
      {
         minSumInfo.minSum = sum;
         minSumInfo.rowIndex = i;
      }
   }

   // #pragma omp parallel for num_threads(num_threads)
   // for (int i = 0; i < arr_size; i++)
   // {
   //    int sum = 0;

   //    for(int j = 0; j < arr_size; j++) {
   //       sum += arr[i][j];
   //    }

   //    #pragma omp critical
   //    if (minSum > sum)
   //    {
   //       minSum = sum;
   //       rowIndex = i;
   //    }
   // }


   double t2 = omp_get_wtime();

   cout << "min sum is " << minSumInfo.minSum << " with index " << minSumInfo.rowIndex << ", number threads: " << num_threads << " threads worked - " << t2 - t1 << " seconds" << endl;

   return minSum;
}