#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 1024

void normalMM(int x[N], int a[N][N], int y[N]) {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        y[i] += a[i][j]*x[j];
}

void main()
{
	int in_x[N];
	int in_y[N];
	int in_a[N][N];

        struct timeval tstart,tend;
        int sec,usec;

        if(gettimeofday(&tstart, NULL) != 0){
          printf("Error in gettimeofday()\n");
          exit(1);
        }

	normalMM(in_x,in_a,in_y);

        if(gettimeofday(&tend, NULL) != 0){
          printf("Error in gettimeofday()\n");
          exit(1);
        }

	    // calculate run time
	if(tstart.tv_usec > tend.tv_usec){

          tend.tv_usec += 1000000;
          tend.tv_sec--;
	}
	usec = tend.tv_usec - tstart.tv_usec;
	sec = tend.tv_sec - tstart.tv_sec;

	printf("Run Time = %d sec and %d usec\n",sec,usec);

}
