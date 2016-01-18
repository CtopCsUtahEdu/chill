#define N1 4096
#define N2 4096
#define WINDOW_SIZE 16

void mpeg4_cpu(float result[N1][N2], float prev[N2+WINDOW_SIZE][N2+WINDOW_SIZE], float  curr[WINDOW_SIZE*WINDOW_SIZE])
{
	unsigned int i;
	unsigned int j;
	unsigned int k;
	unsigned int l;

	for ( i = 0; i < N1; ++i)    
		for ( j = 0; j < N2; ++j) 
                       for ( k = 0; k < WINDOW_SIZE; ++k) 
				for ( l = 0; l < WINDOW_SIZE; ++l) 
					result[i][j] += prev[i+k][j+l] * curr[k*WINDOW_SIZE+l];
				
			

		
	
}

