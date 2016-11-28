#define NZ 1666
#define NUMROWS 494

void spmv(int n, int index[NUMROWS], float a[NZ], float y[NUMROWS], float x[NUMROWS], int col[NZ]) {
	int i, j;
  
	for (i = 0; i < n; i++)
    for (j = index[i]; j < index[i+1]; j++)
      y[i] += a[j]*x[col[j]];
}


int main(){
  int n=NUMROWS;
  float a[NZ], y[NUMROWS],x[NUMROWS];
  int   index[NUMROWS+1];
  int col[NZ];
  
  spmv(n,index,a,y,x,col);
  return 0;
}
