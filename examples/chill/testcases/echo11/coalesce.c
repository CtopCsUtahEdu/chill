

int main(){
   int i,j,n;
   int x[10];
   int y[10];
   int a[100];
   int index[11];
   int col[100];
   
   n = 10;
   for(i=0; i <  n; i++)
      for(j=index[i]; j < index[i+1]; j++)
         x[i] += a[j]*y[col[j]]; 
   
   return 0;
}
