
//#include <math.h>

extern double sqrt(double in);
//  return in/10;
//}

void ic0_csr(int n, double *val, int * row, int *col)
{
  int i, k,l,m;
  double temp;
  for (i = 0; i < n - 1; i++){
    temp = val[row[i]];
    val[row[i]] = val[row[i]]/sqrt(temp);//S1

    for (m = row[i] + 1; m < row[i+1]; m++){
      val[m] = val[m] / val[row[i]];//S2
    }

    for (m = row[i] + 1; m < row[i+1]; m++) {
      for (k = row[col[m]] ; k < row[col[m]+1]; k++){
        for ( l = m; l < row[i+1] ; l++){
          if (col[l] == col[k] ){
            if(col[l+1] <= col[k]){
              val[k] -= val[m]* val[l]; //S3
            }
          }
        }
      }
    }
  }
}

