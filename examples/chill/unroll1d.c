


// this source derived from CHILL AST originally from file 'unroll1.c' as parsed by frontend compiler rose


#define N 15

void foo( int n, float *x )
{
  int t2;
  for (t2 = 1; t2 <= 11; t2 += 5) {
    x[t2] = (2.0f * ((float) t2));
    x[t2 + 1] = (2.0f * ((float) t2 + 1));
    x[t2 + 2] = (2.0f * ((float) t2 + 2));
    x[t2 + 3] = (2.0f * ((float) t2 + 3));
    x[t2 + 4] = (2.0f * ((float) t2 + 4));
  }
  return;

}

int main(  )
{
  float x[15];
  foo(15, x);
  return(0);

}
