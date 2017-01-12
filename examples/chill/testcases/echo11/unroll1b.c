


// this source derived from CHILL AST originally from file 'unroll1.c' as parsed by frontend compiler rose


void foo( int n, float *x )
{
  int t2;
  for (t2 = 1; t2 <= 7; t2 += 6) {
    x[t2] = (2.0f * ((float) t2));
    x[t2 + 1] = (2.0f * ((float) t2 + 1));
    x[t2 + 2] = (2.0f * ((float) t2 + 2));
    x[t2 + 3] = (2.0f * ((float) t2 + 3));
    x[t2 + 4] = (2.0f * ((float) t2 + 4));
    x[t2 + 5] = (2.0f * ((float) t2 + 5));
  }
  for (t2 = 13; t2 <= 15; t2 += 1) 
    x[t2] = (2.0f * ((float) t2));

}

int main(  )
{
  float  x[15];
  foo(15, x);
  return(0);

}
