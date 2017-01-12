


// this source derived from CHILL AST originally from file 'unroll1.c' as parsed by frontend compiler rose


#define N 15

void foo( int n, float *x )
{
  x[1] = (2.0f * 1.0f);
  x[2] = (2.0f * 2.0f);
  x[3] = (2.0f * 3.0f);
  x[4] = (2.0f * 4.0f);
  x[5] = (2.0f * 5.0f);
  x[6] = (2.0f * 6.0f);
  x[7] = (2.0f * 7.0f);
  x[8] = (2.0f * 8.0f);
  x[9] = (2.0f * 9.0f);
  x[10] = (2.0f * 10.0f);
  x[11] = (2.0f * 11.0f);
  x[12] = (2.0f * 12.0f);
  x[13] = (2.0f * 13.0f);
  x[14] = (2.0f * 14.0f);
  x[15] = (2.0f * 15.0f);
  return;

}

int main(  )
{
  float x[15];
  foo(15, x);
  return(0);

}
