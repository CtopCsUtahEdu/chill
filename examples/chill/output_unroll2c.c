


// this source derived from CHILL AST originally from file 'unroll2.c' as parsed by frontend compiler rose


#define N 45

void foo( float *y )
{
  y[1] = 1.0f;
  y[4] = 1.0f;
  y[7] = 1.0f;
  y[10] = 1.0f;
  y[13] = 1.0f;
  y[16] = 1.0f;
  y[19] = 1.0f;
  y[22] = 1.0f;
  y[25] = 1.0f;
  y[28] = 1.0f;
  y[31] = 1.0f;
  y[34] = 1.0f;
  y[37] = 1.0f;
  y[40] = 1.0f;
  y[43] = 1.0f;

}

int main(  )
{
  float y[45];
  foo(y);
  return(0);

}
