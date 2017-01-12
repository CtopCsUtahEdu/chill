


// this source derived from CHILL AST originally from file 'unroll2.c' as parsed by frontend compiler rose


void foo( float *y )
{
  int t2;
  for (t2 = 1; t2 <= 19; t2 += 18) {
    y[t2] = 1.0f;
    y[t2 + 3] = 1.0f;
    y[t2 + 6] = 1.0f;
    y[t2 + 9] = 1.0f;
    y[t2 + 12] = 1.0f;
    y[t2 + 15] = 1.0f;
  }
  for (t2 = 37; t2 <= 43; t2 += 3) 
    y[t2] = 1.0f;

}

int main(  )
{
  float  y[45];
  foo(y);
  return(0);

}
