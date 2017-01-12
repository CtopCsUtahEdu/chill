#define N 45

void foo(float* y) {

  int i;

  // strided 
  for (i = 1; i <=N; i += 3)
    y[i] = 1.0;

  return;
}

int main() {
   float y[N];

  foo(y);
  return 0;
}

