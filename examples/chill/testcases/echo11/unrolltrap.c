#define N 14

void foo(int n, float* x, float* y, float* z, float* f3, float* f1, float* w) {

  int i, j;

  // trapezoid
  for (i = 0; i <= n; i++) {
    for (j = i; j <= i + n; j++)
      f3[i] = f3[i] + f1[j] * w[j - i];
    f3[i] = f3[i] * 3.14;
  }

  return;
}

int main() {
  float x[N], y[N], z[N], f3[N], f1[N], w[N];

  foo(N, x, y, z, f3, f1, w);
  return 0;
}

