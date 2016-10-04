void mm(float *a, float *b, float *c, int n) {
  int i;

  for(i = 0; i < n; i++) {
    a[i + 1] = b[i];
    b[i + 1] = a[i];
    c[i] = a[i] + b[i];
  }
}
