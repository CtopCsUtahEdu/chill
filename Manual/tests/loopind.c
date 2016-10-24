void mm(float *a, float *b, int n) {
  int i;
  for(i = 0; i < n; i++) {
      a[i] = b[i];
      b[i] = a[i] + 1;
  }
}
