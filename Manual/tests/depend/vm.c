void vm(float **a, float *b, float *s, int n) {
  int i, j;

  for(i = 2; i < n; i++) {
      s[i] = 0.0f;
    for(j = 1; j < i - 1; j++)
	s[i] = s[i] + b[j] * a[j][i];
    b[i] = b[i] - s[i];
  }
}

