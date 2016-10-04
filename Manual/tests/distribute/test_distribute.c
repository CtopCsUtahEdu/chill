void f(float *a, float *b, float *c, float t, int n) {
  int i;

  for(i = 0; i < n; i++) {
      if (a[i] != 0)
	  if (b[i]/a[i] <= 1.0)
	      a[i] = b[i];
      if (a[i] > t)
      	  t = (b[i] - a[i]) + t;
      else {
      	  t = (t + b[i]) - a[i];
      	  b[i] = a[i];
      }
      c[i] = b[i] + c[i];
  }
}
