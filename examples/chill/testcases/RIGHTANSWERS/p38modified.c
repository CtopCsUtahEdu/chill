// this source is derived from CHILL AST originally from file 'p38.c' as parsed by frontend compiler rose

int xyz;
void zzfoo(float *A, float *B) {
  A[2] = A[1] + B[1];
  B[0] = 2.0f * B[1];
  A[3] = A[2] + B[2];
  B[1] = 2.0f * B[2];
  A[4] = A[3] + B[3];
  B[2] = 2.0f * B[3];
  return;
}
int main() {
  float x[4];
  float y[4];
  zzfoo(x, y);
  return 0;
}
