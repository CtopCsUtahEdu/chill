// this source is derived from CHILL AST originally from file 'coalesce.c' as parsed by frontend compiler rose

#define c(i, j) c.count
struct inspector {
  int *i;
  int *j;
  int count;
};
#define index____(i) index[1 * i]
#define index__(i) index[1 * i + 1]
#define index___(i) index[1 * i + 1]
#define index_(i) index[1 * i]
int main() {
  inspector c;
  int t6;
  int t4;
  int t2;
  int n;
  int x[10];
  int y[10];
  int a[100];
  int col[100];
  n = 10;
  c.count = 0;
  for (t2 = 0; t2 <= n - 1; t2 += 1) 
    for (t4 = index_(t2); t4 <= index__(t2) - 1; t4 += 1) {
      c.i[c.count] = t2;
      c.j[c.count] = t4;
      c.count = c.count + 1;
    }
  for (t6 = 0; t6 <= c(t2, t4) - 1; t6 += 1) 
    x[c.i[t6]] += a[c.j[t6]] * y[col[c.j[t6]]];
  return 0;
}
