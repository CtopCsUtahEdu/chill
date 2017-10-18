int f(int N, int M, int *a) {
    int i, j;
    for (i = 0; i<64; ++i)
    for (j = N; j<M; ++j)
        if ((i + j) % 2 == 1)
            a[i] = 0;
}
