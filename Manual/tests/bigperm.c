void f(float *a1, float **a2, float ***a3, float ****a4, float *****a5, 
       int n1, int n2, int n3, int n4, int n5) {
    int i1, i2, i3, i4, i5;

    for (i1 = 0; i1 < n1; i1++) {
	/* a1[i1] = 0.0f; */
	for (i2 = 0; i2 < n2; i2++) {
	    /* a2[i1][i2] = 0.0f; */
	    for (i3 = 0; i3 < n3; i3++) {
		/* a3[i1][i2][i3] = 0.0f; */
		for (i4 = 0; i4 < n4; i4++) {
		    /* a4[i1][i2][i3][i4] = 0.0f; */
		    for (i5 = 0; i5 < n5; i5++)
			a5[i1][i2][i3][i4][i5] = 0.0f;
		}
	    }
	}
    }
}
