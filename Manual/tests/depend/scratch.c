for(t2 = 0; t2 <= an-1; t2++) {
  for(t4 = 0; t4 <= bm-1; t4++) {
    s0(t2,t4,0);
    s1(t2,t4,0);
    for(t6 = 1; t6 <= ambn-1; t6++) {
      s1(t2,t4,t6);
    }
  }
}


for(t2 = 0; t2 <= an-1; t2++) {
  for(t4 = 0; t4 <= an-1; t4++) {
    if (t2 <= 0) {
      if (bm >= t4+1) {
        for(t6 = 0; t6 <= min(bm-1,ambn-1); t6++) {
          s0(t4,t6,t2);
          s1(t2,t4,t6);
        }
        for(t6 = ambn; t6 <= bm-1; t6++) {
          s0(t4,t6,t2);
        }
      }
      else {
        for(t6 = 0; t6 <= bm-1; t6++) {
          s0(t4,t6,t2);
        }
      }
    }
    else {
      if (bm >= t4+1) {
        for(t6 = 0; t6 <= min(ambn-1,bm-1); t6++) {
          s1(t2,t4,t6);
        }
      }
    }
    if (bm >= t4+1) {
      for(t6 = bm; t6 <= ambn-1; t6++) {
        s1(t2,t4,t6);
      }
    }
  }
  for(t4 = an; t4 <= bm-1; t4++) {
    for(t6 = 0; t6 <= ambn-1; t6++) {
      s1(t2,t4,t6);
    }
  }
}
