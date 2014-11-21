#include "MergeSorter.h"

/* Python
def msort(lst, start, end, pindent = 0):
    if start == end:
        return
    center = start + ((end - start) // 2)
    print(' '*pindent + "SPLIT {}|{}".format(lst[start:center+1], lst[center+1:end+1]))
    msort(lst, start, center, pindent+1)
    msort(lst, center+1, end, pindent+1)
    left = list(lst[start:center+1])
    right = list(lst[center+1:end+1])
    print(' '*pindent + "MERGE {}|{}".format(lst[start:center+1], lst[center+1:end+1]))
    i,j = 0, 0
    for k in range(start, end+1):
        if i >= len(left):
            lst[k] = right[j]
            j += 1
            print(' '*(pindent+1) + 'pull j: {} {} {}'.format(lst[start:k+1], left[i:], right[j:]))
        elif j >= len(right):
            lst[k] = left[i]
            i += 1
            print(' '*(pindent+1) + 'pull i: {} {} {}'.format(lst[start:k+1], left[i:], right[j:]))
        elif left[i] > right[j]:
            lst[k] = right[j]
            j += 1
            print(' '*(pindent+1) + 'pull j: {} {} {}'.format(lst[start:k+1], left[i:], right[j:]))
        else:
            lst[k] = left[i]
            i += 1
            print(' '*(pindent+1) + 'pull i: {} {} {}'.format(lst[start:k+1], left[i:], right[j:]))
    print(' '*pindent + "-- {}".format(lst[start:end+1]))
        

if __name__ == '__main__':
    import random as r
    x = [int(r.random()*12) for i in range(7)]
    print(x)
    msort(x, 0, len(x)-1)
    print(x)
*/

static void mergesort(std::vector<int>& lst, int start, int end) {
    if(start == end) return;
    int center = start + (end-start)/2;
    mergesort(lst, start, center);
    mergesort(lst, center+1, end);
    std::vector<int> left = std::vector<int>(lst.begin()+start, lst.begin()+(center+1));
    std::vector<int> right = std::vector<int>(lst.begin()+(center+1),lst.begin()+(end+1));
    int i = 0;
    int j = 0;
    for(int k = start; k < (end+1); k++) {
        if (i >= left.size()) {
            lst[k] = right[j++];
        }
        else if(j >= right.size()) {
            lst[k] = left[i++];
        }
        else if(left[i] > right[j]) {
            lst[k] = right[j++];
        }
        else {
            lst[k] = left[i++];
        }
    }
}

MergeSorter::MergeSorter() {
    this->name = std::string("mergesort");
}

MergeSorter::~MergeSorter() {
}

void MergeSorter::sort(std::vector<int>& list) const {
    mergesort(list, 0, list.size()-1);
}
