#include "QuickSorter.h"

/* Python

def swap(l, i, k):
    v = l[i]
    l[i] = l[k]
    l[k] = v
    print(str(l))

def partition(l, start, end):
    print("PARTITION {} [{}:{}]".format(l, start, end))
    p_value = l[end]
    p_index = end-1
    
    for i in range(start, end):
        while(i < p_index and l[i] >= p_value):
            swap(l, i, p_index)
            p_index -= 1
        while(i >= p_index and l[i] < p_value):
            swap(l, i, p_index)
            p_index += 1
    swap(l, p_index, end)
    print("DONE {}|[{}]|{}:{}".format(l[start:p_index], l[p_index], l[p_index+1:end+1], p_value))
    return p_index

def qsort(l, i, k):
    if i < k:
        p = partition(l, i, k)
        qsort(l,i,p-1)
        qsort(l,p+1,k)

if __name__ == "__main__":
    import random as r
    x = [int(r.random()*12) for i in range(12)]
    print(x)
    qsort(x, 0, len(x)-1)
    print(x)
    
*/

static void swap(std::vector<int>& list, int i, int k) {
    int v = list[i];
    list[i] = list[k];
    list[k] = v;
}

static int partition(std::vector<int>& list, int i, int k) {
    int pivot_value = list[k];
    int pivot_index = k - 1;
    
    for(int index = i; index < k; index++) {
        while((index < pivot_index) && (list[index] >= pivot_value)) {
            swap(list, index, pivot_index);
            pivot_index--;
        }
        while((index >= pivot_index) && (list[index] < pivot_value)) {
            swap(list, index, pivot_index);
            pivot_index++;
        }
    }
    swap(list, pivot_index, k);
    return pivot_index;
}

static void quicksort(std::vector<int>& list, int i, int k) {
    if(i < k) {
        int p = partition(list, i, k);
        quicksort(list, i, p-1);
        quicksort(list, p+1, k);
    }
}

QuickSorter::QuickSorter() {
    this->name = std::string("quicksort");
}

QuickSorter::~QuickSorter() {
}

void QuickSorter::sort(std::vector<int>& list) const {
    quicksort(list, 0, list.size()-1);
}
