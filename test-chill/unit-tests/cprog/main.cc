#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "Sorter.h"
#include "QuickSorter.h"
#include "MergeSorter.h"
//#include "InsertionSorter.h"
//#include "ShellSorter.h"

void read_vector(std::vector<int>& vec, int start, int stop, char** argv) {
    for(int i = start; i < stop; i++) {
        vec.push_back((int)strtol(argv[i],NULL,0));
    }
}

void print_vector(std::vector<int>& vec) {
    printf("[");
    for(std::vector<int>::iterator iter = vec.begin(); iter != vec.end(); iter++) {
        printf(" %d ", *iter);
    }
    printf("]\n");
}

void addsorter(std::map<std::string, Sorter*>& m, Sorter* s) {
    m[s->name] = s;
}

int main(int argc, char** argv) {
    std::map<std::string, Sorter*> sorter_map;
    std::vector<int> vec;
    
    read_vector(vec, 2, argc, argv);
    print_vector(vec);
    
    addsorter(sorter_map, new QuickSorter());
    addsorter(sorter_map, new MergeSorter());
    //addsorter(sorter_map, new InsertionSorter());
    //addsorter(sorter_map, new ShellSorter());
    sorter_map[std::string(argv[1])]->sort(vec);
    print_vector(vec);
}

