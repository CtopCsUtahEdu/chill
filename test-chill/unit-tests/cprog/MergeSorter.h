#ifndef MERGE_SORTER_H
#define MERGE_SORTER_H

#include <vector>
#include "Sorter.h"

class MergeSorter : public Sorter {
public:
    MergeSorter();
    virtual ~MergeSorter();
    virtual void sort(std::vector<int>& list) const;
};

#endif
