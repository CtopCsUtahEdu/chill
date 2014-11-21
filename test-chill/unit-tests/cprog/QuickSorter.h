#ifndef QUICK_SORTER_H
#define QUICK_SORTER_H

#include <vector>
#include "Sorter.h"

class QuickSorter : public Sorter {
public:
    QuickSorter();
    virtual ~QuickSorter();
    virtual void sort(std::vector<int>& list) const;
};

#endif
