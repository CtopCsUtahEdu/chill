#ifndef SORTER_H
#define SORTER_H

#include <string>
#include <vector>

class Sorter {
public:
    Sorter();
    virtual ~Sorter();
    
    std::string name;
    virtual void sort(std::vector<int>& list) const = 0;
};

#endif
