#ifndef IR_ROSE_UTILS_HH
#define IR_ROSE_UTILS_HH
#include <vector>
#include "rose.h"
#include "sageBuilder.h"



std::vector<SgForStatement *> find_deepest_loops(SgNode *tnl);
std::vector<SgForStatement *> find_loops(SgNode *tnl);



SgNode* loop_body_at_level(SgNode* tnl, int level);
SgNode* loop_body_at_level(SgForStatement* loop, int level);
void swap_node_for_node_list(SgNode* tn, SgNode* new_tnl);

#endif
