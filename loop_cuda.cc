/*****************************************************************************
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   Cudaize methods

 Notes:

 History:
   1/7/10 Created by Gabe Rudy by migrating code from loop.cc
     31/1/11 Modified by Protonu Basu
*****************************************************************************/

#include <code_gen/code_gen.h>
#include <code_gen/CG_stringBuilder.h>
#include <code_gen/output_repr.h>
#include <code_gen/CG_outputRepr.h>
#include "loop_cuda.hh"
#include "loop.hh"
#include <math.h>
#include <useful.h>
#include "omegatools.hh"
#include "ir_cudasuif.hh"
#include "ir_suif.hh"
#include "ir_suif_utils.hh"
#include "chill_error.hh"
#include <vector>

using namespace omega;
char *k_cuda_texture_memory; //protonu--added to track texture memory type
char *k_cuda_constant_memory; //protonu--added to track constant memory type
//extern char *omega::k_cuda_texture_memory; //protonu--added to track texture memory type
extern char *omega::k_ocg_comment;


static int cudaDebug;
class CudaStaticInit{ public: CudaStaticInit(){ cudaDebug=0; //Change this to 1 for debug
}};
static CudaStaticInit junkInitInstance__;



std::string& upcase(std::string& s)
{
  for(int i=0; i<s.size(); i++)
    s[i] = toupper(s[i]);
  return s;
}

void printVs(const std::vector<std::string>& curOrder){
  if(!cudaDebug) return;
  for(int i=0; i<curOrder.size(); i++){
    if(i>0)
      printf(",");
    printf("%s", curOrder[i].c_str());
  }
  printf("\n");
}

void printVS(const std::vector<std::string>& curOrder){
  //if(!cudaDebug) return;
  for(int i=0; i<curOrder.size(); i++){
    if(i>0)
      printf(",");
    printf("%s", curOrder[i].c_str());
  }
  printf("\n");
}

LoopCuda::~LoopCuda() {
  const int m = stmt.size();
  for (int i = 0; i < m; i++)
    stmt[i].code->clear();
}

bool LoopCuda::symbolExists(std::string s){
  if(symtab->lookup_sym(s.c_str(), SYM_VAR, false))
    return true;
  if(globals->lookup_sym(s.c_str(), SYM_VAR, false))
    return true;
  for(int i=0; i<idxNames.size(); i++)
    for(int j=0; j<idxNames[i].size(); j++)
      if(strcmp(idxNames[i][j].c_str(), s.c_str()) == 0)
        return true;
  return false;
}

void LoopCuda::addSync(int stmt_num, std::string idxName)
{
  //we store these and code-gen inserts sync to omega comments where stmt
  //in loop that has idxName being generated
  syncs.push_back(make_pair(stmt_num,idxName));
}

void LoopCuda::renameIndex(int stmt_num, std::string idx, std::string newName)
{
  int level = findCurLevel(stmt_num, idx);
  if(idxNames.size() <= stmt_num || idxNames[stmt_num].size() < level)
    throw std::runtime_error("Invalid statment number of index");
  idxNames[stmt_num][level-1] = newName.c_str();
}



enum Type{ Int };

struct VarDefs{
  std::string name;
  std::string secondName;  
  operand size_expr; //array size as an expression (can be a product of other variables etc)
  type_node * type;
  var_sym* in_data; //Variable of array to copy data in from (before kernel call)
  var_sym* out_data; //Variable of array to copy data out to (after kernel call)
  int size_2d; //-1 if linearized, the constant size N, of a NxN 2D array otherwise
  bool tex_mapped; //protonu-- true if this variable will be texture mapped, so no need to pass it as a argument
  bool cons_mapped; //protonu-- true if this variable will be constant mem mapped, so no need to pass it as a argument
  std::string original_name; //this is such a hack, to store the original name, to store a table to textures used
  int var_ref_size ;
};

tree_node_list* wrapInIfFromMinBound(tree_node_list* then_part, tree_for* loop, base_symtab* symtab, var_sym* bound_sym)
{
  tree_node_list* ub = loop->ub_list();
  tree_node_list_iter upli(ub);
  while(!upli.is_empty()){
    tree_node *node = upli.step();
    if(node->kind() == TREE_INSTR && ((tree_instr*)node)->instr()->format() == inf_rrr)
    {
      in_rrr* ins = (in_rrr*)((tree_instr*)node)->instr();
      //expect the structure: cpy( _ = min(grab_me, _))
      if(ins->opcode() == io_cpy && ins->src1_op().is_instr()){
        ins = (in_rrr*)ins->src1_op().instr();
        if(ins->opcode() == io_min){
          tree_node_list* tnl = new tree_node_list;
          tnl->append(if_node(symtab, fold_sle(operand(bound_sym), ins->src1_op().instr()->clone()), then_part));
          return tnl;
        }
      }
    }
  }
  return then_part; //Failed to go to proper loop level
}

/**
 * This would be better if it was done by a CHiLL xformation instead of at codegen
 *
 * state:
 * for(...)
 *   for(...)
 *     cur_body
 *   stmt1
 *
 * stm1 is in-between two loops that are going to be reduced. The
 * solution is to put stmt1 at the end of cur_body but conditionally run
 * in on the last step of the for loop.
 *
 * A CHiLL command that would work better:
 *
 * for(...)
 *   stmt0
 *   for(for i=0; i<n; i++)
 *     cur_body
 *   stmt1
 * =>
 * for(...)
 *   for(for i=0; i<n; i++)
 *     if(i==0) stmt0
 *     cur_body
 *     if(i==n-1) stmt1
 */

std::vector<tree_for*> findCommentedFors(const char* index, tree_node_list* tnl){
  std::vector<tree_for *> result;
  
  tree_node_list_iter iter(tnl);
  bool next_loop_ok = false;
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    if (tn->kind() == TREE_INSTR && ((tree_instr*)tn)->instr()->opcode() == io_mrk)
    {
      instruction* inst = ((tree_instr*)tn)->instr();
      std::string comment;
      if ((inst->peek_annote(k_ocg_comment) != NULL))
      {
        immed_list *data = (immed_list *)(inst->peek_annote(k_ocg_comment));
        immed_list_iter data_iter(data);
        if(!data_iter.is_empty()){
          immed first_immed = data_iter.step();
          if(first_immed.kind() == im_string)
            comment = first_immed.string();
        }
      }
      if(comment.find("~cuda~") != std::string::npos
         && comment.find("preferredIdx: ") != std::string::npos){
        std::string idx = comment.substr(comment.find("preferredIdx: ")+14,std::string::npos);
        if(idx.find(" ") != std::string::npos)
          idx = idx.substr(0,idx.find(" "));
        if(strcmp(idx.c_str(),index) == 0)
          next_loop_ok = true;
      }
    }
    if (tn->kind() == TREE_FOR){
      if(next_loop_ok){
        //printf("found loop %s\n", static_cast<tree_for *>(tn)->index()->name());
        result.push_back(static_cast<tree_for *>(tn));
      }
      else{
        //printf("looking down for loop %s\n", static_cast<tree_for *>(tn)->index()->name());
        std::vector<tree_for*> t = findCommentedFors(index, static_cast<tree_for *>(tn)->body());
        std::copy(t.begin(), t.end(), back_inserter(result));
      }
      next_loop_ok = false;
    }
    if (tn->kind() == TREE_IF) {
      //printf("looking down if\n");
      tree_if *tni = static_cast<tree_if *>(tn);
      std::vector<tree_for*> t = findCommentedFors(index, tni->then_part());
      std::copy(t.begin(), t.end(), back_inserter(result));
    }
  }
  
  return result;
}

tree_node_list* forReduce(tree_for* loop, var_sym* reduceIndex, proc_symtab* proc_syms)
{
  //We did the replacements all at once with recursiveFindPreferedIdxs
  //replacements r;
  //r.oldsyms.append(loop->index());
  //r.newsyms.append(reduceIndex);
  //tree_for* new_loop = (tree_for*)loop->clone_helper(&r, true);
  tree_for* new_loop = loop;
  
  //return body one loops in
  tree_node_list* tnl = loop_body_at_level(new_loop, 1);
  //wrap in conditional if necessary
  tnl = wrapInIfFromMinBound(tnl, new_loop, proc_syms, reduceIndex);
  return tnl;
}

void recursiveFindRefs(tree_node_list* code, proc_symtab* proc_syms, replacements* r)
{
  if(code->parent() && code->scope()->is_block())
    ((block_symtab*)code->scope())->find_exposed_refs(proc_syms, r);
  tree_node_list_iter tnli(code);
  while (!tnli.is_empty()) {
    tree_node *node = tnli.step();
    //printf("node kind: %d\n", node->kind());
    if(node->is_instr())
    {
      tree_instr* t_instr = (tree_instr*)node;
      t_instr->find_exposed_refs(proc_syms, r);
    }
    if(node->is_block()){
      recursiveFindRefs(static_cast<tree_block *>(node)->body(), proc_syms, r);
    }
    else if(node->is_for()){
      tree_for* tn_for = static_cast<tree_for *>(node);
      //Find refs in statemetns and body
      tn_for->find_exposed_refs(proc_syms, r);
      //recursiveFindRefs(tn_for->body(), proc_syms, r);
    }
  }
}

tree_node_list* recursiveFindReplacePreferedIdxs(tree_node_list* code, proc_symtab* proc_syms,
                                                 proc_sym* cudaSync, func_type* unkown_func, 
                                                 std::map<std::string, var_sym*>& loop_idxs)
{
  tree_node_list* tnl = new tree_node_list;
  tree_node_list_iter tnli(code);
  var_sym* idxSym=0;
  bool sync = false;
  std::vector<tree_node*>      r1;
  std::vector<tree_node_list*> r2;
  while (!tnli.is_empty()) {
    tree_node *node = tnli.step();
    //printf("node kind: %d\n", node->kind());
    if(node->is_instr())
    {
      if(((tree_instr*)node)->instr()->format() == inf_rrr){
        in_rrr* inst = (in_rrr*)((tree_instr*)node)->instr();
        if(inst->opcode() == io_mrk){
          std::string comment;
          if ((inst->peek_annote(k_ocg_comment) != NULL))
          {
            immed_list *data = (immed_list *)(inst->peek_annote(k_ocg_comment));
            immed_list_iter data_iter(data);
            if(!data_iter.is_empty()){
              immed first_immed = data_iter.step();
              if(first_immed.kind() == im_string)
                comment = first_immed.string();
            }
          }
          if(comment.find("~cuda~") != std::string::npos
             && comment.find("preferredIdx: ") != std::string::npos){
            std::string idx = comment.substr(comment.find("preferredIdx: ")+14,std::string::npos);
            if(idx.find(" ") != std::string::npos)
              idx = idx.substr(0,idx.find(" "));
            //printf("sym_tab preferred index: %s\n", idx.c_str());
            if(loop_idxs.find(idx) != loop_idxs.end())
              idxSym = loop_idxs.find(idx)->second;
            //Get the proc variable sybol for this preferred index
            if(idxSym == 0){
              idxSym = (var_sym*)proc_syms->lookup_sym(idx.c_str(), SYM_VAR, false);
              //printf("idx not found: lookup %p\n", idxSym);
              if(!idxSym){
                idxSym = new var_sym(type_s32, (char*)idx.c_str());
                proc_syms->add_sym(idxSym);
                //printf("idx created and inserted\n");
              }
              //Now insert into our map for future
              loop_idxs.insert(make_pair(idx, idxSym));
            }
            //See if we have a sync as well
            if(comment.find("sync") != std::string::npos){
              //printf("Inserting sync after current block\n");
              sync = true;
            }
          }
        }
      }
      tnl->append(node);
    }
    else if(node->is_block()){
      tree_block* b = static_cast<tree_block *>(node);
      b->set_body(recursiveFindReplacePreferedIdxs(b->body(), proc_syms, cudaSync, unkown_func, loop_idxs));
      tnl->append(b);
    }
    else if(node->is_for()){
      tree_for* tn_for = static_cast<tree_for *>(node);
      if(idxSym){
        //Replace the current tn_for's index variable with idxSym
        //printf("replacing sym %s -> %s\n", tn_for->index()->name(), idxSym->name());
        replacements r;
        r.oldsyms.append(tn_for->index());
        r.newsyms.append(idxSym);
        tree_for* new_loop = (tree_for*)tn_for->clone_helper(&r, true);
        idxSym = 0; //Reset for more loops in this tnl
        new_loop->set_body(recursiveFindReplacePreferedIdxs(new_loop->body(), proc_syms, cudaSync, unkown_func, loop_idxs));
        tnl->append(new_loop);
        
        if(sync){
          in_cal *the_call =
            new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaSync))), 0);
          tnl->append(new tree_instr(the_call));
          //tnl->print();
          sync = true;
        }
      }else{
        tn_for->set_body(recursiveFindReplacePreferedIdxs(tn_for->body(), proc_syms, cudaSync, unkown_func, loop_idxs));
        tnl->append(tn_for);
      }
    }else if (node->kind() == TREE_IF) {
      tree_if *tni = static_cast<tree_if *>(node);
      tni->set_then_part(recursiveFindReplacePreferedIdxs(tni->then_part(), proc_syms, cudaSync, unkown_func, loop_idxs));
      tnl->append(tni);
    }
  }
  //Do this after the loop to not screw up the pointer interator
  /*
    for(int i=0; i<r1.size(); i++){
    swap_node_for_node_list(r1[i],r2[i]);
    }*/
  return tnl;
}

// loop_vars -> array references
// loop_idxs -> <idx_name,idx_sym> map for when we encounter a loop with a different preferredIndex
// dim_vars -> out param, fills with <old,new> var_sym pair for 2D array dimentions (messy stuff)
tree_node_list* swapVarReferences(tree_node_list* code, replacements* r, CG_suifBuilder *ocg,
                                  std::map<std::string, var_sym*>& loop_vars,
                                  proc_symtab *proc_syms,
                                  std::vector< std::pair<var_sym*,var_sym*> >& dim_vars)
{
  //Iterate over every expression, looking up each variable and type
  //reference used and possibly replacing it or adding it to our symbol
  //table
  //
  //We use the built-in cloning helper methods to seriously help us with this!
  
  //Need to do a recursive mark
  recursiveFindRefs(code, proc_syms, r);
  
  
  //We can't rely on type_node->clone() to do the heavy lifting when the
  //old type is a two dimentional array with variable upper bounds as
  //that requires creating and saveing variable references to the upper
  //bounds. So we do one pass over the oldtypes doing this type of
  //conversion, putting results in the fixed_types map for a second pass
  //to pick up.
  std::map<type_node*,type_node*> fixed_types; //array_types needing their upper bound installed
  type_node_list_iter tlip(&r->oldtypes);
  while(!tlip.is_empty())
  {
    type_node* old_tn = tlip.step();
    type_node* new_tn = 0;
    type_node* base_type = old_tn;
    std::vector< std::pair<var_sym*, type_node*> > variable_upper_bouneds;
    if(old_tn->is_ptr()){
      while (base_type->is_array() || base_type->is_ptr()) {
        if (base_type->is_array()){
          array_bound ub = ((array_type*)base_type)->upper_bound();
          if(ub.is_variable()){
            var_sym* old_ub = (var_sym*)ub.variable();
            var_sym *new_ub = proc_syms->new_unique_var(type_s32);
            dim_vars.push_back(std::pair<var_sym* , var_sym*>(old_ub, new_ub));
            variable_upper_bouneds.push_back( std::pair<var_sym*, type_node*>(new_ub, base_type) );
          }
          base_type = static_cast<array_type *>(base_type)->elem_type();
        }
        else if (base_type->is_ptr())
          base_type = static_cast<ptr_type *>(base_type)->ref_type();
      }
    }
    for (int i = variable_upper_bouneds.size()-1; i >= 0; i--) {
      var_sym *var_ub = variable_upper_bouneds[i].first;
      type_node* old_tn = variable_upper_bouneds[i].second;
      if(new_tn == 0)
        new_tn = new array_type(base_type, array_bound(1), array_bound(var_ub));
      else
        new_tn = new array_type(new_tn, array_bound(1), array_bound(var_ub));
      proc_syms->add_type(new_tn);
      fixed_types.insert(std::pair<type_node*,type_node*>(old_tn, new_tn));
    }
    if(new_tn){
      if(old_tn->is_ptr()){
        new_tn = new ptr_type(new_tn);
        proc_syms->add_type(new_tn);
      }
      fixed_types.insert(std::pair<type_node*,type_node*>(old_tn, new_tn));
    }
  }
  
  //Quickly look for modifiers on our our array types (__shared__ float [][])
  type_node_list_iter tliq(&r->oldtypes);
  while(!tliq.is_empty())
  {
    type_node* old_tn = tliq.step();
    if(old_tn->is_modifier()){
      type_node* base_type = static_cast<modifier_type *>(old_tn)->base();
      if(fixed_types.find(base_type) != fixed_types.end()){
        type_node* fixed_base = (*fixed_types.find(base_type)).second;
        //printf("Fix modifier with fixed base\n");
        //This should work to copy over the annotations, but apparently doesn't work so well
        type_node* new_tn = new modifier_type(static_cast<modifier_type*>(old_tn)->op(), fixed_base);
        old_tn->copy_annotes(new_tn);
        fixed_types.insert(std::pair<type_node*,type_node*>(old_tn, new_tn));
      }
    }
  }
  
  //Run through the types and create entries in r->newtypes but don't install
  type_node_list_iter tli(&r->oldtypes);
  while(!tli.is_empty())
  {
    type_node* old_tn = tli.step();
    type_node* new_tn = 0;
    
    //If we recorded this as fixed by our special case, use that type
    //instead of cloning.
    if(fixed_types.find(old_tn) != fixed_types.end()){
      new_tn = (*fixed_types.find(old_tn)).second;
      //printf("Reusing fixed typ %u: ", new_tn->type_id());
    }else{
      new_tn = old_tn->clone();
      //printf("Cloning type %u: ", old_tn->type_id());
    }
    new_tn = proc_syms->install_type(new_tn);
    
    //Ok, there is a weird case where an array type that has var_sym as
    //their upper bounds can't be covered fully in this loop or the
    //var_sym loop, so we need special code.
    /*
      if(old_tn->op() == TYPE_PTR && ((ptr_type*)old_tn)->ref_type()->op() == TYPE_ARRAY){
      array_type* outer_array = (array_type*)((ptr_type*)old_tn)->ref_type();
      array_bound ub = outer_array->upper_bound();
      if(ub.is_variable()){
      var_sym* old_ub = (var_sym*)ub.variable();
      var_sym* new_ub = (var_sym*)((array_type*)((ptr_type*)new_tn)->ref_type())->upper_bound().variable();
      //r->oldsyms.append(old_ub);
      fix_ub.insert(std::pair<var_sym*,array_type*>(old_ub, (array_type*)((ptr_type*)new_tn)->ref_type()));
      dim_vars.push_back(std::pair<var_sym* , var_sym*>(old_ub, new_ub));
      printf("array var_sym: %p\n", new_ub);
      }
      if(outer_array->elem_type()->op() == TYPE_ARRAY)
      {
      array_type* inner_array = (array_type*)outer_array->elem_type();
      array_bound ub = inner_array->upper_bound();
      if(ub.is_variable()){
      var_sym* old_ub = (var_sym*)ub.variable();
      var_sym* new_ub = (var_sym*)((array_type*)((array_type*)((ptr_type*)new_tn)->ref_type())->elem_type())->upper_bound().variable();
      dim_vars.push_back(std::pair<var_sym* , var_sym*>(old_ub, new_ub));
      printf("array var_sym: %p\n", new_ub);
      //r->oldsyms.append(old_ub);
      fix_ub.insert(std::pair<var_sym*,array_type*>(old_ub, (array_type*)((array_type*)((ptr_type*)new_tn)->ref_type())->elem_type()));
      }
      }
      }
    */
    r->newtypes.append(new_tn);
  }
  
  //printf("proc_syms symbol run through\n");
  //proc_syms->print();
  
  //Run through the syms creating new copies
  sym_node_list_iter snli(&r->oldsyms);
  while(!snli.is_empty())
  {
    sym_node *old_sn = snli.step();
    
    if(loop_vars.count(std::string(old_sn->name())) > 0)
    {
      r->newsyms.append(loop_vars[std::string(old_sn->name())]);
      //printf("def exists: %s\n", old_sn->name());
    }else{
      sym_node *new_sn = old_sn->copy();
      if(new_sn->is_var()){
        var_sym* var = (var_sym*)new_sn;
        type_node* new_type = var->type()->clone_helper(r);
        
        //TODO: Have a tagged list of variables to make shared
        //Make local 2D arrays __shared__
        if(new_type->op() == TYPE_ARRAY && ((array_type*)new_type)->elem_type()->op() == TYPE_ARRAY){
          //protonu--changes suggested by Malik
          //printf("Adding __shared__ annotation to : %s\n", new_sn->name());
          //new_type = ocg->ModifyType(new_type, "__shared__");
          //proc_syms->add_type(new_type);
        }
        var->set_type(new_type);
      }
      proc_syms->add_sym(new_sn);
      r->newsyms.append(new_sn);
      //printf("def new: %s\n", new_sn->name());
    }
  }
  
  //printf("proc_syms var runthrough\n");
  //proc_syms->print();
  return code->clone_helper(r);
}

bool LoopCuda::validIndexes(int stmt, const std::vector<std::string>& idxs){
  for(int i=0; i<idxs.size(); i++){
    bool found = false;
    for(int j=0; j<idxNames[stmt].size(); j++){
      if(strcmp(idxNames[stmt][j].c_str(), idxs[i].c_str()) == 0){
        found=true;
      }
    }
    if(!found){
      return false;
    }
  }
  return true;
}


bool LoopCuda::cudaize_v2(std::string kernel_name, std::map<std::string, int> array_dims,
                          std::vector<std::string> blockIdxs, std::vector<std::string> threadIdxs)
{
  int stmt_num = 0;
  if(cudaDebug){
    printf("cudaize_v2(%s, {", kernel_name.c_str());
    //for(
    printf("}, blocks={"); printVs(blockIdxs); printf("}, thread={"); printVs(threadIdxs); printf("})\n");
  }
  
  this->array_dims = array_dims;
  if(!validIndexes(stmt_num, blockIdxs)){
    throw std::runtime_error("One of the indexes in the block list was not "
                             "found in the current set of indexes.");
  }
  if(!validIndexes(stmt_num, threadIdxs)){
    throw std::runtime_error("One of the indexes in the thread list was not "
                             "found in the current set of indexes.");
  }
  if(blockIdxs.size() ==0)
    throw std::runtime_error("Cudaize: Need at least one block dimention");
  int block_level=0;
  //Now, we will determine the actual size (if possible, otherwise
  //complain) for the block dimentions and thread dimentions based on our
  //indexes and the relations for our stmt;
  for(int i=0; i<blockIdxs.size(); i++){
    int level = findCurLevel(stmt_num, blockIdxs[i]);
    int ub,lb;
    extractCudaUB(stmt_num,level,ub,lb);
    if(lb!= 0){
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if(cudaDebug) printf("Cudaize: doing tile at level %d to try and normalize lower bounds\n", level);
      tile(stmt_num,level,1,level,CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin()+(level),"");//TODO: possibly handle this for all sibling stmts
      extractCudaUB(stmt_num,level,ub,lb);
    }
    if(lb != 0){
      char buf[1024];
      sprintf(buf, "Cudaize: Loop at level %d does not have 0 as it's lower bound", level);
      throw std::runtime_error(buf);
    }
    if(ub < 0){
      char buf[1024];
      sprintf(buf, "Cudaize: Loop at level %d does not have a hard upper bound", level);
      throw std::runtime_error(buf);
    }
    if(cudaDebug) printf("block idx %s level %d lb: %d ub %d\n", blockIdxs[i].c_str(), level, lb, ub);
    if(i == 0){
      block_level = level;
      cu_bx = ub+1;
      idxNames[stmt_num][level-1] = "bx";
    }
    else if(i == 1){
      cu_by = ub+1;
      idxNames[stmt_num][level-1] = "by";
    }
  }
  if(!cu_by)
    block_level=0;
  int thread_level1 = 0;
  int thread_level2 = 0;
  for(int i=0; i<threadIdxs.size(); i++){
    int level = findCurLevel(stmt_num, threadIdxs[i]);
    int ub,lb;
    extractCudaUB(stmt_num,level,ub,lb);
    if(lb!= 0){
      //attempt to "normalize" the loop with an in-place tile and then re-check our bounds
      if(cudaDebug) printf("Cudaize: doing tile at level %d to try and normalize lower bounds\n", level);
      tile(stmt_num,level,1,level,CountedTile);
      idxNames[stmt_num].insert(idxNames[stmt_num].begin()+(level),"");
      extractCudaUB(stmt_num,level,ub,lb);
    }
    if(lb != 0){
      char buf[1024];
      sprintf(buf, "Cudaize: Loop at level %d does not have 0 as it's lower bound", level);
      throw std::runtime_error(buf);
    }
    if(ub < 0){
      char buf[1024];
      sprintf(buf, "Cudaize: Loop at level %d does not have a hard upper bound", level);
      throw std::runtime_error(buf);
    }
    
    if(cudaDebug) printf("thread idx %s level %d lb: %d ub %d\n", threadIdxs[i].c_str(), level, lb, ub);
    if(i == 0){
      thread_level1 = level;
      cu_tx = ub+1;
      idxNames[stmt_num][level-1] = "tx";
    }
    else if(i == 1){
      thread_level2 = level;
      cu_ty = ub+1;
      idxNames[stmt_num][level-1] = "ty";
    }
    else if(i == 2){
      cu_tz = ub+1;
      idxNames[stmt_num][level-1] = "tz";
    }
  }
  if(!cu_ty)
    thread_level1 = 0; 
  if(!cu_tz)
    thread_level2 = 0; 
  
  //Make changes to nonsplitlevels
  const int m = stmt.size();
  for (int i = 0; i < m; i++) {
    if(block_level){
      //stmt[i].nonSplitLevels.append((block_level)*2);
      stmt_nonSplitLevels[i].append((block_level)*2);
    }
    if(thread_level1){
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[i].append((thread_level1)*2);
    }
    if(thread_level2){
      //stmt[i].nonSplitLevels.append((thread_level1)*2);
      stmt_nonSplitLevels[i].append((thread_level1)*2);
    }
  }
  
  if(cudaDebug) {
    printf("Codegen: current names: ");
    printVS(idxNames[stmt_num]);
  }
  //Set codegen flag
  code_gen_flags |= GenCudaizeV2;
  
  //Save array dimention sizes
  this->array_dims = array_dims;
  cu_kernel_name = kernel_name.c_str();
  
}

tree_node_list* LoopCuda::cudaize_codegen_v2()
{
    //printf("cudaize codegen V2\n");
  CG_suifBuilder *ocg = dynamic_cast<CG_suifBuilder*>(ir->builder());
  if(!ocg) return false;
  
  //protonu--adding an annote to track texture memory type
  ANNOTE(k_cuda_texture_memory, "cuda texture memory", TRUE);
  ANNOTE(k_cuda_constant_memory, "cuda constant memory", TRUE);
  int tex_mem_on = 0;
  int cons_mem_on = 0;
  
  
  
  CG_outputRepr* repr;
  std::vector<VarDefs> arrayVars;
  std::vector<VarDefs> localScopedVars;
  
  std::vector<IR_ArrayRef *> ro_refs;
  std::vector<IR_ArrayRef *> wo_refs;
  std::set<std::string> uniqueRefs;
  std::set<std::string> uniqueWoRefs;
  //protonu--let's try a much simpler approach of a map instead
  //we also keep a map for constant memories
  std::map<std::string , var_sym *>tex_ref_map;
  std::map<std::string , var_sym *>cons_ref_map;
  
  for(int j=0; j<stmt.size(); j++)
  {
    std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(stmt[j].code);
    for (int i = 0; i < refs.size(); i++)
    {
      //printf("ref %s wo %d\n", static_cast<const char*>(refs[i]->name()), refs[i]->is_write());
      var_sym* var = symtab->lookup_var((char*)refs[i]->name().c_str(),false);
      //If the array is not a parameter, then it's a local array and we
      //want to recreate it as a stack variable in the kernel as opposed to
      //passing it in.
      if(!var->is_param())
        continue;
      if (uniqueRefs.find(refs[i]->name()) == uniqueRefs.end())
      {
        uniqueRefs.insert(refs[i]->name());
        if(refs[i]->is_write()){
          uniqueWoRefs.insert(refs[i]->name());
          wo_refs.push_back(refs[i]);
        }
        else
          ro_refs.push_back(refs[i]);
      }
      if (refs[i]->is_write() && uniqueWoRefs.find(refs[i]->name()) == uniqueWoRefs.end()){
        uniqueWoRefs.insert(refs[i]->name());
        wo_refs.push_back(refs[i]);
        //printf("adding %s to wo\n", static_cast<const char*>(refs[i]->name()));
      }
    }
  }
  
  // printf("reading from array ");
  // for(int i=0; i<ro_refs.size(); i++)
  //   printf("'%s' ", ro_refs[i]->name().c_str());
  // printf("and writting to array ");
  // for(int i=0; i<wo_refs.size(); i++)
  //   printf("'%s' ", wo_refs[i]->name().c_str());
  // printf("\n");
  
  const char* gridName = "dimGrid";
  const char* blockName = "dimBlock";
  
  //TODO: Could allow for array_dims_vars to be a mapping from array
  //references to to variable names that define their length.
  var_sym* dim1 = 0;
  var_sym* dim2 = 0;
  
  for(int i=0; i<wo_refs.size(); i++)
  {
    //TODO: Currently assume all arrays are floats of one or two dimentions
    var_sym* outArray = 0;
    std::string name = wo_refs[i]->name();
    outArray = symtab->lookup_var((char*)name.c_str(),false);
    
    VarDefs v;
    v.size_2d = -1;
    char buf[32];
    snprintf(buf, 32, "devO%dPtr", i+1);
    v.name = buf;
    if(outArray->type()->is_ptr())
      if(((ptr_type *)(outArray->type()))->ref_type()->is_array())
        v.type = ((array_type *)(((ptr_type *)(outArray->type()))->ref_type()))->elem_type();
      else
        v.type = ((ptr_type *)(outArray->type()))->ref_type();
    else
      v.type = type_f32;
    v.tex_mapped = false;
    v.cons_mapped = false;
    v.original_name = wo_refs[i]->name();
    //Size of the array = dim1 * dim2 * num bytes of our array type
    
    //If our input array is 2D (non-linearized), we want the actual
    //dimentions of the array
    CG_outputRepr* size;
    //Lookup in array_dims
    std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
    if(outArray->type()->is_ptr() && outArray->type()->ref_type(0)->is_array())
    {
      array_type* t = (array_type*)outArray->type()->ref_type(0);
      v.size_2d = t->upper_bound().constant()+1;
      printf("Detected 2D array sized of %d for %s\n", v.size_2d, (char*)wo_refs[i]->name().c_str());
      size = ocg->CreateInt(v.size_2d * v.size_2d);
    }else if(it != array_dims.end()){
      int ref_size = it->second;
      v.var_ref_size = ref_size;
      size = ocg->CreateInt(ref_size);
    }
    else{
      if(dim1){
        size = ocg->CreateTimes(new CG_suifRepr(operand(dim1)),
                                new CG_suifRepr(operand(dim2)));
      }else{
        char buf[1024];
        sprintf(buf, "CudaizeCodeGen: Array reference %s does not have a "
                "detectable size or specififed dimentions", name.c_str());
        throw std::runtime_error(buf);
      }
    }
    v.size_expr = operand(static_cast<CG_suifRepr*>(ocg->CreateTimes(
                                                      size,
                                                      ocg->CreateInt(v.type->size()/8)))->GetExpression());
    v.in_data = 0;
    v.out_data = outArray;
    //Check for in ro_refs and remove it at this point
    std::vector<IR_ArrayRef *>::iterator it_;
    for(it_ = ro_refs.begin(); it_ != ro_refs.end(); it_++)
    {
      if((*it_)->name() == wo_refs[i]->name()){
        break;
      }
    }
    if(it_ != ro_refs.end())
    {
      v.in_data = outArray;
      ro_refs.erase(it_);
    }
    
    arrayVars.push_back(v);
    
  }
  
  //protonu-- assuming that all texture mapped memories were originally read only mems
  //there should be safety checks for that, will implement those later
  
  int cs_ref_size = 0;
  
  for(int i=0; i<ro_refs.size(); i++)
  {
    var_sym* inArray = 0;
    std::string name = ro_refs[i]->name();
    inArray = symtab->lookup_var((char*)name.c_str(),false);
    VarDefs v;
    v.size_2d = -1;
    char buf[32];
    snprintf(buf, 32, "devI%dPtr", i+1);
    v.name = buf;
    if(inArray->type()->is_ptr())
      if(((ptr_type *)(inArray->type()))->ref_type()->is_array())
        v.type = ((array_type *)(((ptr_type *)(inArray->type()))->ref_type()))->elem_type();
      else
        v.type = ((ptr_type *)(inArray->type()))->ref_type(); 
    else
      v.type = type_f32;
    v.tex_mapped = false;
    v.cons_mapped = false;
    v.original_name = ro_refs[i]->name();
    if ( texture != NULL)
      v.tex_mapped = (texture->is_array_tex_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    if (v.tex_mapped){
      printf("this variable  %s is mapped to texture memory", name.c_str());
    }
    if ( constant_mem != NULL)
      v.cons_mapped = (constant_mem->is_array_cons_mapped(name.c_str()))? true:false; //protonu-track tex mapped vars
    if (v.cons_mapped){
      printf("this variable  %s is mapped to constant memory", name.c_str());
    }
    
    //Size of the array = dim1 * dim2 * num bytes of our array type
    
    //If our input array is 2D (non-linearized), we want the actual
    //dimentions of the array (as it might be less than cu_n
    CG_outputRepr* size;
    //Lookup in array_dims
    std::map<std::string, int>::iterator it = array_dims.find(name.c_str());
    int ref_size = 0;
    if(inArray->type()->is_ptr() && inArray->type()->ref_type(0)->is_array())
    {
      array_type* t = (array_type*)inArray->type()->ref_type(0);
      v.size_2d = t->upper_bound().constant()+1;
      printf("Detected 2D array sized of %d for %s\n", v.size_2d, (char*)ro_refs[i]->name().c_str());
      size = ocg->CreateInt(v.size_2d * v.size_2d);
    }else if(it != array_dims.end()){
      ref_size = it->second;
      v.var_ref_size = ref_size;
      size = ocg->CreateInt(ref_size);
    }else{
      if(dim1){
        size = ocg->CreateTimes(new CG_suifRepr(operand(dim1)),
                                new CG_suifRepr(operand(dim2)));
      }else{
        char buf[1024];
        sprintf(buf, "CudaizeCodeGen: Array reference %s does not have a "
                "detectable size or specififed dimentions", name.c_str());
        throw std::runtime_error(buf);
      }
    }
    
    
    
    v.size_expr = operand(static_cast<CG_suifRepr*>(ocg->CreateTimes(
                                                      size,
                                                      ocg->CreateInt(v.type->size()/8)))->GetExpression());
    
    v.in_data = inArray;
    v.out_data = 0;
    arrayVars.push_back(v);
  }
  
  
  if(arrayVars.size() < 2)
  {
    fprintf(stderr, "cudaize error: Did not find two arrays being accessed\n");
    return false;
  }
  
  //protonu--debugging tool--the printf statement
  //tex_mem_on signals use of tex mem
  for(int i=0; i<arrayVars.size(); i++)
  {
    //printf("var name %s, tex_mem used %s\n", arrayVars[i].name.c_str(), (arrayVars[i].tex_mapped)?"true":"false");
    if (arrayVars[i].tex_mapped  ) tex_mem_on ++;
    if (arrayVars[i].cons_mapped  ) cons_mem_on ++;
  }
  
  //Add CUDA function extern prototypes and function types
  func_type* unkown_func = new func_type(type_s32); //function on unkown args that returns a i32
  unkown_func = (func_type*)symtab->install_type(unkown_func);
  func_type* void_func = new func_type(type_void); //function on unkown args that returns a void
  void_func = (func_type*)globals->install_type(void_func);
  func_type* float_func = new func_type(type_f32); //function on unkown args that returns a float
  float_func = (func_type*)globals->install_type(float_func);
  
  type_node* result = ocg->ModifyType(type_void, "__global__");
  result = globals->install_type(result);
  func_type* kernel_type = new func_type(result); //function returns a '__global__ void'
  
  int numArgs =  arrayVars.size() + (dim1 ? 2 : 0) + localScopedVars.size();
  //protonu--need to account for texture memory here, reduce the #args
  if( tex_mem_on ) numArgs -= tex_mem_on;
  if( cons_mem_on ) numArgs -= cons_mem_on;
  kernel_type->set_num_args(numArgs);
  int argCount = 0;
  for(int i=0; i<arrayVars.size(); i++)
  {
    type_node* fptr;
    if(arrayVars[i].in_data)
      fptr = arrayVars[i].in_data->type()->clone();
    else
      fptr = arrayVars[i].out_data->type()->clone();
    //protonu--skip this for texture mems
    if( arrayVars[i].tex_mapped != true && arrayVars[i].cons_mapped !=true )
      kernel_type->set_arg_type(argCount++, fptr);
  }
  if(dim1){
    kernel_type->set_arg_type(argCount++, type_s32); //width x height dimentions
    kernel_type->set_arg_type(argCount++, type_s32);
  }
  kernel_type = (func_type*)globals->install_type(kernel_type);
  
  proc_sym* cudaMalloc = globals->new_proc(unkown_func, src_c, "cudaMalloc");
  proc_sym* cudaMemcpy = globals->new_proc(unkown_func, src_c, "cudaMemcpy");
  proc_sym* cudaFree = globals->new_proc(unkown_func, src_c, "cudaFree");
  proc_sym* cudaSync = globals->new_proc(void_func, src_c, "__syncthreads");
  proc_sym* cudaBind = globals->new_proc(unkown_func, src_c, "cudaBindTexture");
  proc_sym* cudaMemcpySym = globals->new_proc(unkown_func, src_c, "cudaMemcpyToSymbol");
  
  
  //protonu-removing Gabe's function, introducing mine, this is pretty cosmetic
  //proc_sym* cudaFetch = globals->new_proc(float_func, src_c, "tex1Dfetch");
  proc_sym* tex1D = globals->new_proc(float_func, src_c, "tex1Dfetch");
  
  var_sym *cudaMemcpyHostToDevice = new var_sym(type_s32, "cudaMemcpyHostToDevice");
  var_sym *cudaMemcpyDeviceToHost = new var_sym(type_s32, "cudaMemcpyDeviceToHost");
  cudaMemcpyDeviceToHost->set_param();
  cudaMemcpyHostToDevice->set_param();
  globals->add_sym(cudaMemcpyHostToDevice);
  globals->add_sym(cudaMemcpyDeviceToHost);
  
  //protonu--adding the bool tex_mem to the structure struct_type
  //to bypass the re-naming of struct texture, this is a hack fix
  struct_type* texType = new struct_type(TYPE_GROUP, 0, "texture<float, 1, cudaReadModeElementType>", 0, true);
  immed_list *iml_tex = new immed_list;
  iml_tex->append(immed("texture memory"));
  texType->append_annote(k_cuda_texture_memory, iml_tex);
  //protonu--end my changes
  texType = (struct_type*)globals->install_type(texType);
  //protonu--should register the locals later on
  //when we do the bind operation
  //var_sym* texRef = new var_sym(texType, "texRef");
  //globals->add_sym(texRef);
  
  //Add our mallocs (and input array memcpys)
  for(int i=0; i<arrayVars.size(); i++)
  {
    //protonu--check if the variable is not a tex-mapped variable. If it is tex mapped
    // allow a malloc and memcpy operation, and a bind, but only if it is tex mapped, but dont call
    // the kernel with it as an argument.
    
    //Make a pointer of type a[i].type
    //type_node* fptr = new ptr_type(arrayVars[i].type->clone());
    //protonu--temporary change 
    type_node* fptr = new ptr_type(arrayVars[i].type);
    fptr = symtab->install_type(fptr);
    var_sym *dvs = new var_sym(fptr, const_cast<char*>(
                                 arrayVars[i].name.c_str()));
    dvs->set_addr_taken();
    symtab->add_sym(dvs);
    
    //cudaMalloc args
    //protonu--no cudaMalloc required for constant memory
    tree_node_list* tnl = new tree_node_list;
    if(arrayVars[i].cons_mapped != true )
    {
      in_cal *the_call =
        new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaMalloc))), 2);
      the_call->set_argument(0, operand(new in_ldc(type_void->ptr_to()->ptr_to(), operand(), immed(dvs))));
      the_call->set_argument(1, arrayVars[i].size_expr);
      
      tnl->append(new tree_instr(the_call));
      setup_code = ocg->StmtListAppend(setup_code,
                                       new CG_suifRepr(tnl));
    }
    if(arrayVars[i].in_data)
    {
      //cudaMemcpy args
      //protonu-- no cudaMemcpy required for constant memory
      if ( arrayVars[i].cons_mapped != true )
      {
        in_cal *the_call =
          new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaMemcpy))), 4);
        the_call->set_argument(0, operand(dvs));
        the_call->set_argument(1, operand(arrayVars[i].in_data));
        the_call->set_argument(2, arrayVars[i].size_expr.clone());
        the_call->set_argument(3, operand(cudaMemcpyHostToDevice));
        
        tnl = new tree_node_list;
        tnl->append(new tree_instr(the_call));
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_suifRepr(tnl));
      }
      
      //protonu--check if the arrayvar is tex mapped
      if(arrayVars[i].tex_mapped == true)
      {
        //Need a texture reference variable
        char buf[32];
        snprintf(buf, 32, "tex%dRef", i+1);
        arrayVars[i].secondName = buf;
        
        var_sym* texRef = new var_sym(texType, buf);
        //printf("\n putting in %s\n", arrayVars[i].original_name.c_str());
        tex_ref_map[arrayVars[i].original_name] = texRef;
        globals->add_sym(texRef);
        //protonu--added the above two lines
        
        in_cal *the_call =
          new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaBind))), 4);
        in_ldc *ins = new in_ldc(type_s32, operand(), immed(0));
        the_call->set_argument(0, operand(ins));
        the_call->set_argument(1, operand(texRef));//protonu--change to add the new sym
        the_call->set_argument(2, operand(dvs));
        the_call->set_argument(3, arrayVars[i].size_expr.clone());
        
        tnl = new tree_node_list;
        tnl->append(new tree_instr(the_call));
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_suifRepr(tnl));
      }
      
      //protonu--if arrayvar is mapped to constant memory
      if(arrayVars[i].cons_mapped == true)
      {
        char buf[32];
        snprintf(buf, 32, "cs%dRef", i+1);
        //arrayVars[i].secondName = buf;
        array_bound low (0);
        array_bound high (arrayVars[i].var_ref_size -1);
        array_type *arr = new array_type(arrayVars[i].type,low, high);
        type_node* cons_arr = ocg->ModifyType(arr, "__device__ __constant__");
        cons_arr = globals->install_type(cons_arr);
        var_sym* consRef = new var_sym(cons_arr, buf);
        cons_ref_map[arrayVars[i].original_name] = consRef;
        globals->add_sym(consRef);
        
        
        
        in_cal *the_call =
          new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaMemcpySym))), 3);
        the_call->set_argument(0, operand(new in_ldc(type_void->ptr_to(), operand(), immed(consRef))));
        the_call->set_argument(1, operand(arrayVars[i].in_data));
        the_call->set_argument(2, arrayVars[i].size_expr.clone());
        
        tnl = new tree_node_list;
        tnl->append(new tree_instr(the_call));
        setup_code = ocg->StmtListAppend(setup_code,
                                         new CG_suifRepr(tnl));
        
      }
    }
  }
  
  //Build dimGrid dim3 variables based on loop dimentions and ti/tj
  char blockD1[120];
  char blockD2[120];
  if(dim1){
    snprintf(blockD1, 120, "%s/%d", dim1->name(), cu_tx);
    snprintf(blockD2, 120, "%s/%d", dim2->name(), cu_ty);
  }else{
    snprintf(blockD1, 120, "%d", cu_bx);
    snprintf(blockD2, 120, "%d", cu_by);
    //snprintf(blockD1, 120, "%d/%d", cu_nx, cu_tx);
    //snprintf(blockD2, 120, "%d/%d", cu_ny, cu_ty);
  }
  repr = ocg->CreateDim3(immed((char*)gridName),
                         immed(blockD1),
                         immed(blockD2));
  setup_code = ocg->StmtListAppend(setup_code, repr);
  
  repr = ocg->CreateDim3(immed((char*)blockName), immed(cu_tx),immed(cu_ty));
  
  if(cu_tz > 1)
    repr = ocg->CreateDim3(immed((char*)blockName), immed(cu_tx), immed(cu_ty), immed(cu_tz));
  else
    repr = ocg->CreateDim3(immed((char*)blockName), immed(cu_tx), immed(cu_ty));
  setup_code = ocg->StmtListAppend(setup_code, repr);
  
  //call kernel function with name loop_name
  //like: transpose_k<<<dimGrid,dimBlock>>>(devOPtr, devIPtr , width, height);
  char dims[120];
  snprintf(dims,120,"<<<%s,%s>>>",gridName, blockName);
  immed_list *iml = new immed_list;
  iml->append(immed((char*)cu_kernel_name.c_str()));
  iml->append(immed(dims));
  //printf("%s %s\n", static_cast<const char*>(cu_kernel_name), dims);
  for(int i=0; i<arrayVars.size(); i++)
    //Throw in a type cast if our kernel takes 2D array notation
    //like (float(*) [1024])
  {
    //protonu--throwing in another hack to stop the caller from passing tex mapped 
    //vars to the kernel.
    if(arrayVars[i].tex_mapped == true || arrayVars[i].cons_mapped == true )
      continue;     
    if(arrayVars[i].size_2d >= 0)
    {
      snprintf(dims,120,"(float(*) [%d])%s", arrayVars[i].size_2d,
               const_cast<char*>(arrayVars[i].name.c_str()));
      //printf("%d %s\n", i, dims);
      iml->append(immed(dims));
    }else{
      //printf("%d %s\n", i, static_cast<const char*>(arrayVars[i].name));
      iml->append(immed(const_cast<char*>(
                          arrayVars[i].name.c_str())));
    }
  }
  if(dim1){
    iml->append(immed(dim1));
    iml->append(immed(dim2));
  }
  repr = ocg->CreateKernel(iml);//kernel call
  setup_code = ocg->StmtListAppend(setup_code, repr);
  
  //cuda free variables
  for(int i=0; i<arrayVars.size(); i++)
  {
    if(arrayVars[i].out_data)
    {
      //cudaMemcpy args
      in_cal *the_call =
        new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaMemcpy))), 4);
      the_call->set_argument(0, operand(arrayVars[i].out_data));
      the_call->set_argument(1, operand(symtab->lookup_var(const_cast<char*>(
                                                             arrayVars[i].name.c_str()))));
      the_call->set_argument(2, arrayVars[i].size_expr.clone());
      the_call->set_argument(3, operand(cudaMemcpyDeviceToHost));
      
      tree_node_list* tnl = new tree_node_list;
      tnl->append(new tree_instr(the_call));
      teardown_code = ocg->StmtListAppend(teardown_code,
                                          new CG_suifRepr(tnl));
    }
    
    in_cal *the_call =
      new in_cal(type_s32, operand(), operand(new in_ldc(unkown_func->ptr_to(), operand(), immed(cudaFree))), 1);
    the_call->set_argument(0, operand(symtab->lookup_var(const_cast<char*>(
                                                           arrayVars[i].name.c_str()))));
    
    tree_node_list* tnl = new tree_node_list;
    tnl->append(new tree_instr(the_call));
    teardown_code = ocg->StmtListAppend(teardown_code,
                                        new CG_suifRepr(tnl));
  }
  
  // ---------------
  // BUILD THE KERNEL
  // ---------------
  
  //Extract out kernel body
  tree_node_list* code = getCode();
  //Get rid of wrapper if that original() added
  if(code->head()->contents->kind() == TREE_IF)
  {
    tree_if* ifn = (tree_if*)code->head()->contents;
    code = ifn->then_part();
  }
  
  //Create kernel function body
  proc_sym *new_psym = globals->new_proc(kernel_type, src_c, (char*)cu_kernel_name.c_str());
  proc_symtab *new_proc_syms = new proc_symtab(new_psym->name());
  globals->add_child(new_proc_syms);
  
  //Add Params
  std::map<std::string, var_sym*> loop_vars;
  //In-Out arrays 
  type_node* fptr;
  for(int i=0; i<arrayVars.size(); i++)
  {
    if(arrayVars[i].in_data)
      //fptr = arrayVars[i].in_data->type()->clone();
      fptr = arrayVars[i].in_data->type();
    else
      //fptr = arrayVars[i].out_data->type()->clone();
      fptr = arrayVars[i].out_data->type();
    fptr = new_proc_syms->install_type(fptr);
    std::string name = arrayVars[i].in_data ? arrayVars[i].in_data->name() : arrayVars[i].out_data->name();
    var_sym* sym = new var_sym(fptr, arrayVars[i].in_data ? arrayVars[i].in_data->name() : arrayVars[i].out_data->name());
    //protonu--adding a check to ensure that texture memories are not passed in as arguments
    if(arrayVars[i].tex_mapped != true     && arrayVars[i].cons_mapped !=true  ) 
    {
      sym->set_param();
      new_proc_syms->params()->append(sym);
      new_proc_syms->add_sym(sym);//protonu--added to suppress the addition of the redundant var in the kernel
    }
    if (arrayVars[i].cons_mapped == true)
    {       
      sym->set_param();
      new_proc_syms->add_sym(sym);
    }
    //printf("inserting name: %s\n", static_cast<const char*>(name));
    loop_vars.insert(std::pair<std::string, var_sym*>(std::string(name), sym));
  }
  
  if(dim1)
  {
    //Array dimentions
    var_sym* kdim1 = new var_sym(dim1->type(), dim1->name());
    kdim1->set_param();
    new_proc_syms->add_sym(kdim1);
    loop_vars.insert(std::pair<std::string, var_sym*>(std::string(dim1->name()), kdim1));
    var_sym* kdim2 = new var_sym(dim2->type(), dim2->name());
    kdim2->set_param();
    new_proc_syms->add_sym(kdim2);
    loop_vars.insert(std::pair<std::string, var_sym*>(std::string(dim2->name()), kdim2));
    new_proc_syms->params()->append(kdim1);
    new_proc_syms->params()->append(kdim2);
  }
  //Put block and thread implicit variables into scope
  std::vector<var_sym *> index_syms;
  /* Currently we don't use the block dimentions
     var_sym* blockDim_x = new var_sym(type_s32, "blockDim.x");
     blockDim_x->set_param();
     new_proc_syms->add_sym(blockDim_x);
     var_sym* blockDim_y = new var_sym(type_s32, "blockDim.y");
     blockDim_y->set_param();
     new_proc_syms->add_sym(blockDim_y);
  */
  if(cu_bx > 1){
    var_sym* blockIdx_x = new var_sym(type_s32, "blockIdx.x");
    blockIdx_x->set_param();
    new_proc_syms->add_sym(blockIdx_x);
    index_syms.push_back(blockIdx_x);
  }
  if(cu_by > 1){
    var_sym* blockIdx_y = new var_sym(type_s32, "blockIdx.y");
    blockIdx_y->set_param();
    new_proc_syms->add_sym(blockIdx_y);
    index_syms.push_back(blockIdx_y);
  }
  if(cu_tx > 1){
    var_sym* threadIdx_x = new var_sym(type_s32, "threadIdx.x");
    threadIdx_x->set_param();
    new_proc_syms->add_sym(threadIdx_x);
    index_syms.push_back(threadIdx_x);
  }
  if(cu_ty > 1){
    var_sym* threadIdx_y = new var_sym(type_s32, "threadIdx.y");
    threadIdx_y->set_param();
    new_proc_syms->add_sym(threadIdx_y);
    index_syms.push_back(threadIdx_y);
  }
  
  if(cu_tz > 1){
    var_sym* threadIdx_z = new var_sym(type_s32, "threadIdx.z");
    threadIdx_z->set_param();
    new_proc_syms->add_sym(threadIdx_z);
    index_syms.push_back(threadIdx_z);
  }
  
  //Figure out which loop variables will be our thread and block dimention variables
  std::vector<var_sym *> loop_syms;
  //Get our indexes
  std::vector<const char*> indexes;// = get_loop_indexes(code,cu_num_reduce);
  int threadsPos=0;
  if(cu_bx > 1)
    indexes.push_back("bx");
  if(cu_by > 1)
    indexes.push_back("by");
  if(cu_tx > 1){
    threadsPos = indexes.size();
    indexes.push_back("tx");
  }
  if(cu_ty > 1)
    indexes.push_back("ty");
  if(cu_tz > 1)
    indexes.push_back("tz");
  for(int i=0; i<indexes.size(); i++)
  {
    //printf("indexes[%d] = %s\n", i, (char*)indexes[i]);
    loop_syms.push_back(new var_sym(type_s32, (char*)indexes[i]));
    new_proc_syms->add_sym(loop_syms[i]);
    //loop_vars.insert(std::pair<std::string, var_sym*>(std::string(indexes[i]), loop_syms[i]));
  }
  
  //Generate this code
  //int bx = blockIdx.x
  //int by = blockIdx.y
  //int tx = threadIdx.x
  //int ty = threadIdx.y
  CG_outputRepr *body=NULL;
  for(int i=0; i<indexes.size(); i++){
    CG_outputRepr *lhs = new CG_suifRepr(operand(loop_syms[i]));
    //body = ocg->StmtListAppend(body, ocg->CreateStmtList(
    //                             ocg->CreateAssignment(0, lhs, new CG_suifRepr(operand(index_syms[i])))));
    body = ocg->StmtListAppend(body, ocg->StmtListAppend(
                                 ocg->CreateAssignment(0, lhs, new CG_suifRepr(operand(index_syms[i]))), NULL));
  }
  
  //Get our inital code prepped for loop reduction. First we need to swap
  //out internal SUIF variable references to point to the new local
  //function symbol table.
  std::map<std::string, var_sym*> loop_idxs; //map from idx names to their new syms
  std::vector< std::pair<var_sym*, var_sym*> > dim_vars; //pair is of <old,new> var_sym (for 2D array size initializations)
  replacements r;
  tree_node_list* swapped = swapVarReferences(code, &r, ocg, loop_vars, new_proc_syms, dim_vars);
  //printf("\n code before recursiveFindReplacePreferedIdxs :\n");
  //swapped->print();
  swapped = recursiveFindReplacePreferedIdxs(swapped, new_proc_syms, cudaSync, void_func, loop_idxs);//in-place swapping
  //printf("\n code after recursiveFindReplacePreferedIdxs :\n");
  //swapped->print();
  
  for(int i=0; i<indexes.size(); i++){
    std::vector<tree_for*> tfs = findCommentedFors(indexes[i], swapped);
    for(int k=0; k<tfs.size(); k++){
      //printf("replacing %p tfs for index %s\n", tfs[k], indexes[i]);
      tree_node_list* newBlock = forReduce(tfs[k], loop_idxs[indexes[i]], new_proc_syms);
      //newBlock->print();
      swap_node_for_node_list(tfs[k], newBlock);
      //printf("AFTER SWAP\n");        newBlock->print();
    }
  }
  //printf("AFTER REDUCE\n"); swapped->print();
  
  if(static_cast<const IR_cudasuifCode *>(ir)->init_code()){
    tree_node_list* orig_init_code = static_cast<CG_suifRepr *>(static_cast<const IR_cudasuifCode *>(ir)->init_code())->GetCode();
    for(int i=0; i<dim_vars.size(); i++){
      //We have a map of var_sym from the original function body and we know
      //that these var_syms have initialization statements which define the
      //array size. We need to mimic these initialization statements.
      
      //First find the assignment and pull out the constant initialization
      //value
      int value = -1;
      tree_node_list_iter tnli(orig_init_code);
      while (!tnli.is_empty()) {
        tree_node *node = tnli.step();
        if(node->kind() == TREE_INSTR && ((tree_instr*)node)->instr()->format() == inf_rrr)
        {
          in_rrr* inst = (in_rrr*)((tree_instr*)node)->instr();
          //expect the structure: cpy( _ = min(grab_me, _))
          if(inst->opcode() == io_cpy && inst->dst_op().is_symbol()){
            //printf("looking at instruction: ");
            //inst->print();
            var_sym* dest = inst->dst_op().symbol();
            if(dest == dim_vars[i].first)
            {
              if(inst->src1_op().is_instr() && inst->src1_op().instr()->format() == inf_ldc){
                value = ((in_ldc*)inst->src1_op().instr())->value().integer();
              }
            }
          }
        }
      }
      if(value < 0){
        fprintf(stderr, "ERROR: Could not find initializing statement for variable used in upper_bound of array type");
      }
      CG_outputRepr *lhs = new CG_suifRepr(operand(dim_vars[i].second));
      //body = ocg->StmtListAppend(body, ocg->CreateStmtList(ocg->CreateAssignment(0, lhs, ocg->CreateInt(value))));
      body = ocg->StmtListAppend(body, ocg->StmtListAppend(ocg->CreateAssignment(0, lhs, ocg->CreateInt(value)), NULL));
    }
  }
  
  
  body = ocg->StmtListAppend(body, new CG_suifRepr(swapped));
  
  //protonu--lets try creating our function definiton here
  var_sym *tsym = NULL;
  
  
  std::vector<IR_ArrayRef *> refs = ir->FindArrayRef(body);
  for(int i=0; i<refs.size(); i++)
  {
    //check if the array is tex mapped
    if(texture != NULL && texture->is_array_tex_mapped(refs[i]->name().c_str()))
    {
      //protonu--our new tex lookup function
      in_cal *tex_lookup =
        new in_cal(type_f32, operand(), operand(new in_ldc(float_func->ptr_to(), operand(), immed(tex1D))), 2);
      
      //printf("name of the array to be mapped is %s\n", refs[i]->name().c_str());
      tsym = tex_ref_map[refs[i]->name()];
      tex_lookup->set_argument(0, operand(tsym));
      
      
      int array_dims = ((IR_suifArrayRef *)refs[i])->ia_->dims();
      
      if (array_dims == 1){ 
        tex_lookup->set_argument(1, ((IR_suifArrayRef *)refs[i])->ia_->index(0).clone());
      }else if (array_dims > 2) {
        printf(" \n we don't handle more than 2D arrays mapped to textures yet\n");
      }else if (array_dims == 2) {
        
        IR_ArraySymbol *sym = refs[i]->symbol();
        CG_outputRepr *sz = sym->size(1);
        delete sym;  // free the wrapper object only
        // find the builder ocg
        CG_outputRepr *expr = ocg->CreateTimes(sz->clone(),refs[i]->index(0));
        delete sz; // free the wrapper object only
        expr = ocg->CreatePlus(expr, refs[i]->index(1));
        // expr holds the 1D access expression and take it out
        tex_lookup->set_argument(1, ((CG_suifRepr *)expr)->GetExpression());
      }
      
      //using chun's function to replace the array look up with the function call
      ((IR_suifCode *)ir)->ReplaceExpression(refs[i] , new CG_suifRepr(operand(tex_lookup)));
    }
    
  }
  
  
  tsym = NULL;
  //protonu--now let's try what we did above for constant memory
  for(int i=0; i<refs.size(); i++)
  {
    //check if the array is tex mapped
    if(constant_mem != NULL && constant_mem->is_array_cons_mapped(refs[i]->name().c_str()))
    {
      
      //printf("name of the array to be cons mapped is %s\n", refs[i]->name().c_str());
      tsym = cons_ref_map[refs[i]->name()];
      //we should create a IR_SuifArray here
      IR_ArraySymbol *ar_sym = new IR_suifArraySymbol(ir,tsym);
      std::vector<CG_outputRepr *> ar_index;
      ar_index.push_back(((IR_suifArrayRef *)refs[i])->index(0));
      IR_ArrayRef *ar_ref = ((IR_suifCode *)ir)->CreateArrayRef(ar_sym, ar_index);
      //using chun's function to replace the array look up with the function call
      ((IR_suifCode *)ir)->ReplaceExpression(refs[i] , new CG_suifRepr(operand(((IR_suifArrayRef *)ar_ref)->ia_)));
      
    }
  }
  
  
  tree_proc *new_body = new tree_proc(static_cast<CG_suifRepr*>(body)->GetCode(), new_proc_syms);
  //globals->add_child(new_proc_syms);
  new_psym->set_block(new_body);
  new_procs.push_back(new_psym);
  
  return swapped;
}

//Order taking out dummy variables
std::vector<std::string> cleanOrder(std::vector<std::string> idxNames){
  std::vector<std::string> results;
  for(int j=0; j<idxNames.size(); j++){
    if(idxNames[j].length() != 0)
      results.push_back(idxNames[j]);
  }
  return results;
}

//First non-dummy level in ascending order
int LoopCuda::nonDummyLevel(int stmt, int level){
  //level comes in 1-basd and should leave 1-based
  for(int j=level-1; j<idxNames[stmt].size(); j++){
    if(idxNames[stmt][j].length() != 0){
      //printf("found non dummy level of %d with idx: %s when searching for %d\n", j+1, (const char*) idxNames[stmt][j], level);
      return j+1;
    }
  }
  char buf[128]; sprintf(buf, "%d", level);
  throw std::runtime_error(std::string("Unable to find a non-dummy level starting from ") + std::string(buf));
}

int LoopCuda::findCurLevel(int stmt, std::string idx){
  for(int j=0; j<idxNames[stmt].size(); j++){
    if(strcmp(idxNames[stmt][j].c_str(),idx.c_str()) == 0)
      return j+1;
  }
  throw std::runtime_error(std::string("Unable to find index ") + idx + std::string(" in current list of indexes"));
}

void LoopCuda::permute_cuda(int stmt, const std::vector<std::string>& curOrder)
{
  //printf("curOrder: ");
  //printVs(curOrder);
  //printf("idxNames: ");
  //printVS(idxNames[stmt]);
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt]);
  bool same=true;
  std::vector<int> pi;
  for(int i=0; i<curOrder.size(); i++){
    bool found = false;
    for(int j=0; j<cIdxNames.size(); j++){
      if(strcmp(cIdxNames[j].c_str(), curOrder[i].c_str()) == 0){
        pi.push_back(j+1);
        found=true;
        if(j!=i)
          same=false;
      }
    }
    if(!found){
      throw std::runtime_error("One of the indexes in the permute order where not "
                               "found in the current set of indexes.");
    }
  }
  for(int i=curOrder.size(); i<cIdxNames.size(); i++){
    pi.push_back(i);
  }
  if(same)
    return;
  permute(stmt, pi);
  //Set old indexe names as new
  for(int i=0; i<curOrder.size(); i++){
    idxNames[stmt][i] = curOrder[i].c_str(); //what about sibling stmts?
  }
}


bool LoopCuda::permute(int stmt_num, const std::vector<int> &pi)
{
// check for sanity of parameters
  if (stmt_num >= stmt.size() || stmt_num < 0)
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  const int n = stmt[stmt_num].xform.n_out();
  if (pi.size() > (n-1)/2)
    throw std::invalid_argument("iteration space dimensionality does not match permute dimensionality");
  int first_level = 0;
  int last_level = 0;
  for (int i = 0; i < pi.size(); i++) {
    if (pi[i] > (n-1)/2 || pi[i] <= 0)
      throw std::invalid_argument("invalid loop level " + to_string(pi[i]) + " in permuation");
    
    if (pi[i] != i+1) {
      if (first_level == 0)
        first_level = i+1;      
      last_level = i+1;
    }
  }
  if (first_level == 0)
    return true;
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> active = getStatements(lex, 2*first_level-2);
  Loop::permute(active, pi);
}


void LoopCuda::tile_cuda(int stmt, int level, int outer_level)
{
  tile_cuda(stmt,level,1,outer_level,"","",CountedTile);
}
void LoopCuda::tile_cuda(int level, int tile_size, int outer_level, std::string idxName,
                         std::string ctrlName, TilingMethodType method){
  tile_cuda(0, level, tile_size, outer_level, idxName, ctrlName, method);
}

void LoopCuda::tile_cuda(int stmt, int level, int tile_size, int outer_level, std::string idxName,
                         std::string ctrlName, TilingMethodType method){
  //Do regular tile but then update the index and control loop variable
  //names as well as the idxName to reflect the current state of things.
  //printf("tile(%d,%d,%d,%d)\n", stmt, level, tile_size, outer_level);
  //printf("idxNames before: ");
  //printVS(idxNames[stmt]);
  
  tile(stmt, level, tile_size, outer_level, method);
  
  if(idxName.size())
    idxNames[stmt][level-1] = idxName.c_str();
  if(tile_size == 1){
    //potentially rearrange loops
    if(outer_level < level){
      std::string tmp = idxNames[stmt][level-1];
      for(int i=level-1; i>outer_level-1; i--){
        if(i-1 >= 0)
          idxNames[stmt][i] = idxNames[stmt][i-1];
      }
      idxNames[stmt][outer_level-1] = tmp;
    }
    //TODO: even with a tile size of one, you need a insert (of a dummy loop)
    idxNames[stmt].insert(idxNames[stmt].begin()+(level),"");
  }else{
    if(!ctrlName.size())
      throw std::runtime_error("No ctrl loop name for tile");
    //insert
    idxNames[stmt].insert(idxNames[stmt].begin()+(outer_level-1),ctrlName.c_str());
  }
  
  //printf("idxNames after: ");
  //printVS(idxNames[stmt]);
}


bool LoopCuda::datacopy_privatized_cuda(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read , int fastest_changing_dimension , int padding_stride , int padding_alignment , bool cuda_shared)
{
  int old_stmts =stmt.size();
  //datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, cuda_shared);
  if(cuda_shared)
    datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, 1);
  else
    datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, 0);
  
  
  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  int new_stmts = stmt.size();
  for(int i=old_stmts; i<new_stmts; i++){
    //printf("fixing up statement %d\n", i);
    std::vector<std::string> idxs;
    
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(omega::Tuple<int>());
    
    //Indexes up to level will be the same
    for(int j=0; j<level-1; j++)
      idxs.push_back(cIdxNames[j]);
    
    //Expect privatized_levels to match
    for(int j=0; j<privatized_levels.size(); j++)
      idxs.push_back(cIdxNames[privatized_levels[j]-1]);//level is one-based
    
    //all further levels should match order they are in originally
    if(privatized_levels.size()){
      int last_privatized = privatized_levels.back();
      int top_level = last_privatized + (stmt[i].IS.n_set()-idxs.size());
      //printf("last privatized_levels: %d top_level: %d\n", last_privatized, top_level);
      for(int j=last_privatized; j<top_level; j++){
        idxs.push_back(cIdxNames[j]);
        //printf("pushing back: %s\n", (const char*)cIdxNames[j]);
      }
    }
    idxNames.push_back(idxs);
  }
}

bool LoopCuda::datacopy_cuda(int stmt_num, int level, const std::string &array_name, std::vector<std::string> new_idxs, bool allow_extra_read, int fastest_changing_dimension, int padding_stride, int padding_alignment, bool cuda_shared)
{
  
  int old_stmts =stmt.size();
  //datacopy(stmt_num,level,array_name,allow_extra_read,fastest_changing_dimension,padding_stride,padding_alignment,cuda_shared);
  if(cuda_shared)
    datacopy(stmt_num,level,array_name,allow_extra_read,fastest_changing_dimension,padding_stride,padding_alignment, 1);
  else
    datacopy(stmt_num,level,array_name,allow_extra_read,fastest_changing_dimension,padding_stride,padding_alignment, 0);
  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  int new_stmts = stmt.size();
  for(int i=old_stmts; i<new_stmts; i++){
    //printf("fixing up statement %d\n", i);
    std::vector<std::string> idxs;
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(omega::Tuple<int>());
    
    //protonu--lets dump out the code from each statement here
    //printf("\n dumping statement :%d", i);
    //stmt[i].code->Dump();
    
    //Indexes up to level will be the same
    for(int j=0; j<level-1; j++)
      idxs.push_back(cIdxNames[j]);
    
    //all further levels should get names from new_idxs
    int top_level = stmt[i].IS.n_set();
    //printf("top_level: %d level: %d\n", top_level, level);
    if(new_idxs.size() < top_level-level+1)
      throw std::runtime_error("Need more new index names for new datacopy loop levels");
    
    for(int j=level-1; j<top_level; j++){
      idxs.push_back(new_idxs[j-level+1].c_str());
      //printf("pushing back: %s\n", new_idxs[j-level+1].c_str());
    }
    idxNames.push_back(idxs);
  }
}

bool LoopCuda::unroll_cuda(int stmt_num, int level, int unroll_amount)
{
  int old_stmts =stmt.size();
  //bool b= unroll(stmt_num, , unroll_amount);
  
  
  int dim = 2*level-1;
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  level = nonDummyLevel(stmt_num,level);
  //printf("unrolling %d at level %d\n", stmt_num,level);
  
  //protonu--using the new version of unroll, which returns
  //a set of ints instead of a bool. To keep Gabe's logic
  //I'll check the size of the set, if it's 0 return true
  //bool b= unroll(stmt_num, level, unroll_amount);
  std::set<int> b_set= unroll(stmt_num, level, unroll_amount);
  bool b = false;
  if (b_set.size() == 0) b = true;
  //end--protonu
  
  //Adjust idxNames to reflect updated state
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
  std::vector<std::string> origSource = idxNames[stmt_num];;
  //Drop index names at level
  if(unroll_amount == 0){
    //For all statements that were in this unroll together, drop index name for unrolled level
    idxNames[stmt_num][level-1] = "";
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
      //printf("in same loop as %d is %d\n", stmt_num, (*i));
      //idxNames[(*i)][level-1] = "";
      idxNames[(*i)] = idxNames[stmt_num];
    }
  }
  
  lex = getLexicalOrder(stmt_num);
  same_loop = getStatements(lex, dim-1);
  
  bool same_as_source = false;
  int new_stmts = stmt.size();
  for(int i=old_stmts; i<new_stmts; i++){
    //Check whether we had a sync for the statement we are unrolling, if
    //so, propogate that to newly created statements so that if they are
    //in a different loop structure, they will also get a syncthreads
    int size = syncs.size();
    for(int j=0; j<size; j++){
      if(syncs[j].first == stmt_num)
        syncs.push_back(make_pair(i,syncs[j].second));
    }
    
    //protonu-making sure the vector of nonSplitLevels grows along with
    //the statement structure
    stmt_nonSplitLevels.push_back(omega::Tuple<int>());
    
    
    //We expect that new statements have a constant for the variable in
    //stmt[i].IS at level (as seen with print_with_subs), otherwise there
    //will be a for loop at level and idxNames should match stmt's
    //idxNames pre-unrolled
    Relation IS = stmt[i].IS;
    //Ok, if you know how the hell to get anything out of a Relation, you
    //should probably be able to do this more elegantly. But for now, I'm
    //hacking it.
    std::string s = IS.print_with_subs_to_string();
    //s looks looks like
    //{[_t49,8,_t51,_t52,128]: 0 <= _t52 <= 3 && 0 <= _t51 <= 15 && 0 <= _t49 && 64_t49+16_t52+_t51 <= 128}
    //where level == 5, you see a integer in the input set
    
    //If that's not an integer and this is the first new statement, then
    //we think codegen will have a loop at that level. It's not perfect,
    //not sure if it can be determined without round-tripping to codegen.
    int sIdx = 0;
    int eIdx = 0;
    for(int j=0; j<level-1; j++){
      sIdx = s.find(",",sIdx+1);
      if(sIdx < 0) break;
    }
    if(sIdx > 0){
      eIdx = s.find("]");
      int tmp = s.find(",",sIdx+1);
      if(tmp > 0 && tmp < eIdx)
        eIdx = tmp; //", before ]"
      if(eIdx > 0){
        sIdx++;
        std::string var = s.substr(sIdx,eIdx-sIdx);
        //printf("%s\n", s.c_str());
        //printf("set var for stmt %d at level %d is %s\n", i, level, var.c_str());
        if(atoi(var.c_str()) == 0 && i ==old_stmts){
          //TODO:Maybe do see if this new statement would be in the same
          //group as the original and if it would, don't say
          //same_as_source
          if(same_loop.find(i) == same_loop.end()){
            printf("stmt %d level %d, newly created unroll statement should have same level indexes as source\n", i, level);
            same_as_source = true;
          }
        }
      }
    }
    
    
    //printf("fixing up statement %d n_set %d with %d levels\n", i, stmt[i].IS.n_set(), level-1);
    if(same_as_source)
      idxNames.push_back(origSource);
    else
      idxNames.push_back(idxNames[stmt_num]);
  }
  
  return b;
}

void LoopCuda::copy_to_texture(const char *array_name)
{
  //protonu--placeholder for now
  //set the bool for using cuda memory as true
  //in a vector of strings, put the names of arrays to tex mapped
  if ( !texture )
    texture = new texture_memory_mapping(true, array_name);
  else
    texture->add(array_name);
  
  
}


void LoopCuda::copy_to_constant(const char *array_name)
{
  //protonu--placeholder for now
  //set the bool for using cuda memory as true
  //in a vector of strings, put the names of arrays to tex mapped
  if ( !constant_mem )
    constant_mem = new constant_memory_mapping(true, array_name);
  else
    constant_mem->add(array_name);
}

//protonu--moving this from Loop
tree_node_list* LoopCuda::codegen()
{
  if(code_gen_flags & GenCudaizeV2)
    return cudaize_codegen_v2();
  //Do other flagged codegen methods, return plain vanilla generated code
  return getCode();
}

//These three are in Omega code_gen.cc and are used as a massive hack to
//get out some info from MMGenerateCode. Yea for nasty side-effects.
namespace omega{
  extern int checkLoopLevel;
  extern int stmtForLoopCheck;
  extern int upperBoundForLevel;
  extern int lowerBoundForLevel;
}


void LoopCuda::extractCudaUB(int stmt_num, int level, int &outUpperBound, int &outLowerBound){
  // check for sanity of parameters
  const int m = stmt.size();
  if (stmt_num >= m || stmt_num < 0)
    throw std::invalid_argument("invalid statement " + to_string(stmt_num));
  const int n = stmt[stmt_num].xform.n_out();
  if (level > (n-1)/2 || level <= 0)
    throw std::invalid_argument("invalid loop level " + to_string(level));
  
  int dim = 2*level-1;
  
  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> same_loop = getStatements(lex, dim-1);
  
  // extract the intersection of the iteration space to be considered
  Relation hull;
  {
    hull = Relation::True(n);
    for (std::set<int>::iterator i = same_loop.begin(); i != same_loop.end(); i++) {
      hull = Intersection(hull, project_onto_levels(getNewIS(*i), dim+1, true));
      hull.simplify(2, 4);
    }
    
    for (int i = 2; i <= dim+1; i+=2) {
      //std::string name = std::string("_t") + to_string(t_counter++);
      std::string name = std::string("_t") + to_string(tmp_loop_var_name_counter++);
      hull.name_set_var(i, name);
    }
    hull.setup_names();
  }
  
  // extract the exact loop bound of the dimension to be unrolled
  if (is_single_iteration(hull, dim)){
    throw std::runtime_error("No loop availabe at level to extract upper bound.");
  }
  Relation bound = get_loop_bound(hull, dim);
  if (!bound.has_single_conjunct() || !bound.is_satisfiable() || bound.is_tautology())
    throw loop_error("loop error: unable to extract loop bound for cudaize");
  
  // extract the loop stride
  EQ_Handle stride_eq;
  int stride = 1;
  {
    bool simple_stride = true;
    int strides = countStrides(bound.query_DNF()->single_conjunct(), bound.set_var(dim+1), stride_eq, simple_stride);
    if (strides > 1)
      throw loop_error("loop error: too many strides");
    else if (strides == 1) {
      int sign = stride_eq.get_coef(bound.set_var(dim+1));
//      assert(sign == 1 || sign == -1);
      Constr_Vars_Iter it(stride_eq, true);
      stride = abs((*it).coef/sign);
    }
  }
  if(stride != 1){
    char buf[1024];
    sprintf(buf, "Cudaize: Loop at level %d has non-one stride of %d", level, stride);
    throw std::runtime_error(buf);
  }
  
  //Use code generation system to build tell us our bound information. We
  //need a hard upper bound a 0 lower bound.
  
  checkLoopLevel = level*2;
  stmtForLoopCheck = stmt_num;
  upperBoundForLevel = -1;
  lowerBoundForLevel = -1;
  printCode(1,false);
  checkLoopLevel = 0;
  
  outUpperBound = upperBoundForLevel;
  outLowerBound = lowerBoundForLevel;
  return;
}


void LoopCuda::printCode(int effort, bool actuallyPrint) const {
  const int m = stmt.size();
  if (m == 0)
    return;
  const int n = stmt[0].xform.n_out();
  
  
  
  Tuple<Relation> IS(m);
  Tuple<Relation> xform(m);
  Tuple<IntTuple > nonSplitLevels(m);
  for (int i = 0; i < m; i++) {
    IS[i+1] = stmt[i].IS;
    xform[i+1] = stmt[i].xform;
    nonSplitLevels[i+1] = stmt_nonSplitLevels[i];
    //nonSplitLevels[i+1] = stmt[i].nonSplitLevels;
  }
  
  Tuple< Tuple<std::string> > idxTupleNames;
  if(useIdxNames){
    for(int i=0; i<idxNames.size(); i++){
      Tuple<std::string> idxs;
      for(int j=0; j<idxNames[i].size(); j++)
        idxs.append(idxNames[i][j]);
      idxTupleNames.append( idxs );
    }
  }
  
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
  CG_stringBuilder *ocg = new CG_stringBuilder();
  Tuple<CG_outputRepr *> nameInfo;
  for (int i = 1; i <= m; i++)
    nameInfo.append(new CG_stringRepr("s" + to_string(i)));
  CG_outputRepr* repr = MMGenerateCode(ocg, xform, IS, nameInfo, known, nonSplitLevels, syncs, idxTupleNames, effort);
  if(actuallyPrint)
    std::cout << GetString(repr);
/*
  for (int i = 1; i <= m; i++)
  delete nameInfo[i];
*/
  
  delete ocg;
}



void LoopCuda::printRuntimeInfo() const {
  for(int i=0; i<stmt.size(); i++){
    Relation IS = stmt[i].IS;
    Relation xform = stmt[i].xform;
    printf("stmt[%d]\n", i);
    printf("IS\n");
    IS.print_with_subs();
    
    printf("xform[%d]\n", i);
    xform.print_with_subs();
    
    //printf("code\n");
    //static_cast<CG_suifRepr *>(stmt[i].code)->GetCode()->print_expr();
  }
}

void LoopCuda::printIndexes() const {
  for(int i=0; i<stmt.size(); i++){
    printf("stmt %d nset %d ", i, stmt[i].IS.n_set());
    
    for(int j=0; j<idxNames[i].size(); j++){
      if(j>0)
        printf(",");
      printf("%s", idxNames[i][j].c_str());
    }
    printf("\n");
  }
}

tree_node_list* LoopCuda::getCode(int effort) const {
  const int m = stmt.size();
  if (m == 0)
    return new tree_node_list;
  const int n = stmt[0].xform.n_out();
  
  
  
  Tuple<CG_outputRepr *> ni(m);
  Tuple<Relation> IS(m);
  Tuple<Relation> xform(m);
  Tuple< IntTuple > nonSplitLevels(m);
  for (int i = 0; i < m; i++) {
    ni[i+1] = stmt[i].code;
    IS[i+1] = stmt[i].IS;
    xform[i+1] = stmt[i].xform;
    nonSplitLevels[i+1] = stmt_nonSplitLevels[i];
    //nonSplitLevels[i+1] = stmt[i].nonSplitLevels;
  }
  
  
  Relation known = Extend_Set(copy(this->known), n - this->known.n_set());
#ifdef DEBUG
//  std::cout << GetString(MMGenerateCode(new CG_stringBuilder(), xform, IS, known, effort));
#endif
  Tuple< Tuple<std::string> > idxTupleNames;
  if(useIdxNames){
    for(int i=0; i<idxNames.size(); i++){
      Tuple<std::string> idxs;
      for(int j=0; j<idxNames[i].size(); j++)
        idxs.append(idxNames[i][j]);
      idxTupleNames.append( idxs );
    }
  }
  
  CG_outputBuilder *ocg = ir->builder();
  CG_outputRepr *repr = MMGenerateCode(ocg, xform, IS, ni, known, nonSplitLevels, syncs, idxTupleNames, effort);
  
  //CG_outputRepr *overflow_initialization = ocg->CreateStmtList();
  //protonu--using the new function CG_suifBuilder::StmtListAppend
  CG_outputRepr *overflow_initialization = ocg->StmtListAppend(NULL, NULL);
  for (std::map<int, std::vector<Free_Var_Decl *> >::const_iterator i = overflow.begin(); i != overflow.end(); i++)
    for (std::vector<Free_Var_Decl *>::const_iterator j = i->second.begin(); j != i->second.end(); j++)
      //overflow_initialization = ocg->StmtListAppend(overflow_initialization, ocg->CreateStmtList(ocg->CreateAssignment(0, ocg->CreateIdent((*j)->base_name()), ocg->CreateInt(0))));
      overflow_initialization = ocg->StmtListAppend(overflow_initialization, ocg->StmtListAppend(ocg->CreateAssignment(0, ocg->CreateIdent((*j)->base_name()), ocg->CreateInt(0)), NULL));
  
  repr = ocg->StmtListAppend(overflow_initialization, repr);
  tree_node_list *tnl = static_cast<CG_suifRepr *>(repr)->GetCode();
  
  delete repr;
  /*
    for (int i = 1; i <= m; i++)
    delete ni[i];
  */
  
  return tnl;
}


//protonu--adding constructors for the new derived class
LoopCuda::LoopCuda():Loop(), code_gen_flags(GenInit){}

LoopCuda::LoopCuda(IR_Control *irc, int loop_num)
  :Loop(irc)
{
    setup_code = NULL;
  teardown_code = NULL;
  code_gen_flags = 0;
  cu_bx = cu_by = cu_tx = cu_ty = cu_tz = 1;
  cu_num_reduce = 0;
  cu_mode = GlobalMem;
  texture = NULL;
  constant_mem = NULL;
  
  int m=stmt.size();
  //printf("\n the size of stmt(initially) is: %d\n", stmt.size());
  for(int i=0; i<m; i++)
    stmt_nonSplitLevels.push_back(omega::Tuple<int>());
  
  
  //protonu--setting up
  //proc_symtab *symtab
  //global_symtab *globals
  
  globals =  ((IR_cudasuifCode *)ir)->gsym_ ;
  std::vector<tree_for *> tf = ((IR_cudasuifCode *)ir)->get_loops();
  
  symtab = tf[loop_num]->proc()->block()->proc_syms();
  
  std::vector<tree_for *> deepest = find_deepest_loops(tf[loop_num]);
  
  for (int i = 0; i < deepest.size(); i++){
    index.push_back(deepest[i]->index()->name()); //reflects original code index names
  }
  
  for(int i=0; i< stmt.size(); i++)
    idxNames.push_back(index); //refects prefered index names (used as handles in cudaize v2)
  useIdxNames=false;
  
}

