/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   SUIF interface utilities.

 Notes:

 Update history:
   01/2006 created by Chun Chen
*****************************************************************************/

#include <suif1.h>
#include <useful.h>
#include <vector>
#include <algorithm>
#include <code_gen/CG_suifRepr.h>
#include "ir_suif_utils.hh"

// ----------------------------------------------------------------------------
// Mandatory SUIF stuff
// ----------------------------------------------------------------------------
char *prog_ver_string = "1.3.0.5-gccfix";
char *prog_who_string = "automatically generated from chill";
char *prog_suif_string = "suif";

// static file_set_entry *fse = NULL;
// static proc_sym *psym = NULL;

// class SUIF_IR;

// SUIF_IR *ir = NULL;

// SUIF_IR::SUIF_IR(char *filename, int proc_num) {
// //  LIBRARY(ipmath, init_ipmath, exit_ipmath);
//   LIBRARY(useful, init_useful, exit_useful);
//   LIBRARY(annotes, init_annotes, exit_annotes);

//   int argc = 3;
//   char *argv[3];
//   argv[0] = "loop_xform";
//   argv[1] = strdup(filename);
//   argv[2] = strdup(filename);
//   char *pos = strrchr(argv[2], '.');
//   if (pos == NULL)
//     strcat(argv[2], ".lxf");
//   else {
//     *pos = '\0';
//     strcat(argv[2], ".lxf");
//   }
//   init_suif(argc, argv);

//   fileset->add_file(argv[1], argv[2]);
//   fileset->reset_iter();
//   _fse = fileset->next_file();
//   _fse->reset_proc_iter();
//   int cur_proc = 0;
//   while ((_psym = _fse->next_proc()) && cur_proc < proc_num)
//     ++cur_proc;
//   if (cur_proc != proc_num) {
//     fprintf(stderr, "procedure number %d couldn't be found\n", proc_num);
//     exit(1);
//   }
//   if (!_psym->is_in_memory())
//     _psym->read_proc(TRUE, _psym->src_lang() == src_fortran);

//   push_clue(_psym->block());
// }


// SUIF_IR::~SUIF_IR() {
//   pop_clue(_psym->block());
//     if (!_psym->is_written())
//       _psym->write_proc(_fse);
//     _psym->flush_proc();

//     exit_suif1();
// }


// tree_for *SUIF_IR::get_loop(int loop_num) {
//   std::vector<tree_for *> loops = find_loops(_psym->block()->body());
//   if (loop_num >= loops.size()) {
//     fprintf(stderr, "loop number %d couldn't be found\n", loop_num);
//     exit(1);
//   }
//   return loops[loop_num];
// }


// void SUIF_IR::commit(Loop *lp, int loop_num) {
//   if (lp == NULL)
//     return;

//   if (lp->init_code != NULL) {
//     tree_node_list *init_tnl = static_cast<CG_suifRepr *>(lp->init_code->clone())->GetCode();
//     tree_node_list_iter iter(lp->symtab->block()->body());
//     iter.step();
//     lp->symtab->block()->body()->insert_before(init_tnl, iter.cur_elem());
//   }

//   tree_node_list *code = lp->getCode();
//   std::vector<tree_for *> loops = find_loops(_psym->block()->body());
//   tree_node_list *tnl = loops[loop_num]->parent();
//   tnl->insert_before(code, loops[loop_num]->list_e());
//   tnl->remove(loops[loop_num]->list_e());
// }


// extern void start_suif(int &argc, char *argv[]) {
// //    LIBRARY(ipmath, init_ipmath, exit_ipmath);
//     LIBRARY(useful, init_useful, exit_useful);
//     LIBRARY(annotes, init_annotes, exit_annotes);

//     init_suif(argc, argv);
// }

// tree_for *init_loop(char *filename, int proc_num, int loop_num) {
// //  LIBRARY(ipmath, init_ipmath, exit_ipmath);
//   LIBRARY(useful, init_useful, exit_useful);
//   LIBRARY(annotes, init_annotes, exit_annotes);

//   int argc = 3;
//   char *argv[3];
//   argv[0] = "loop_xform";
//   argv[1] = filename;
//   argv[2] = strdup(filename);
//   char *pos = strrchr(argv[2], '.');
//   if (pos == NULL)
//     strcat(argv[2], ".lxf");
//   else {
//     *pos = '\0';
//     strcat(argv[2], ".lxf");
//   }
//   printf("%s %s %s\n", argv[0], argv[1], argv[2]);
//   init_suif(argc, argv);

//   fileset->add_file(argv[1], argv[2]);
//   fileset->reset_iter();
//   fse = fileset->next_file();
//   fse->reset_proc_iter();
//   int cur_proc = 0;
//   while ((psym = fse->next_proc()) && cur_proc < proc_num)
//     ++cur_proc;
//   if (cur_proc != proc_num) {
//     fprintf(stderr, "procedure number %d couldn't be found\n", proc_num);
//     exit(1);
//   }

//   if (!psym->is_in_memory())
//      psym->read_proc(TRUE, psym->src_lang() == src_fortran);

//   push_clue(psym->block());
//   std::vector<tree_for *> loops = find_loops(psym->block()->body());
//   if (loop_num >= loops.size())
//     return NULL;
//   return loops[loop_num];
// }


// void finalize_loop() {

//   printf("finalize %d\n", fse);
//   pop_clue(psym->block());
//   if (!psym->is_written())
//     psym->write_proc(fse);
//   psym->flush_proc();
// }



// // ----------------------------------------------------------------------------
// // Class: CG_suifArray
// // ----------------------------------------------------------------------------
// CG_suifArray::CG_suifArray(in_array *ia_): ia(ia_) {
//   var_sym *vs = get_sym_of_array(ia);
//   name = String(vs->name());

//   for (int i = 0; i < ia->dims(); i++)
//     index.push_back(new CG_suifRepr(ia->index(i)));
// }

// bool CG_suifArray::is_write() {
//   return is_lhs(ia);
// }


// ----------------------------------------------------------------------------
// Find array index in various situations.
// ----------------------------------------------------------------------------
operand find_array_index(in_array *ia, int n, int dim, bool is_fortran) {
  if (!is_fortran)
    dim = n - dim - 1;
  int level = n - dim -1;
  
  in_array *current = ia;
  
  while (true) {
    int n = current->dims();
    if (level < n) {
      return current->index(level);
    }
    else {
      level = level - n;
      operand op = current->base_op();
      assert(op.is_instr());
      instruction *ins = op.instr();
      if (ins->opcode() != io_cvt)
        return operand();
      operand op2 = static_cast<in_rrr *>(ins)->src_op();
      assert(op2.is_instr());
      instruction *ins2 = op2.instr();
      assert(ins2->opcode() == io_lod);
      operand op3 = static_cast<in_rrr *>(ins2)->src_op();
      assert(op3.is_instr());
      instruction *ins3 = op3.instr();
      assert(ins3->opcode() == io_array);
      current = static_cast<in_array *>(ins3);
    }
  }
}




// ----------------------------------------------------------------------------
// Check if a tree_node is doing nothing
// ----------------------------------------------------------------------------
bool is_null_statement(tree_node *tn) {
  if (tn->kind() != TREE_INSTR)
    return false;
  
  instruction *ins = static_cast<tree_instr*>(tn)->instr();
  
  if (ins->opcode() == io_mrk || ins->opcode() == io_nop)
    return true;
  else
    return false;
}

// ----------------------------------------------------------------------------
// Miscellaneous loop functions
// ----------------------------------------------------------------------------
std::vector<tree_for *> find_deepest_loops(tree_node *tn) {
  if (tn->kind() == TREE_FOR) {
    std::vector<tree_for *> loops;
    
    tree_for *tnf = static_cast<tree_for *>(tn);
    loops.insert(loops.end(), tnf);
    std::vector<tree_for *> t = find_deepest_loops(tnf->body());
    std::copy(t.begin(), t.end(), std::back_inserter(loops));
    
    return loops;
  }
  else if (tn->kind() == TREE_BLOCK) {
    tree_block *tnb = static_cast<tree_block *>(tn);
    return find_deepest_loops(tnb->body());
  }
  else
    return std::vector<tree_for *>();
}

std::vector<tree_for *> find_deepest_loops(tree_node_list *tnl) {
  std::vector<tree_for *> loops;
  
  tree_node_list_iter iter(tnl);
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    
    std::vector<tree_for *> t = find_deepest_loops(tn);
    
    if (t.size() > loops.size())
      loops = t;
  }
  
  return loops;
}

std::vector<tree_for *> find_loops(tree_node_list *tnl) {
  std::vector<tree_for *> result;
  
  tree_node_list_iter iter(tnl);
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    if (tn->kind() == TREE_FOR)
      result.push_back(static_cast<tree_for *>(tn));
  }
  
  return result;
}


std::vector<tree_for *> find_outer_loops(tree_node *tn) {
  std::vector<tree_for *> loops;
  
  while(tn) {
    if(tn->kind() == TREE_FOR)
      loops.insert(loops.begin(),static_cast<tree_for*>(tn));
    tn = (tn->parent())?tn->parent()->parent():NULL; 
  }
  
  return loops;
}

std::vector<tree_for *> find_common_loops(tree_node *tn1, tree_node *tn2) {
  std::vector<tree_for *> loops1 = find_outer_loops(tn1);
  std::vector<tree_for *> loops2 = find_outer_loops(tn2);
  
  std::vector<tree_for *> loops;
  
  for (unsigned i = 0; i < std::min(loops1.size(), loops2.size()); i++) {
    if (loops1[i] == loops2[i])
      loops.insert(loops.end(), loops1[i]);
    else
      break;
  }
  
  return loops;
}


//-----------------------------------------------------------------------------
// Determine the lexical order between two instructions.
//-----------------------------------------------------------------------------
LexicalOrderType lexical_order(tree_node *tn1, tree_node *tn2) {
  if (tn1 == tn2)
    return LEX_MATCH;
  
  std::vector<tree_node *> tnv1;
  std::vector<tree_node_list *> tnlv1;
  while (tn1 != NULL && tn1->parent() != NULL) {
    tnv1.insert(tnv1.begin(), tn1);
    tnlv1.insert(tnlv1.begin(), tn1->parent());
    tn1 = tn1->parent()->parent();
  }
  
  std::vector<tree_node *> tnv2;
  std::vector<tree_node_list *> tnlv2;
  while (tn2 != NULL && tn2->parent() != NULL) {
    tnv2.insert(tnv2.begin(), tn2);
    tnlv2.insert(tnlv2.begin(), tn2->parent());
    tn2 = tn2->parent()->parent();
  }
  
  for (int i = 0; i < std::min(tnlv1.size(), tnlv2.size()); i++) {
    if (tnlv1[i] == tnlv2[i] && tnv1[i] != tnv2[i]) {
      tree_node_list_iter iter(tnlv1[i]);
      
      while (!iter.is_empty()) {
        tree_node *tn = iter.step();
        
        if (tn == tnv1[i])
          return LEX_BEFORE;
        else if (tn == tnv2[i])
          return LEX_AFTER;
      }
      
      break;
    }
  }
  
  return LEX_UNKNOWN;
}  



//-----------------------------------------------------------------------------
// Get the list of array instructions
//-----------------------------------------------------------------------------
std::vector<in_array *> find_arrays(instruction *ins) {
  std::vector<in_array *> arrays;
  if (ins->opcode() == io_array) {
    arrays.insert(arrays.end(), static_cast<in_array *>(ins));
  }
  else {
    for (int i = 0; i < ins->num_srcs(); i++) {
      operand op(ins->src_op(i));
      if (op.is_instr()) {
        std::vector<in_array *> t = find_arrays(op.instr());
        std::copy(t.begin(), t.end(), back_inserter(arrays));
      }
    }
  }
  return arrays;
}

std::vector<in_array *> find_arrays(tree_node_list *tnl) {
  std::vector<in_array *> arrays, t;
  tree_node_list_iter iter(tnl);
  
  while (!iter.is_empty()) {
    tree_node *tn = iter.step();
    
    if (tn->kind() == TREE_FOR) {
      tree_for *tnf = static_cast<tree_for *>(tn);
      
      t = find_arrays(tnf->body());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
    }
    else if (tn->kind() == TREE_IF) {
      tree_if *tni = static_cast<tree_if *>(tn);
      
      t = find_arrays(tni->header());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
      t = find_arrays(tni->then_part());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
      t = find_arrays(tni->else_part());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
    }
    else if (tn->kind() == TREE_BLOCK) {
      t = find_arrays(static_cast<tree_block *>(tn)->body());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
    }
    else if (tn->kind() == TREE_INSTR) {
      t = find_arrays(static_cast<tree_instr *>(tn)->instr());
      std::copy(t.begin(), t.end(), back_inserter(arrays));
    }      
  }
  
  return arrays;
}

// std::vector<CG_suifArray *> find_array_access(instruction *ins) {
//   std::vector<CG_suifArray *> arrays;

//   if (ins->opcode() == io_array) {
//     arrays.push_back(new CG_suifArray(static_cast<in_array *>(ins)));
//   }
//   else {
//     for (int i = 0; i < ins->num_srcs(); i++) {
//       operand op(ins->src_op(i));
//       if (op.is_instr()) {
//         std::vector<CG_suifArray *> t = find_array_access(op.instr());
//         std::copy(t.begin(), t.end(), back_inserter(arrays));
//       }
//     }
//   }
//   return arrays;
// }

// std::vector<CG_suifArray *> find_array_access(tree_node_list *tnl) {
//   std::vector<CG_suifArray *> arrays, t;
//   tree_node_list_iter iter(tnl);

//   while (!iter.is_empty()) {
//     tree_node *tn = iter.step();

//     if (tn->kind() == TREE_FOR) {
//       tree_for *tnf = static_cast<tree_for *>(tn);

//       t = find_array_access(tnf->body());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//     }
//     else if (tn->kind() == TREE_IF) {
//       tree_if *tni = static_cast<tree_if *>(tn);

//       t = find_array_access(tni->header());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//       t = find_array_access(tni->then_part());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//       t = find_array_access(tni->else_part());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//     }
//     else if (tn->kind() == TREE_BLOCK) {
//       t = find_array_access(static_cast<tree_block *>(tn)->body());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//     }
//     else if (tn->kind() == TREE_INSTR) {
//       t = find_array_access(static_cast<tree_instr *>(tn)->instr());
//       std::copy(t.begin(), t.end(), back_inserter(arrays));
//     }      
//   }

//   return arrays;
// }
