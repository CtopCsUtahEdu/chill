
// chill interface to python

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "chill_io.hh"
#include <fstream>
#include "parser/clang.h"
#include "parser/rose.h"
#include "chillmodule.hh"

// TODO 
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

using namespace omega;

loop_t *myloop = nullptr;
IR_Code *ir_code = nullptr;
bool is_interactive = false;

// -- Cuda CHiLL global variables --
#ifndef CUDACHILL
#include "chill_run_util.hh"

std::string procedure_name;
std::string source_filename;

int loop_start_num;
int loop_end_num;
#endif

std::string dest_filename;

std::vector<IR_Control *> ir_controls;
std::vector<int> loops;

namespace {
  chill::Parser *parser;
  int effort;
}

// ----------------------- //
// CHiLL support functions //
// ----------------------- //
#ifndef CUDACHILL
// not sure yet if this actually needs to be exposed to the python interface
// these four functions are here to maintain similarity to the Lua interface
int get_loop_num_start() {
  return loop_start_num;
}

int get_loop_num_end() {
  return loop_end_num;
}

static void set_loop_num_start(int start_num) {
  loop_start_num = start_num;
}

static void set_loop_num_end(int end_num) {
  loop_end_num = end_num;
}

// TODO: finalize_loop(int,int) and init_loop(int,int) are identical to thier Lua counterparts.
// consider integrating them

void finalize_loop(int loop_num_start, int loop_num_end) {
  if (loop_num_start == loop_num_end) {
    // Mahdi: Following if is a HACK for getting dependence extraction test cases 
    // running for now. Refer to loop.h::replaceCode() comments for more details. 
    if( myloop->replaceCode() ){
      ir_code->ReplaceCode(ir_controls[loops[loop_num_start]], myloop->getCode(effort));
    }
    ir_controls[loops[loop_num_start]] = NULL;
  }
  else {
    std::vector<IR_Control *> parm;
    for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++)
      parm.push_back(ir_controls[i]);
    IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
    ir_code->ReplaceCode(block, myloop->getCode(effort));
    for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
      delete ir_controls[i];
      ir_controls[i] = NULL;
    }
  }
  delete myloop;
}
void finalize_loop() {
  int loop_num_start = get_loop_num_start();
  int loop_num_end = get_loop_num_end();
  finalize_loop(loop_num_start, loop_num_end);
}
static void init_loop(int loop_num_start, int loop_num_end) {
  if (source_filename.empty()) {
    fprintf(stderr, "source file not set when initializing the loop");
    if (!is_interactive)
      exit(2);
  }
  else {
    if (ir_code == NULL) {

      if(dest_filename.empty()) {
        ir_code = new IR_chillCode(parser, source_filename.c_str(), procedure_name.c_str(), NULL);
      }
      else {
        ir_code = new IR_chillCode(parser, source_filename.c_str(), procedure_name.c_str(), dest_filename.c_str());
      }

      IR_Block *block = ir_code->GetCode();
      ir_controls = ir_code->FindOneLevelControlStructure(block);
      for (int i = 0; i < ir_controls.size(); i++) {
        if (ir_controls[i]->type() == IR_CONTROL_LOOP)
          loops.push_back(i);
      }
      delete block;
    }
    if (myloop != NULL && myloop->isInitialized()) {
       finalize_loop();
    }
  }
  set_loop_num_start(loop_num_start);
  set_loop_num_end(loop_num_end);
  if (loop_num_end < loop_num_start) {
    fprintf(stderr, "the last loop must be after the start loop");
    if (!is_interactive)
      exit(2);
  }              
  if (loop_num_end >= loops.size()) {
    fprintf(stderr, "loop %d does not exist", loop_num_end);
    if (!is_interactive)
      exit(2);
  }
  std::vector<IR_Control *> parm;
  for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
    if (ir_controls[i] == NULL) {
      fprintf(stderr, "loop has already been processed");
      if (!is_interactive)
        exit(2);
    }
    parm.push_back(ir_controls[i]);
  }
  IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
  myloop = new Loop(block);
  delete block;
  //if (is_interactive) printf("%s ", PROMPT_STRING);
}
#endif

// ----------------------- //
// Python support funcions //
// ----------------------- //

static void strict_arg_num(PyObject* args, int arg_num, const char* fname = NULL) {
  int arg_given = PyTuple_Size(args);
  char msg[128];
  if(arg_num != arg_given) {
    if(fname)
      sprintf(msg, "%s: expected %i arguments, was given %i.", fname, arg_num, arg_given);
    else
      sprintf(msg, "Expected %i argumets, was given %i.", arg_num, arg_given);
    throw std::runtime_error(msg);
  }
}

static int strict_arg_range(PyObject* args, int arg_min, int arg_max, const char* fname = NULL) {
  int arg_given = PyTuple_Size(args);
  char msg[128];
  if(arg_given < arg_min || arg_given > arg_max) {
    if(fname)
      sprintf(msg, "%s: expected %i to %i arguments, was given %i.", fname, arg_min, arg_max, arg_given);
    else
      sprintf(msg, "Expected %i to %i, argumets, was given %i.", arg_min, arg_max, arg_given);
    throw std::runtime_error(msg);
  }
  return arg_given;
}

static int intArg(PyObject* args, int index, int dval = 0) {
  if(PyTuple_Size(args) <= index)
    return dval; 
  int ival;
  PyObject *item = PyTuple_GetItem(args, index); 
  Py_INCREF(item);
  if (PyInt_Check(item)) ival = PyInt_AsLong(item);
  else
    throw std::runtime_error("argument at index "+to_string(index)+" is not an int");
  return ival;
}

static std::string strArg(PyObject* args, int index, const char* dval = NULL) {
  if(PyTuple_Size(args) <= index)
    return dval;
  std::string strval;
  PyObject *item = PyTuple_GetItem(args, index); 
  Py_INCREF(item);
  if (PyString_Check(item)) strval = strdup(PyString_AsString(item));
  else
    throw std::runtime_error("argument at index "+to_string(index)+" is not an string");
  return strval;
}

static bool boolArg(PyObject* args, int index, bool dval = false) {
  if(PyTuple_Size(args) <= index)
    return dval;
  bool bval;
  PyObject* item = PyTuple_GetItem(args, index);
  Py_INCREF(item);
  return (bool)PyObject_IsTrue(item);
}

static bool readArg(PyObject* args, int index, std::vector<std::map<std::string,int> >& vec) {
  if(PyTuple_Size(args) <= index)
    return false;
  PyObject* seq = PyTuple_GetItem(args, index);

  //TODO: Typecheck
  int seq_len = PyList_Size(seq);
  for(int i = 0; i < seq_len; i++) {
    std::map<std::string,int> map;
    PyObject* dict = PyList_GetItem(seq, i);
    PyObject* keys = PyDict_Keys(dict);
    //TODO: Typecheck
    int dict_len = PyList_Size(keys);
    for(int j = 0; j < dict_len; j++) {
      PyObject* key = PyList_GetItem(keys, j);
      PyObject* value = PyDict_GetItem(dict, key);
      std::string str_key = strdup(PyString_AsString(key));
      int int_value = PyInt_AsLong(value);
      map[str_key] = int_value;
    }
    vec.push_back(map);
  }
  return true;
}

/**
 * Read argument at index as a map<string, int>
 */
static bool readArg(PyObject* args, int index, std::map<std::string, int>& m) {
  if(PyTuple_Size(args) <= index)
    __throw_runtime_error("bad input");

  auto dict = PyTuple_GetItem(args, index);
  if(!PyDict_Check(dict))
    __throw_runtime_error("bad input");

  auto kv_list = PyDict_Items(dict);
  for(int i = 0; i < PyList_Size(kv_list); i++) {
    auto        kv = PyList_GetItem(kv_list, i);
    std::string k  = PyString_AsString(PyTuple_GetItem(kv, 0));
    int         v  = PyInt_AsLong(PyTuple_GetItem(kv, 1));
    m[k] = v;
  }

  return true;
}

static bool to_int_vector(PyObject* seq, std::vector<int>& vec) {
  //TODO: Typecheck
  int seq_len = PyList_Size(seq);
  for(int i = 0; i < seq_len; i++) {
    PyObject* item = PyList_GetItem(seq, i);
    vec.push_back(PyInt_AsLong(item));
  }
  return true;
}

static bool to_string_vector(PyObject* seq, std::vector<std::string>& vec) {
    int seq_len = PyList_Size(seq);
    for(int i = 0; i < seq_len; i++) {
        PyObject* item = PyList_GetItem(seq, i);
        vec.push_back(std::string(strdup(PyString_AsString(item))));
    }
    return true;
}

static bool to_int_vector(PyObject* args, int index, std::vector<int>& vec) {
  if(PyTuple_Size(args) <= index)
    return false;
  PyObject* seq = PyTuple_GetItem(args, index);
  return to_int_vector(seq, vec);
}

static bool to_string_vector(PyObject* args, int index, std::vector<std::string>& vec) {
  if(PyTuple_Size(args) <= index)
    return false;
  PyObject* seq = PyTuple_GetItem(args, index);
  return to_string_vector(seq, vec);
}

static bool to_int_set(PyObject* args, int index, std::set<int>& set) {
  if(PyTuple_Size(args) <= index)
    return false;
  PyObject* seq = PyTuple_GetItem(args, index);
  //TODO: Typecheck
  int seq_len = PyList_Size(seq);
  for(int i = 0; i < seq_len; i++) {
    PyObject* item = PyList_GetItem(seq, i);
    set.insert(PyInt_AsLong(item));
  }
  return true;
}
static bool to_int_matrix(PyObject* args, int index, std::vector<std::vector<int> >& mat) {
  if(PyTuple_Size(args) <= index)
    return false;
  PyObject* seq_one = PyTuple_GetItem(args, index);
  int seq_one_len = PyList_Size(seq_one);
  for(int i = 0; i < seq_one_len; i++) {
    std::vector<int> vec;
    PyObject* seq_two = PyList_GetItem(seq_one, i);
    int seq_two_len = PyList_Size(seq_two);
    for(int j = 0; j < seq_two_len; j++) {
      PyObject* item = PyList_GetItem(seq_two, j);
      vec.push_back(PyInt_AsLong(item));
    }
    mat.push_back(vec);
  }
  return true;
}



// ---------------------------------------- //
// Cuda CHiLL and CHiLL interface functions //
// ---------------------------------------- //

static PyObject* chill_pragma(PyObject* self, PyObject* args) {
    strict_arg_num(args, 3, "pragma");

    auto stmt       = intArg(args, 0);
    auto level      = intArg(args, 1);
    auto name       = strArg(args, 2);

    myloop->omp_mark_pragma(stmt, level, name);

    Py_RETURN_NONE;
}

//TODO: this should be specific to CHiLL
static PyObject* chill_omp_for(PyObject* self, PyObject* args) {
    strict_arg_range(args, 2, 3, "omp_parallel_for");

    auto stmt       = intArg(args, 0);
    auto level      = intArg(args, 1);

    std::vector<std::string> privitized_vars;
    std::vector<std::string> shared_vars;

    // read privatized variables
    if(PyTuple_Size(args) >= 3) {
        to_string_vector(args, 2, privitized_vars);
    }

    // read shared variables
    if(PyTuple_Size(args) >= 4) {
        to_string_vector(args, 3, shared_vars);
    }

    myloop->omp_mark_parallel_for(stmt, level, privitized_vars, shared_vars);

    Py_RETURN_NONE;
}

#ifdef CUDACHILL
// ------------------------------ //
// Cuda CHiLL interface functions //
// ------------------------------ //

static PyObject *
chill_destination(PyObject *self, PyObject* args) {
  strict_arg_num(args, 1, "destination");
  dest_filename = strArg(args, 0);
  Py_RETURN_NONE;
}

static PyObject *
chill_print_code(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("\nC print_code() PY\n"); 
  
  ((Loop*)myloop)->printCode();
  
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
  
}

static PyObject *
chill_print_ri(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("\nC chill_print_ri() called from python\n"); 
  myloop->printRuntimeInfo();
  debug_fprintf(stderr, "\n");
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}

static PyObject *
chill_print_idx(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("\nC chill_print_idx() called from python\n"); 
  myloop->printIndexes();
  debug_fprintf(stderr, "\n");
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}

static PyObject *
chill_print_dep(PyObject *self, PyObject *args)
{
  debug_fprintf(stderr, "\nC chill_print_dep()\n"); 
  std::cout << myloop->dep;
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}

static PyObject *
chill_print_space(PyObject *self, PyObject *args)
{
  debug_fprintf(stderr, "\nC chill_print_space()\n"); 
  for (int i = 0; i < myloop->stmt.size(); i++) {
    debug_fprintf(stderr, "s%d: ", i+1);
    Relation r;
    if (!myloop->stmt[i].xform.is_null())
      r = Composition(copy(myloop->stmt[i].xform), copy(myloop->stmt[i].IS));
    else
      r = copy(myloop->stmt[i].IS);
    r.simplify(2, 4);
    r.print();
  }
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}

static PyObject *
chill_num_statements(PyObject *self, PyObject *args)  
{
  //DEBUG_PRINT("\nC chill_num_statements() called from python\n"); 
  int num = myloop->stmt.size();
  //DEBUG_PRINT("C num_statement() = %d\n", num); 
  return Py_BuildValue( "i", num ); // BEWARE "d" is DOUBLE, not int
}  

static PyObject *
chill_does_var_exist( PyObject *self, PyObject *args)
{
  debug_fprintf(stderr, "\nC chill_does_var_exist()\n"); 
  int yesno = 0;
  // TODO if (myloop->symbolExists(symName)) yesno = 1;
  debug_fprintf(stderr, "*** chill_does_var_exist *** UNIMPLEMENTED\n"); 
  return Py_BuildValue( "i", yesno); // there seems to be no boolean type
}


static PyObject *
chill_add_sync(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("\nC chill_add_sync()  *UNTESTED*\n"); 
  int sstmt = -123;
  //   char index_name[180];
  static char Buffer[1024];
  static char *index_name = &Buffer[0]; 
  
  if (!PyArg_ParseTuple(args, "is", &sstmt, &index_name)){
    debug_fprintf(stderr, "chill_add_sync, can't parse statement number and name passed from python\n");
    exit(-1);
  }
  
  debug_fprintf(stderr, "chill_add_sync, statement %d   index_name '%s'\n", 
              sstmt, index_name);
  std::string idxName( index_name); // ?? 
  myloop->addSync(sstmt, idxName);
  
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}

static PyObject *
chill_rename_index(PyObject *self, PyObject *args)
{
  debug_fprintf(stderr, "\nC chill_rename_index() called from python\n"); 
  int sstmt;
  //char oldname[80], newname[80];
  static char old[1024], newn[1024];
  
  static char *oldname = &old[0], *newname=&newn[0];
  
  if (!PyArg_ParseTuple(args, "iss", &sstmt, &oldname, &newname)){
    debug_fprintf(stderr, "chill_rename_index, can't parse statement number and names passed from python\n");
    exit(-1);
  }
  
  //DEBUG_PRINT("chill_rename_index, statement %d   oldname '%s'   newname '%s'\n", 
  //sstmt, oldname, newname);
  
  std::string idxName(oldname);
  std::string newName(newname); 
  
  //DEBUG_PRINT("calling myloop->renameIndex( %d, %s, %s )\n", 
  //sstmt, idxName.c_str(), newName.c_str()); 
  
  myloop->renameIndex(sstmt, idxName, newName);
  
  //DEBUG_PRINT("after myloop->renameIndex()\n");  
  
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}



//THIS NEEDS TO MOVE



static PyObject *
chill_permute_v2(PyObject *self, PyObject *args)
{
  strict_arg_num(args, 2, "permute");
  
  std::vector<std::string> order;
  int stmt = intArg(args, 0);
  to_string_vector(args, 1, order);
  myloop->permute_cuda(stmt, order);

  //DEBUG_PRINT("returned from permute_cuda()\n"); 

  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}


static PyObject *
chill_tile_v2_3arg( PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("in chillmodule.cc, chill_tile_v2_3arg()\n"); 
  
  int sstmt, level, tile_size, outer_level;
  //char index_name[80], control_name[80];
  static char *index_name, *control_name;
  int tiling_method;
  
  if (!PyArg_ParseTuple(args, "iii", &sstmt, &level, &outer_level)) {
    debug_fprintf(stderr,"chill_tile_v2, can't parse parameters passed from python\n");
    exit(-1);
  }
  
  // 3 parameter version 
  //DEBUG_PRINT("chill_tile_v2( %d %d %d)   (3 parameter version) \n", 
  //sstmt,level,outer_level);  
  myloop->tile_cuda(sstmt,level,outer_level);
  //DEBUG_PRINT("chill_tile_v2 3 parameter version returning normally\n"); 
  Py_RETURN_NONE; 
}


static PyObject *
chill_tile_v2_7arg( PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("in chillmodule.cc, chill_tile_v2_7arg()\n"); 
  
  int sstmt, level, tile_size, outer_level;
  //char index_name[80], control_name[80];
  static char iname[1024], cname[1024];
  static char *index_name = &iname[0], *control_name=&cname[0];
  int tiling_method;
  
  if (!PyArg_ParseTuple(args, "iiiissi", 
                        &sstmt, &level, &tile_size, &outer_level, 
                        &index_name, &control_name, &tiling_method)){
    debug_fprintf(stderr, "chill_tile_v2_7arg, can't parse parameters passed from python\n");
    exit(-1);
  }
  
  //DEBUG_PRINT("7 parameter version was called?\n"); 
  
  // 7 parameter version was called 
  //DEBUG_PRINT("tile_v2( %d, %d, %d, %d ... )\n", 
  // sstmt, level, tile_size, outer_level);
  
  //DEBUG_PRINT("tile_v2( %d, %d, %d, %d, %s, %s, %d)\n", 
  //sstmt,level,tile_size,outer_level,index_name, control_name, tiling_method);
  
  TilingMethodType method = StridedTile;    
  if (tiling_method == 0)  method = StridedTile;
  else if (tiling_method == 1) method = CountedTile;
  else debug_fprintf(stderr, "ERROR: tile_v2 illegal tiling method, using StridedTile\n");
  
  //DEBUG_PRINT("outer level %d\n", outer_level);  
  //DEBUG_PRINT("calling myloop->tile_cuda( %d, %d, %d, %d, %s, %s, method)\n", 
  // sstmt, level, tile_size, outer_level, index_name, control_name); 
  
  // level+1?
  myloop->tile_cuda(sstmt, level, tile_size, outer_level, index_name, control_name, method); 
  Py_RETURN_NONE; 
}


// --------------- //
// From cudaize.py //
// --------------- //
static void cur_indices(int stmt, std::vector<std::string>& indices) {
    indices.clear();
    int num = myloop->idxNames[stmt].size();
    for(int i = 0; i < num; i++) {
        indices.push_back(myloop->idxNames[stmt][i]);
    }
}

static void block_indices(std::vector<std::string>& idxs) {
    idxs.clear();
    if(myloop->cu_bx > 1) {
        idxs.push_back("bx");
    }
    if(myloop->cu_by > 1) {
        idxs.push_back("by");
    }
}

static void thread_indices(std::vector<std::string>& idxs) {
    idxs.clear();
    if(myloop->cu_tx > 1) { idxs.push_back("tx"); }
    if(myloop->cu_ty > 1) { idxs.push_back("ty"); }
    if(myloop->cu_tz > 1) { idxs.push_back("tz"); }
}

static int find_cur_level(int stmt, std::string idx) {
    std::vector<std::string> cur;
    cur_indices(stmt, cur);

    int index = 1;
    for(auto c: cur) {
        if(c == idx) {
            return index;
        }
        index++;
    }
    return -1; // Not found
}

template<typename T>
static inline bool in(const std::vector<T>& vec, T val) {
    return std::find(std::begin(vec), std::end(vec), val) != vec.end();
}

template<typename T>
static inline int indexof(const std::vector<T>& vec, T val) {
    for(int i = 0; i < vec.size(); i++) {
        if(vec[i] == val) return i;
    }
    return -1;
}

static PyObject* chill_copy_to_registers(PyObject* self, PyObject* args) {
    /*
     *  From cudaize.py
     *  def copy_to_shared(start_loop: str, array_name:str, alignment: int)
     */

    strict_arg_num(args, 2, "copy_to_registers");
    int                         stmt                    = 0;
    auto                        start_loop              = strArg(args, 0);
    auto                        array_name              = strArg(args, 1);
    std::vector<std::string>    cur_idxs;
    cur_indices(stmt, cur_idxs);

    int                         level_tx                = find_cur_level(stmt, "tx");
    int                         level_ty                = find_cur_level(stmt, "ty");

    std::string                 ty_lookup_idx           = "";

    //               //
    // Find ty index //
    //               //
    if(level_ty > -1 && !cur_idxs[level_ty].empty()) {
        ty_lookup_idx = cur_idxs[level_ty];
        myloop->tile_cuda(stmt, level_ty, level_tx + 1);
    }
    else {
        // Not sure if correct
        ty_lookup_idx = cur_idxs[cur_idxs.size() - 1];
    }

    cur_indices(stmt, cur_idxs);

    //               //
    // Find tx index //
    //               //
    level_tx = find_cur_level(stmt, "tx");

    //                 //
    // ReFind ty index //
    //                 //
    if(!ty_lookup_idx.empty()) {
        level_ty = find_cur_level(stmt, ty_lookup_idx.c_str());
    }

    // Do something //
    int idx_flag = -1;
    if(!ty_lookup_idx.empty()) {
        for(int n = level_ty; n < cur_idxs.size(); n++) {
            if(!cur_idxs[n].empty()) {
                idx_flag = find_cur_level(stmt, cur_idxs[n]);
                break;
            }
        }
    }

    int how_many_levels = 0;
    int start_idx = idx_flag;
    if(idx_flag == -1) {
        start_idx = 0;
    }
    for(int ch_lev = start_idx; ch_lev < cur_idxs.size(); ch_lev++) {
        if(!cur_idxs[ch_lev].empty()) {
            how_many_levels++;
        }
    }

    if(how_many_levels == 0) {
        while(idx_flag >= 0) {
            for(int n = level_ty + 1; n < cur_idxs.size(); n++) {
                if(!cur_idxs[n].empty()) {
                    std::string idx = cur_idxs[n];
                    auto idx_level = find_cur_level(stmt, idx);

                    myloop->tile_cuda(stmt, idx_level, idx_level);

                    idx_level = find_cur_level(stmt, idx);
                    myloop->tile_cuda(stmt, idx_level, idx_level);

                    cur_indices(stmt, cur_idxs);
                    level_tx = find_cur_level(stmt, "tx");
                    level_ty = find_cur_level(stmt, ty_lookup_idx);
                    idx_flag = -1;

                    for(int n2 = level_ty; n2 < cur_idxs.size(); n2++) {
                        if(!cur_idxs[n2].empty()) {
                            idx_flag = find_cur_level(stmt, cur_idxs[n2]);
                            break;
                        }
                    }
                }
            }
        }
    }

    int start_level = find_cur_level(stmt, start_loop);
    std::vector<std::string> blocks;  block_indices(blocks);        // block index names
    std::vector<std::string> threads; thread_indices(threads);      // thread index names

    std::vector<int> privatized_levels;
    for(auto idx: blocks) {
        auto block_level = find_cur_level(stmt, idx);
        if(block_level >= start_level) {
            privatized_levels.push_back(block_level);
        }
    }

    for(auto idx: threads) {
        auto thread_level = find_cur_level(stmt, idx);
        if(thread_level >= start_level) {
            privatized_levels.push_back(thread_level);
        }
    }

    myloop->datacopy_privatized_cuda(
            stmt, start_level, array_name, privatized_levels, false, -1, 1, 1, false);


    Py_RETURN_NONE;
}

static PyObject* chill_copy_to_shared(PyObject* self, PyObject* args) {
    /*
     *  From cudaize.py
     *  def copy_to_shared(start_loop: str, array_name:str, alignment: int)
     */

    strict_arg_num(args, 3, "copy_to_shared");
    int                     stmt                    = 0;
    std::string             start_loop              = strArg(args, 0);
    int                     start_level             = find_cur_level(stmt, start_loop);
    std::string             array_name              = strArg(args, 1);
    int                     alignment               = intArg(args, 2);

    int                     old_num_stmts           = myloop->stmt.size();

    myloop->datacopy_cuda(stmt, start_level, array_name, {"tmp1", "tmp2"}, 0, 0, 1, alignment, 1);
    myloop->addSync(stmt, start_loop);

    int                     new_num_stmts           = myloop->stmt.size();

    // Foreach new stmt added from by data copy & sync...
    for(int new_stmt = old_num_stmts; new_stmt < new_num_stmts; new_stmt++) {
        int level_tmp2 = find_cur_level(new_stmt, "tmp2");
        if(level_tmp2 != -1) {
            // --------------------------- //
            // datacopy created two levels //
            // --------------------------- //
            int upper_bound, lower_bound;

            myloop->tile_cuda(new_stmt, level_tmp2, level_tmp2);
            myloop->extractCudaUB(new_stmt, level_tmp2, upper_bound, lower_bound);

            int level_tmp1 = find_cur_level(new_stmt, "tmp1");
                level_tmp2 = find_cur_level(new_stmt, "tmp2");
            if(upper_bound + 1 == myloop->cu_tx && myloop->cu_ty == 1) {
                // Upper bound and tx are the same AND only single thread dimensions

                // just move 2nd loop up
                myloop->tile_cuda(new_stmt, level_tmp2, 1, level_tmp1, "tx", "tx", TilingMethodType::CountedTile);
            }
            else {
                // Upper bound and tx are not the same OR multi thread dimensions

                // move 2nd loop up
                myloop->tile_cuda(new_stmt, level_tmp2, 1, level_tmp1, "tx", "tx", TilingMethodType::CountedTile);

                // update parapmeters
                int upper_bound_tx,   lower_bound_tx;
                int upper_bound_tmp1, lower_bound_tmp1;
                    level_tmp1 = find_cur_level(new_stmt, "tmp1");
                    level_tmp2 = find_cur_level(new_stmt, "tmp2");
                int level_tx   = find_cur_level(new_stmt, "tx");
                myloop->extractCudaUB(new_stmt, level_tmp1, upper_bound_tmp1, lower_bound_tmp1);
                myloop->extractCudaUB(new_stmt, level_tx,   upper_bound_tx,   lower_bound_tx);

                // Handle tx thread

                if(upper_bound_tx + 1 > myloop->cu_tx) {
                    myloop->tile_cuda(new_stmt, level_tx, myloop->cu_tx, level_tx, "tx", "tmp_tx", TilingMethodType::CountedTile);

                    auto repeat = find_cur_level(new_stmt, "tx");
                    myloop->tile_cuda(new_stmt, repeat, repeat);

                    if(find_cur_level(new_stmt, "tx") > find_cur_level(new_stmt, "tmp_tx")) {
                        myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "tx"), find_cur_level(new_stmt, "tmp_tx"));
                    }
                }

                myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "tmp1"), find_cur_level(new_stmt, "tmp1"));

                // update parameters
                int upper_bound_ty, lower_bound_ty;
                int level_ty = find_cur_level(new_stmt, "tmp1");
                    level_tx = find_cur_level(new_stmt, "tx");
                myloop->extractCudaUB(new_stmt, level_ty, upper_bound_ty, lower_bound_ty);
                myloop->extractCudaUB(new_stmt, level_tx, upper_bound_tx, lower_bound_tx);

                // Handle ty thread

                if(upper_bound_ty + 1 > myloop->cu_ty) {
                    myloop->tile_cuda(new_stmt, level_ty, myloop->cu_ty, level_ty, "ty", "tmp_ty", TilingMethodType::CountedTile);
                    myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "ty"), find_cur_level(new_stmt, "ty"));

                    std::vector<std::string> cur_idxs;
                    cur_indices(new_stmt, cur_idxs);

                    // Put ty before any tmp_tx
                    if(in<std::string>(cur_idxs, "tmp_tx")) {
                        if(find_cur_level(new_stmt, "ty") > find_cur_level(new_stmt, "tmp_tx")) {
                            // Swap level of ty and tmp_ty (I think)
                            myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "ty"), find_cur_level(new_stmt, "tmp_tx"));
                        }
                    }

                    // Put ty before any tmp_ty
                    if(in<std::string>(cur_idxs, "tmp_ty")) {
                        if(find_cur_level(new_stmt, "ty") > find_cur_level(new_stmt, "tmp_ty")) {
                            myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "ty"), find_cur_level(new_stmt, "tmp_ty"));
                        }
                    }
                }
                else {
                    myloop->tile_cuda(new_stmt, level_ty, 1, level_ty, "ty", "ty", TilingMethodType::CountedTile);
                    myloop->tile_cuda(new_stmt, find_cur_level(new_stmt, "ty"), find_cur_level(new_stmt, "tx") + 1);
                }
            }
        }
        else {
            // -------------------------- //
            // datacopy created one level //
            // -------------------------- //
            int level_tmp1          = find_cur_level(new_stmt, "tmp1");         // first level
            int lower_bound, upper_bound;

            myloop->tile_cuda(new_stmt, level_tmp1, level_tmp1);
            myloop->extractCudaUB(new_stmt, level_tmp1, upper_bound, lower_bound);

            if(upper_bound + 1 == myloop->cu_tx) {
                // Same upper bound as thread tx
                myloop->renameIndex(new_stmt, "tmp1", "tx");
            }
            else {
                // Different upper bound as thread tx, so will require some tiling
                myloop->tile_cuda(new_stmt, level_tmp1,     myloop->cu_tx, level_tmp1,     "tx", "tmp_tx", TilingMethodType::CountedTile);
                myloop->tile_cuda(new_stmt, level_tmp1 + 1, 1,             level_tmp1 + 1, "tx", "tx",     TilingMethodType::CountedTile);
                myloop->tile_cuda(new_stmt, level_tmp1 + 1, level_tmp1);

                if(myloop->cu_ty > 1) {
                    myloop->extractCudaUB(new_stmt, level_tmp1 + 1, upper_bound, lower_bound);
                    auto bound = (int) std::ceil((float) (upper_bound + 1) / (float) myloop->cu_ty);
                    myloop->tile_cuda(new_stmt, level_tmp1 + 1, bound, level_tmp1 + 1, "tmp_ty", "ty", TilingMethodType::CountedTile);

                    //TODO: tz ???
                }
            }

        }
        myloop->addSync(new_stmt, start_loop);
    }

    Py_RETURN_NONE;
}

static PyObject *
chill_cur_indices(PyObject *self, PyObject *args)
{
  int stmt_num = -123;
  if (!PyArg_ParseTuple(args, "i", &stmt_num)){
    chill_fprintf(stderr, "chill_cur_indices, can't parse statement number passed from python\n");
    exit(-1);
  }
  
  int num = myloop->idxNames[stmt_num].size();

  PyObject *pylist, *item;
  pylist = PyList_New(num);
  for (int i=0; i<num; i++) {
    item = PyString_FromString(myloop->idxNames[stmt_num][i].c_str());
    PyList_SetItem(pylist, i, item);
  }
  return pylist;
}


static PyObject *
chill_block_indices(PyObject *self, PyObject *args) {
  
  // I'm unsure what the legal states are here
  // is it always "bx", or  ("bx" and "by") ?
  int howmany = 0;
  char *loopnames[2]; 
  if (myloop->cu_bx > 1) {
    loopnames[howmany] = strdup("bx");
    howmany++;
  }
  if (myloop->cu_by > 1) {
    loopnames[howmany] = strdup("by");
    howmany++;
  }
  
  if (howmany == 0) return Py_BuildValue("()");
  if (howmany == 1) return Py_BuildValue("(s)", loopnames[0]);
  if (howmany == 2) return Py_BuildValue("(ss)", loopnames[0], loopnames[1]);
  debug_fprintf(stderr, "chill_block_indices(), gonna die, howmany == %d", howmany);
  exit(666);
  
  Py_RETURN_NONE;
}

static PyObject *
chill_thread_indices(PyObject *self, PyObject *args) {
  
  // I'm unsure what the legal states are here
  // is it always "tx", or  ("tx" and "ty") or ("tx" and "ty" and "tz") ?
  int howmany = 0;
  char *loopnames[3];
  if (myloop->cu_tx > 1) {
    loopnames[howmany++] = strdup("tx");
  }
  if (myloop->cu_ty > 1) {
    loopnames[howmany++] = strdup("ty");
  }
  if (myloop->cu_tz > 1) {
    loopnames[howmany++] = strdup("tz");
  }
  
  if (howmany == 0) return Py_BuildValue("()");
  if (howmany == 1) return Py_BuildValue("(s)",   
                                         loopnames[0]);
  if (howmany == 2) return Py_BuildValue("(ss)",  
                                         loopnames[0], 
                                         loopnames[1]);
  if (howmany == 3) return Py_BuildValue("(sss)", 
                                         loopnames[0],
                                         loopnames[1],
                                         loopnames[2]);
  
  debug_fprintf(stderr, "chill_thread_indices(), gonna die, howmany == %d", howmany);
  exit(999);
}





static PyObject *
block_dims(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("block_dims() returning %d %d\n", myloop->cu_bx, myloop->cu_by);
  Py_BuildValue( "i i", myloop->cu_bx, myloop->cu_by);
}


static PyObject *
thread_dims(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("thread_dims() returning %d %d %d\n", 
  //myloop->cu_tx, myloop->cu_ty, myloop->cu_tz);
  
  Py_BuildValue( "i i i", myloop->cu_tx, myloop->cu_ty, myloop->cu_tz);
}


static PyObject *
chill_hard_loop_bounds(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("hard_loop_bounds("); 
  int sstmt, level;  // input parameters
  int upper, lower;  // output
  
  if (!PyArg_ParseTuple(args, "ii", &sstmt, &level)){
    debug_fprintf(stderr, "hard_loop_bounds, ");
    debug_fprintf(stderr, "can't parse statement numbers passed from python\n");
    exit(-1);
  }
  //DEBUG_PRINT(" %d, %d )\n", sstmt, level); 
  
  myloop->extractCudaUB(sstmt, level, upper, lower); 
  
  //DEBUG_PRINT("lower %d  upper %d\n", lower, upper);
  
  Py_BuildValue( "i i", lower, upper);
}


static PyObject *
chill_datacopy9(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("\n\n\n*****  datacopy_v2()  9ARGS\n"); 
  
  int sstmt;
  int level;
  std::string cppstr;
  std::string array_name;
  std::vector<std::string> new_idxs;
  bool allow_extra_read;
  int fastest_changing_dimension;
  int padding_stride;
  int padding_alignment;
  bool cuda_shared; 
  
  PyObject     *pyObj;

  if (!PyArg_ParseTuple( args, "O", &pyObj)) { // everything on a single tuple
    
    debug_fprintf(stderr, "failed to parse tuple\n");
    exit(-1);
  }
  Py_XINCREF( pyObj );
  
  //if (PyList_Check(pyObj))  debug_fprintf(stderr, "it's a list\n");
  //if (PyTuple_Check(pyObj)) debug_fprintf(stderr, "it's a tuple\n");
  
  
  
  // the ONLY arg is a tuple. figure out how big it is 
  int tupleSize = PyTuple_Size(pyObj);
  //DEBUG_PRINT("%d things in object tuple\n", tupleSize); 
  
  // first has to be the statement number
  PyObject *tupleItem1 = PyTuple_GetItem(pyObj, 0); 
  Py_INCREF(tupleItem1);
  if (PyInt_Check( tupleItem1)) sstmt = PyInt_AsLong( tupleItem1 );
  else {
    debug_fprintf(stderr, "second tuple item in chill_datacopy9 is not an int?\n");
    exit(-1);
  }
  //DEBUG_PRINT("stmt %d\n", sstmt);
  
  PyObject *tupleItem2 = PyTuple_GetItem(pyObj, 1);  // second item is level
  Py_INCREF(tupleItem2);
  if (PyInt_Check( tupleItem2 )) level = PyInt_AsLong( tupleItem2);
  else {
    debug_fprintf(stderr, "second tuple item in chill_datacopy9 is not an int?\n");
    exit(-1);
  }
  //DEBUG_PRINT("level %d\n", level );
  
  // third item is array name 
  PyObject *tupleItem3 = PyTuple_GetItem(pyObj, 2);  
  Py_INCREF(tupleItem3);
  array_name = strdup(PyString_AsString(tupleItem3)); 
  //DEBUG_PRINT("array name '%s'\n", array_name.c_str()); 
  
  
  // integer number of indices  
  PyObject *tupleItem4 = PyTuple_GetItem(pyObj, 3); 
  Py_INCREF(tupleItem4);  
  int numindex= PyInt_AsLong( tupleItem4 );
  //DEBUG_PRINT("%d indices\n", numindex); 
  
  
  PyObject *tupleItemTEMP;
  for (int i=0; i<numindex; i++)  {
    tupleItemTEMP = PyTuple_GetItem(pyObj, 4+i);
    Py_INCREF(tupleItemTEMP);
    cppstr = strdup(PyString_AsString(tupleItemTEMP)); 
    new_idxs.push_back( cppstr );
    //DEBUG_PRINT("%s\n", cppstr.c_str());
  }
  
  PyObject *tupleItem5 = PyTuple_GetItem(pyObj, 4+numindex);
  Py_INCREF(tupleItem5);
  allow_extra_read = PyInt_AsLong( tupleItem5 );
  
  PyObject *tupleItem6 = PyTuple_GetItem(pyObj, 5+numindex);
  Py_INCREF(tupleItem6);
  fastest_changing_dimension = PyInt_AsLong( tupleItem6 );
  
  PyObject *tupleItem7 = PyTuple_GetItem(pyObj, 6+numindex);
  Py_INCREF(tupleItem7);
  padding_stride = PyInt_AsLong( tupleItem7 );
  
  PyObject *tupleItem8 = PyTuple_GetItem(pyObj, 7+numindex);
  Py_INCREF(tupleItem8);
  padding_alignment = PyInt_AsLong( tupleItem8 );
  
  PyObject *tupleItem9 = PyTuple_GetItem(pyObj, 8+numindex);
  Py_INCREF(tupleItem9);
  cuda_shared = PyInt_AsLong( tupleItem9 );
  
  
  //DEBUG_PRINT("calling myloop->datacopy_cuda()\n");      
  
  // corruption happenes in here??? 
  myloop->datacopy_cuda(sstmt, level, array_name, new_idxs,
                        allow_extra_read, fastest_changing_dimension,
                        padding_stride, padding_alignment, cuda_shared);
  
  debug_fprintf(stderr, "before attempt (after actual datacopy)\n"); 
  //myloop->printCode(); // attempt to debug 
  debug_fprintf(stderr, "back from attempt\n"); 
  
  //DEBUG_PRINT("datacopy_9args returning\n"); 
  
  Py_RETURN_NONE;  
}





static PyObject *
chill_datacopy_privatized(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("C datacopy_privatized\n"); 
  PyObject *pyObj;
  if (!PyArg_ParseTuple( args, "O", &pyObj)) { // everything on a single tuple
    debug_fprintf(stderr, "failed to parse tuple\n");
    exit(-1);
  }
  
  PyObject *tupleItem = PyTuple_GetItem(pyObj, 0); //  statement number
  Py_XINCREF(tupleItem);
  int sstmt = PyInt_AsLong( tupleItem );
  
  tupleItem = PyTuple_GetItem(pyObj, 1);  // start_loop
  Py_XINCREF(tupleItem);
  std::string start_loop = strdup(PyString_AsString(tupleItem));
  int level = myloop->findCurLevel(sstmt, start_loop);
  
  
  tupleItem = PyTuple_GetItem(pyObj, 2);  // array_name
  Py_XINCREF(tupleItem);
  std::string array_name = strdup(PyString_AsString(tupleItem));
  
  // things to hold constant  - first a count, then the things
  tupleItem = PyTuple_GetItem(pyObj, 3);  // how many things in the array
  Py_XINCREF(tupleItem);
  int howmany = PyInt_AsLong( tupleItem );
  
  //DEBUG_PRINT("%d things to hold constant: ", howmany);
  std::vector<std::string> holdconstant;
  std::string cppstr;
  
  for (int i=0; i<howmany; i++) {
    tupleItem = PyTuple_GetItem(pyObj, 4+i);
    Py_XINCREF(tupleItem);
    cppstr = strdup(PyString_AsString(tupleItem));
    holdconstant.push_back( cppstr );   // add at end
  }
  
  std::vector<int> privatized_levels(howmany);  
  for(int i=0; i<howmany; i++) { 
    privatized_levels[i] = myloop->findCurLevel(sstmt, holdconstant[i]);
    //DEBUG_PRINT("privatized_levels[ %d ] = %d\n", i, privatized_levels[i] );
  }
  
  bool allow_extra_read = false;
  int fastest_changing_dimension = -1; 
  int padding_stride = 1;
  int padding_alignment = 1;
  bool cuda_shared = false;
  
  
  myloop->datacopy_privatized_cuda(sstmt, level, array_name, privatized_levels,
                                   allow_extra_read, fastest_changing_dimension,
                                   padding_stride, padding_alignment, 
                                   cuda_shared);
  
  
  Py_RETURN_NONE;
}






static PyObject *
chill_unroll(PyObject *self, PyObject *args)
{
  int sstmt, level, unroll_amount;
  
  if (!PyArg_ParseTuple(args, "iii", &sstmt, &level, &unroll_amount)) {
    debug_fprintf(stderr, "chill_unroll, can't parse parameters passed from python\n");
    exit(-1);
  }
  
  //DEBUG_PRINT("chill_unroll( %d, %d, %d)\n", sstmt, level, unroll_amount );
  bool does_expand = myloop->unroll_cuda(sstmt,level,unroll_amount);
  
  // TODO return the boolean?
  Py_RETURN_NONE;
}


//#if 0
static PyObject* chill_flatten(PyObject* self, PyObject* args) {
    int                 stmt                = intArg(args, 0);
    std::string         idxs                = strArg(args, 1);
    std::vector<int>    loop_levels;
    std::string         inspector_name      = strArg(args, 3);

    to_int_vector(args, 2, loop_levels);
    
    myloop->flatten_cuda(stmt, idxs, loop_levels, inspector_name);
    Py_RETURN_NONE;
}
//#endif

//#if 0
static PyObject* chill_compact(PyObject* self, PyObject* args) {
    int                 stmt                = intArg(args, 0);
    int                 loop_level          = intArg(args, 1);
    std::string         new_array           = strArg(args, 2);
    int                 zero                = intArg(args, 3);
    std::string         data_array          = strArg(args, 4);
    
    myloop->compact_cuda(stmt, loop_level, new_array, zero, data_array);
    Py_RETURN_NONE;
}
//#endif

static PyObject* chill_make_dense(PyObject* self, PyObject* args) {
    int                 stmt                = intArg(args, 0);
    int                 loop_level          = intArg(args, 1);
    std::string         new_loop            = strArg(args, 2);
    
    myloop->make_dense_cuda(stmt, loop_level, new_loop);
    Py_RETURN_NONE;
}


static PyObject *
chill_cudaize_v2(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("chill_cudaize_v2\n"); 
  PyObject *pyObj;
  if (!PyArg_ParseTuple( args, "O", &pyObj)) { // everything on a single tuple
    debug_fprintf(stderr, "failed to parse tuple\n");
    exit(-1);
  }
  
  // the ONLY arg is a tuple. figure out how big it is 
  int tupleSize = PyTuple_Size(pyObj);
  //DEBUG_PRINT("%d things in tuple\n", tupleSize); 
  
  PyObject *tupleItem = PyTuple_GetItem(pyObj, 0);  //the kernel name
  Py_XINCREF(tupleItem);
  std::string kernel_name = strdup(PyString_AsString(tupleItem)); 
  
  std::map<std::string, int> array_sizes;
  tupleItem = PyTuple_GetItem(pyObj, 1); // number of array sizes
  Py_XINCREF(tupleItem);
  int numarraysizes = PyInt_AsLong( tupleItem );
  
  std::string cppstr;
  int offset = 2;
  for (int i=0; i<numarraysizes; i++) {
    tupleItem = PyTuple_GetItem(pyObj, offset++);
    Py_XINCREF(tupleItem);
    cppstr = strdup(PyString_AsString(tupleItem));
    tupleItem = PyTuple_GetItem(pyObj, offset++); // integer size
    int siz = PyInt_AsLong( tupleItem );
    
    //DEBUG_PRINT("arraysize for %s = %d\n", cppstr.c_str(), siz); 
    array_sizes.insert( std::make_pair( cppstr, siz )); 
  }
  
  
  std::vector<std::string> blockIdxs;
  tupleItem = PyTuple_GetItem(pyObj, offset++); // integer number of blocks
  Py_XINCREF(tupleItem);
  int numblocks = PyInt_AsLong( tupleItem );
  //DEBUG_PRINT("%d blocks\n", numblocks);
  for (int i=0; i<numblocks; i++)  {
    tupleItem = PyTuple_GetItem(pyObj, offset++);
    cppstr = strdup(PyString_AsString(tupleItem));
    blockIdxs.push_back( cppstr );
    //DEBUG_PRINT("%s\n", cppstr.c_str());
  }
  
  std::vector<std::string> threadIdxs;
  tupleItem = PyTuple_GetItem(pyObj, offset++); // integer number of threads
  Py_XINCREF(tupleItem);
  int numthreads= PyInt_AsLong( tupleItem );
  //DEBUG_PRINT("%d threads\n", numthreads);
  for (int i=0; i<numthreads; i++)  {
    tupleItem = PyTuple_GetItem(pyObj, offset++);
    Py_XINCREF(tupleItem);
    cppstr = strdup(PyString_AsString(tupleItem));
    threadIdxs.push_back( cppstr );
    //DEBUG_PRINT("%s\n", cppstr.c_str());
  }
  
  
  myloop->cudaize_v2(kernel_name, array_sizes, blockIdxs, threadIdxs);
  
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}


static PyObject* chill_cudaize_v3(PyObject *self, PyObject *args)
{ 
  int stmt_num            = intArg(args, 0);
  std::string kernel_name = strArg(args, 1);
  int offset = 3;
  
  std::map<std::string, int> array_sizes;
  readArg(args, 2, array_sizes);
  
  std::vector<std::string> blockIdxs;
  to_string_vector(args, 3, blockIdxs);
  
  std::vector<std::string> threadIdxs;
  to_string_vector(args, 4, threadIdxs);
  
  std::vector<std::string> kernel_parameters;
  to_string_vector(args, 5, kernel_parameters);
  
  
  myloop->cudaize_v3(stmt_num, kernel_name, array_sizes, blockIdxs, threadIdxs, kernel_parameters);
  
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
}



static PyObject *get_loop_num()  { 
  // TODO get_loop_num()     it's a global value?
  debug_fprintf(stderr, "get_loop_num()  UNIMPLEMENTED\n");
  exit(-1);
}




static PyObject *
chill_copy_to_texture(PyObject *self, PyObject *args)
{
  //DEBUG_PRINT("C copy_to_texture() called from python \n"); 
  const char *array_name;
  if (!PyArg_ParseTuple(args, "s", &array_name)){
    debug_fprintf(stderr, "chill_copy_to_texture can't parse array name\n");
    exit(-1);
  }
  //DEBUG_PRINT("array name = %s\n", array_name);
  myloop->copy_to_texture(array_name);
  
  Py_RETURN_NONE; 
}




static PyObject *
chill_init(PyObject *self, PyObject *args)
{
  const char *filename;
  const char *procname;
  if (!PyArg_ParseTuple(args, "ss", &filename, &procname)){
    debug_fprintf(stderr, "umwut? can't parse file name and procedure name?\n");
    exit(-1);
  }
  
  int loop_num = 0;
  if(dest_filename.empty()) {
    ir_code = new IR_cudaChillCode(parser, filename, procname, NULL);
  }
  else {
    ir_code = new IR_cudaChillCode(parser, filename, procname, dest_filename.c_str());
  }

  fflush(stdout);

  //protonu--here goes my initializations
  //A lot of this code was lifted from Chun's parser.yy
  //the plan is now to create the LoopCuda object directly
  IR_Block *block = ir_code->GetCode();
  debug_fprintf(stderr, "ir_code->FindOneLevelControlStructure(block); chillmodule.cc\n"); 
  ir_controls = ir_code->FindOneLevelControlStructure(block);
  
  int loop_count = 0;
  for (int i = 0; i < ir_controls.size(); i++) {
    if (ir_controls[i]->type() == IR_CONTROL_LOOP) {
      loops.push_back(i);
      loop_count++;
    }
  }
  delete block;
  
  
  std::vector<IR_Control *> parm;
  for(int j = 0; j < loop_count; j++)
    parm.push_back(ir_controls[loops[j]]);
  
  
  debug_fprintf(stderr, "block = ir_code->MergeNeighboringControlStructures(parm);\n"); 
  block = ir_code->MergeNeighboringControlStructures(parm);
  
  //DEBUG_PRINT("myloop = new LoopCuda(block, loop_num); in chillmodule.cc\n"); 
  myloop = new LoopCuda(block, loop_num);
  fflush(stdout); debug_fprintf(stderr, "back\n"); 
  delete block;
  
  //end-protonu
  
  fflush(stdout);
  debug_fprintf(stderr, "myloop->original();\n"); 
  myloop->original();
  fflush(stdout);
  debug_fprintf(stderr, "myloop->useIdxNames=true;\n"); 
  myloop->useIdxNames=true;//Use idxName in code_gen
  //register_v2(L);
  
  fflush(stdout);
  debug_fprintf(stderr, "chill_init DONE\n"); 
  Py_RETURN_NONE;  // return Py_BuildValue( "" );
  
}

#else
// ------------------------- //
// CHiLL interface functions //
// ------------------------- //

static PyObject* chill_source(PyObject* self, PyObject* args) {
  strict_arg_num(args, 1, "source");
  source_filename = strArg(args, 0);
  Py_RETURN_NONE;
}

static PyObject* chill_procedure(PyObject* self, PyObject* args) {
  if(!procedure_name.empty()) {
    chill_fprintf(stderr, "only one procedure can be handled in a script");
    if(!is_interactive)
      exit(2);
  }
  procedure_name = strArg(args, 0);
  Py_RETURN_NONE;
}

static PyObject *
chill_destination(PyObject *self, PyObject* args) {
  //strict_arg_num(args, 1, "destination");
  //((IR_roseCode*)ir_code)->setOutputName(strArg(args, 0).c_str());
  dest_filename = strArg(args, 0);
  Py_RETURN_NONE;
}

static PyObject* chill_effort(PyObject *self, PyObject *args) {
  strict_arg_num(args, 1, "effort");
  effort = intArg(args, 0);
  Py_RETURN_NONE;
}

static PyObject* chill_loop(PyObject* self, PyObject* args) {
  // loop (n)
  // loop (n:m)
  
  int nargs = PyTuple_Size(args);
  int start_num;
  int end_num;
  if(nargs == 1) {
    start_num = intArg(args, 0);
    end_num = start_num;
  }
  else if(nargs == 2) {
    start_num = intArg(args, 0);
    end_num = intArg(args, 1);
  }
  else {
    chill_fprintf(stderr, "loop takes one or two arguments");
    if(!is_interactive)
      exit(2);
  }
  init_loop(start_num, end_num);
  Py_RETURN_NONE;
}

static PyObject* chill_source_procedure_loop(PyObject* self, PyObject* args) {
  int nargs = strict_arg_range(args, 2, 5, "init");
  source_filename = strArg(args, 0);
  procedure_name = strArg(args, 1);
  int start_num = 0;
  int end_num = 0;
  switch(nargs) {
    case 2:
      start_num = 0;
      end_num = 0;
      break;
    case 3:
      start_num = intArg(args, 2);
      end_num = intArg(args, 2);
      break;
    case 4:
      start_num = intArg(args, 2);
      end_num = intArg(args, 3);
      break;
  }
  set_loop_num_start(start_num);
  set_loop_num_end(end_num);
  init_loop(start_num, end_num);
  Py_RETURN_NONE;
}

static PyObject* chill_print_code(PyObject* self, PyObject* args) {
  strict_arg_num(args, 0, "print_code");
  myloop->printCode(effort);
  chill_printf("\n");
  Py_RETURN_NONE;
}

static PyObject* chill_print_dep(PyObject* self, PyObject* args) {
  strict_arg_num(args, 0, "print_dep");
  myloop->printDependenceGraph();
  Py_RETURN_NONE;
}

static PyObject* chill_print_space(PyObject* self, PyObject* args) {
  strict_arg_num(args, 0, "print_space");
  myloop->printIterationSpace();
  Py_RETURN_NONE;
}

static void add_known(std::string cond_expr) {
  int num_dim = myloop->known.n_set();
  std::vector<std::map<std::string, int> >* cond;
  cond = parse_relation_vector(cond_expr.c_str());
  
  Relation rel(num_dim);
  F_And *f_root = rel.add_and();
  for (int j = 0; j < cond->size(); j++) {
    GEQ_Handle h = f_root->add_GEQ();
    for (std::map<std::string, int>::iterator it = (*cond)[j].begin(); it != (*cond)[j].end(); it++) {
      try {
        int dim = std::stoi(it->first.c_str());
        h.update_const(it->second);
      }
      catch (std::invalid_argument &e) {
        Free_Var_Decl *g = NULL;
        for (unsigned i = 0; i < myloop->freevar.size(); i++) {
          std::string name = myloop->freevar[i]->base_name();
          if (name == it->first) {
            g = myloop->freevar[i];
            break;
          }
        }
        if (g == NULL)
          throw std::runtime_error("symbolic variable " + it->first + " not found");
        else
          h.update_coef(rel.get_local(g), it->second);
      }
    }
  }
  myloop->addKnown(rel);
}

static PyObject* chill_known(PyObject* self, PyObject* args) {
  strict_arg_num(args, 1, "known");
  if (PyList_Check(PyTuple_GetItem(args, 0))) {
    PyObject* list = PyTuple_GetItem(args, 0);
    for (int i = 0; i < PyList_Size(list); i++) {
      add_known(std::string(PyString_AsString(PyList_GetItem(list, i))));
    }
  }
  else {
    add_known(strArg(args, 0));
  }
  Py_RETURN_NONE;
}

static PyObject* chill_remove_dep(PyObject* self, PyObject* args) {
  strict_arg_num(args, 0, "remove_dep");
  int from = intArg(args, 0);
  int to = intArg(args, 1);
  myloop->removeDependence(from, to);
  Py_RETURN_NONE;
}

static PyObject* chill_original(PyObject* self, PyObject* args) {
  strict_arg_num(args, 0, "original");
  myloop->original();
  Py_RETURN_NONE;
}

static PyObject* chill_permute(PyObject* self, PyObject* args) {
  int nargs = strict_arg_range(args, 1, 3, "permute");
  if((nargs < 1) || (nargs > 3))
    throw std::runtime_error("incorrect number of arguments in permute");
  if(nargs == 1) {
    // premute ( vector )
     std::vector<int> pi;
    if(!to_int_vector(args, 0, pi))
      throw std::runtime_error("first arg in permute(pi) must be an int vector");
    myloop->permute(pi);
  }
  else if (nargs == 2) {
    // permute ( set, vector )
    std::set<int> active;
    std::vector<int> pi;
    if(!to_int_set(args, 0, active))
      throw std::runtime_error("the first argument in permute(active, pi) must be an int set");
    if(!to_int_vector(args, 1, pi))
      throw std::runtime_error("the second argument in permute(active, pi) must be an int vector");
     myloop->permute(active, pi);
  }
  else if (nargs == 3) {
    int stmt_num = intArg(args, 0);
    int level = intArg(args, 1);
    std::vector<int> pi;
    if(!to_int_vector(args, 2, pi))
      throw std::runtime_error("the third argument in permute(stmt_num, level, pi) must be an int vector");
    myloop->permute(stmt_num, level, pi);
  }
  Py_RETURN_NONE;
}

//static PyObject* chill_pragma(PyObject* self, PyObject* args) {
//  strict_arg_num(args, 3, "pragma");
//  int stmt_num = intArg(args, 1);
//  int level = intArg(args, 1);
//  std::string pragmaText = strArg(args, 2);
//  myloop->pragma(stmt_num, level, pragmaText);
//  Py_RETURN_NONE;
//}

static PyObject* chill_prefetch(PyObject* self, PyObject* args) {
  strict_arg_num(args, 3, "prefetch");
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  std::string prefetchText = strArg(args, 2);
  int hint = intArg(args, 3);
  myloop->prefetch(stmt_num, level, prefetchText, hint);
  Py_RETURN_NONE;
}

static PyObject* chill_tile(PyObject* self, PyObject* args) {
  int nargs = strict_arg_range(args, 3, 7, "tile");
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  int tile_size = intArg(args, 2);
  if(nargs == 3) {
    myloop->tile(stmt_num, level, tile_size);
  }
  else if(nargs >= 4) {
    int outer_level = intArg(args, 3);
    if(nargs >= 5) {
      TilingMethodType method = StridedTile;
      int imethod = intArg(args, 4, 2); //< don't know if a default value is needed
      // check method input against expected values
      if (imethod == 0)
        method = StridedTile;
      else if (imethod == 1)
        method = CountedTile;
      else
        throw std::runtime_error("5th argument must be either strided or counted");
      if(nargs >= 6) {
        int alignment_offset = intArg(args, 5);
        if(nargs == 7) {
          int alignment_multiple = intArg(args, 6, 1);
          myloop->tile(stmt_num, level, tile_size, outer_level, method, alignment_offset, alignment_multiple);
        }
        if(nargs == 6)
          myloop->tile(stmt_num, level, tile_size, outer_level, method, alignment_offset);
      }
      if(nargs == 5)
        myloop->tile(stmt_num, level, tile_size, outer_level, method);
    }
  if(nargs == 4)
    myloop->tile(stmt_num, level, tile_size, outer_level);
  }
  Py_RETURN_NONE;
}

static void chill_datacopy_vec(PyObject* args) {
  // Overload 1: bool datacopy(
  //    const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums,
  //    int level,
  //    bool allow_extra_read = false,
  //    int fastest_changing_dimension = -1,
  //    int padding_stride = 1,
  //    int padding_alignment = 4,
  //    int memory_type = 0);
  std::vector<std::pair<int, std::vector<int> > > array_ref_nums;
  // expect list(tuple(int,list(int)))
  // or dict(int,list(int))
  if(PyList_CheckExact(PyTuple_GetItem(args, 0))) {
    PyObject* list = PyTuple_GetItem(args, 0);
    for(int i = 0; i < PyList_Size(list); i ++) {
      PyObject* tup = PyList_GetItem(list, i);
      int index = PyLong_AsLong(PyTuple_GetItem(tup, 0));
      std::vector<int> vec;
      to_int_vector(PyTuple_GetItem(tup, 1), vec);
      array_ref_nums.push_back(std::pair<int, std::vector<int> >(index, vec));
    }
  }
  else if(PyList_CheckExact(PyTuple_GetItem(args, 0))) {
    PyObject* dict = PyTuple_GetItem(args, 0);
    PyObject* klist = PyDict_Keys(dict);
    for(int ki = 0; ki < PyList_Size(klist); ki++) {
      PyObject* index = PyList_GetItem(klist, ki);
      std::vector<int> vec;
      to_int_vector(PyDict_GetItem(dict,index), vec);
      array_ref_nums.push_back(std::pair<int, std::vector<int> >(PyLong_AsLong(index), vec));
    }
    Py_DECREF(klist);
  }
  else {
    //TODO: this should never happen
  }
  int level = intArg(args, 1);
  bool allow_extra_read = boolArg(args, 2, false);
  int fastest_changing_dimension = intArg(args, 3, -1);
  int padding_stride = intArg(args, 4, 1);
  int padding_alignment = intArg(args, 5, 4);
  int memory_type = intArg(args, 6, 0);
  myloop->datacopy(array_ref_nums, level, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}

static void chill_datacopy_int(PyObject* args) {
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  std::string array_name = strArg(args,2,0);
  bool allow_extra_read = boolArg(args,3,false);
  int fastest_changing_dimension = intArg(args, 4, -1);
  int padding_stride = intArg(args, 5, 1);
  int padding_alignment = intArg(args, 6, 4);
  int memory_type = intArg(args, 7, 0);
  myloop->datacopy(stmt_num, level, array_name, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
}

static PyObject* chill_datacopy(PyObject* self, PyObject* args) {
  // Overload 2: bool datacopy(int stmt_num, int level, const std::string &array_name, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, int memory_type = 0);
  int nargs = strict_arg_range(args, 2, 7, "datacopy");
  if(PyList_CheckExact(PyTuple_GetItem(args,0)) || PyDict_CheckExact(PyTuple_GetItem(args, 0))) {
    chill_datacopy_vec(args);
  }
  else {
    chill_datacopy_int(args);
  }
  Py_RETURN_NONE;
}

static PyObject* chill_datacopy_privatized(PyObject* self, PyObject* args) {
  //  bool datacopy_privatized(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, int memory_type = 0);
  int nargs = strict_arg_range(args, 4, 9, "datacopy_privatized");
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  std::string array_name = strArg(args, 2);
  std::vector<int> privatized_levels;
  to_int_vector(args, 3, privatized_levels);
  bool allow_extra_read = boolArg(args, 4, false);
  int fastest_changing_dimension = intArg(args, 5, -1);
  int padding_stride = intArg(args, 6, 1);
  int padding_alignment = intArg(args, 7, 1);
  int memory_type = intArg(args, 8);
  myloop->datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
  Py_RETURN_NONE;
}

static PyObject* chill_unroll(PyObject* self, PyObject* args) {
  int nargs = strict_arg_range(args, 3, 4, "unroll");
  //std::set<int> unroll(int stmt_num, int level, int unroll_amount, std::vector< std::vector<std::string> >idxNames= std::vector< std::vector<std::string> >(), int cleanup_split_level = 0);
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  int unroll_amount = intArg(args, 2);
  std::vector< std::vector<std::string> > idxNames = std::vector< std::vector<std::string> >();
  int cleanup_split_level = intArg(args, 3);
  myloop->unroll(stmt_num, level, unroll_amount, idxNames, cleanup_split_level);
  Py_RETURN_NONE;
}
  
static PyObject* chill_unroll_extra(PyObject* self, PyObject* args) {
  int nargs = strict_arg_range(args, 3, 4, "unroll_extra");
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  int unroll_amount = intArg(args, 2);
  int cleanup_split_level = intArg(args, 3, 0);
  myloop->unroll_extra(stmt_num, level, unroll_amount, cleanup_split_level); 
  Py_RETURN_NONE;
}
  
static PyObject* chill_split(PyObject* self, PyObject* args) {
  strict_arg_num(args, 3, "split");
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  int num_dim = myloop->stmt[stmt_num].xform.n_out();
  
  std::vector<std::map<std::string, int> >* cond;
  std::string cond_expr = strArg(args, 2);
  cond = parse_relation_vector(cond_expr.c_str());
  
  Relation rel((num_dim-1)/2);
  F_And *f_root = rel.add_and();
  for (int j = 0; j < cond->size(); j++) {
    GEQ_Handle h = f_root->add_GEQ();
    for (std::map<std::string, int>::iterator it = (*cond)[j].begin(); it != (*cond)[j].end(); it++) {
      try {
        int dim = std::stoi(it->first.c_str());
        if (dim == 0)
          h.update_const(it->second);
        else {
          if (dim > (num_dim-1)/2)
            throw std::runtime_error("invalid loop level " + to_string(dim) + " in split condition");
          h.update_coef(rel.set_var(dim), it->second);
        }
      }
      catch (std::invalid_argument &e) {
        Free_Var_Decl *g = NULL;
        for (unsigned i = 0; i < myloop->freevar.size(); i++) {
          std::string name = myloop->freevar[i]->base_name();
          if (name == it->first) {
            g = myloop->freevar[i];
            break;
          }
        }
        if (g == NULL)
          throw std::runtime_error("unrecognized variable " + to_string(it->first.c_str()));
        h.update_coef(rel.get_local(g), it->second);
      }
    }
  }
  myloop->split(stmt_num,level,rel);
  Py_RETURN_NONE;
}

static PyObject* chill_nonsingular(PyObject* self, PyObject* args) {
  std::vector< std::vector<int> > mat;
  to_int_matrix(args, 0, mat);
  myloop->nonsingular(mat);
  Py_RETURN_NONE;
}

static PyObject* chill_skew(PyObject* self, PyObject* args) {
  std::set<int> stmt_nums;
  std::vector<int> skew_amounts;
  int level = intArg(args, 1);
  to_int_set(args, 0, stmt_nums);
  to_int_vector(args, 2, skew_amounts);
  myloop->skew(stmt_nums, level, skew_amounts);
  Py_RETURN_NONE;
}

static PyObject* chill_scale(PyObject* self, PyObject* args) {
  strict_arg_num(args, 3);
  std::set<int> stmt_nums;
  int level = intArg(args, 1);
  int scale_amount = intArg(args, 2);
  to_int_set(args, 0, stmt_nums);
  myloop->scale(stmt_nums, level, scale_amount);
  Py_RETURN_NONE;
}

static PyObject* chill_reverse(PyObject* self, PyObject* args) {
  strict_arg_num(args, 2);
  std::set<int> stmt_nums;
  int level = intArg(args, 1);
  to_int_set(args, 0, stmt_nums);
  myloop->reverse(stmt_nums, level);
  Py_RETURN_NONE;
}

static PyObject* chill_shift(PyObject* self, PyObject* args) {
  strict_arg_num(args, 3);
  std::set<int> stmt_nums;
  int level = intArg(args, 1);
  int shift_amount = intArg(args, 2);
  to_int_set(args, 0, stmt_nums);
  myloop->shift(stmt_nums, level, shift_amount);
  Py_RETURN_NONE;
}

static PyObject* chill_shift_to(PyObject* self, PyObject* args) {
  strict_arg_num(args, 3);
  int stmt_num = intArg(args, 0);
  int level = intArg(args, 1);
  int absolute_pos = intArg(args, 2);
  myloop->shift_to(stmt_num, level, absolute_pos);
  Py_RETURN_NONE;
}

static PyObject* chill_peel(PyObject* self, PyObject* args) {
    strict_arg_range(args, 2, 3);
    int stmt_num    = intArg(args, 0);
    int level       = intArg(args, 1);
    int amount      = intArg(args, 2);
    
    myloop->peel(stmt_num, level, amount);
    Py_RETURN_NONE;
}

static PyObject* chill_fuse(PyObject* self, PyObject* args) {
    strict_arg_num(args, 2);
    std::set<int> stmt_nums;
    int level = intArg(args, 1);
    to_int_set(args, 0, stmt_nums);
    myloop->fuse(stmt_nums, level);
    Py_RETURN_NONE;
}

static PyObject* chill_distribute(PyObject* self, PyObject* args) {
    strict_arg_num(args, 2);
    std::set<int> stmts;
    int level = intArg(args, 1);
    to_int_set(args, 0, stmts);
    myloop->distribute(stmts, level);
    Py_RETURN_NONE;
}


//#if 0
static PyObject* chill_flatten(PyObject* self, PyObject* args) {
    strict_arg_num(args, 4);
    int                 stmt_num        = intArg(args, 0);
    std::string         idxs            = strArg(args, 1);
    std::vector<int>    loop_levels;
    std::string         inspector_name  = strArg(args, 3);
    
    to_int_vector(args, 2, loop_levels);
    
    myloop->flatten(stmt_num, idxs, loop_levels, inspector_name);
    Py_RETURN_NONE;
}
//#endif


static PyObject* chill_compact(PyObject* self, PyObject* args) {
    strict_arg_num(args, 5);
    int                 stmt_num        = intArg(args, 0);
    int                 loop_level      = intArg(args, 1);
    std::string         new_array       = strArg(args, 2);
    int                 zero            = intArg(args, 3);
    std::string         data_array      = strArg(args, 4);
    
    myloop->compact(stmt_num, loop_level, new_array, zero, data_array);
    Py_RETURN_NONE;
}

static PyObject* chill_make_dense(PyObject* self, PyObject* args) {
    strict_arg_num(args, 3);
    int                 stmt_num        = intArg(args, 0);
    int                 loop_level      = intArg(args, 1);
    std::string         new_loop_index  = strArg(args, 2);

    myloop->make_dense(stmt_num, loop_level, new_loop_index);
    Py_RETURN_NONE;
}

static PyObject* chill_stencil_temp(PyObject* self, PyObject* args) {
    strict_arg_num(args, 1);
    int                 stmt        = intArg(args, 0);

    myloop->stencilASEPadded(stmt);
    Py_RETURN_NONE;
}

static PyObject* chill_find_stencil_shape(PyObject* self, PyObject* args) {
    strict_arg_num(args, 1);
    int                 stmt        = intArg(args, 0);

    myloop->find_stencil_shape(stmt);
    Py_RETURN_NONE;
}


static PyObject *
chill_num_statements(PyObject *self, PyObject *args)  
{
  //DEBUG_PRINT("\nC chill_num_statements() called from python\n"); 
  int num = myloop->stmt.size();
  //DEBUG_PRINT("C num_statement() = %d\n", num); 
  return Py_BuildValue( "i", num ); // BEWARE "d" is DOUBLE, not int
}

static PyObject *
chill_print_dep_ufs(PyObject *self, PyObject *args) {
  strict_arg_num(args, 3);
  std::string output_filename = strArg(args, 0);
  std::string privatizable_arrays = strArg(args, 1); // listed in a string separated with comma
  std::string reduction_operations = strArg(args, 2);// listed in a string separated with comma
  std::vector<std::pair<std::string, std::string > > dep_rels = 
     myloop->depRelsForParallelization(privatizable_arrays, reduction_operations);

  //
  std::ofstream outf;
  outf.open (output_filename.c_str(), std::ofstream::out);
  for(int i = 0 ; i < dep_rels.size() ; i++){
    outf<<dep_rels[i].first<<std::endl<<dep_rels[i].second<<std::endl;
  }
  outf.close();


  Py_RETURN_NONE;
}
#endif

#ifdef CUDACHILL
static PyMethodDef ChillMethods[] = { 
  
  // python name            C routine              parameter passing         comment
  {"destination",         chill_destination,         METH_VARARGS,    "set output filename"},
  {"print_code",          chill_print_code,          METH_VARARGS,    "print the code at this point"},
  {"print_ri",            chill_print_ri  ,          METH_VARARGS,    "print Runtime Info          "},
  {"print_idx",           chill_print_idx ,          METH_VARARGS,    "print indices               "},
  {"print_dep",           chill_print_dep ,          METH_VARARGS,    "print dep, dependecies?"},
  {"print_space",         chill_print_space,         METH_VARARGS,    "print something or other "},
  {"add_sync",            chill_add_sync,            METH_VARARGS,    "add sync, whatever that is"},
  {"rename_index",        chill_rename_index,        METH_VARARGS,    "rename a loop index"},
  {"permute",             chill_permute_v2,          METH_VARARGS,    "change the order of loops?"},
  {"tile3",               chill_tile_v2_3arg,        METH_VARARGS,    "something to do with tile"},
  {"tile7",               chill_tile_v2_7arg,        METH_VARARGS,    "something to do with tile"},
  {"thread_dims",         thread_dims,               METH_VARARGS,    "tx, ty, tz"},
  {"block_dims",          block_dims,                METH_VARARGS,    "bx, by"},
  {"thread_indices",      chill_thread_indices,      METH_VARARGS,    "tx, ty, tz"},
  {"block_indices",       chill_block_indices,       METH_VARARGS,    "bx, by"},
  {"hard_loop_bounds",    chill_hard_loop_bounds,    METH_VARARGS,    "lower, upper"},
  {"unroll",              chill_unroll,              METH_VARARGS,    "unroll a loop"},
  {"coalesce",            chill_flatten,             METH_VARARGS,    "Convert a multidimentianal iteration space into a single dimensional one"},
  {"make_dense",          chill_make_dense,          METH_VARARGS,    "Convert a non-affine iteration space into an affine one to enable loop transformations"},
  {"compact",             chill_compact,             METH_VARARGS,    "Call after make_dense to convert an affine iteration space back into a non-affine one"},
  {"cudaize",             chill_cudaize_v3,          METH_VARARGS,    "dunno"},
  {"datacopy_privatized", chill_datacopy_privatized, METH_VARARGS,    "dunno"},
  
  {"datacopy_9arg",       chill_datacopy9,           METH_VARARGS,    "datacopy with 9 arguments"},
  {"copy_to_texture",     chill_copy_to_texture,     METH_VARARGS,    "copy to texture mem"},
  {"read_IR",             chill_init,                METH_VARARGS,    "read an Intermediate Representation file"}, 
  {"cur_indices",         chill_cur_indices,         METH_VARARGS,    "currently active indices"},
  {"num_statements",      chill_num_statements,      METH_VARARGS,    "number of statements in ... something"},

  {"copy_to_shared",      chill_copy_to_shared,      METH_VARARGS,    ""},
  {"copy_to_registers",   chill_copy_to_registers,   METH_VARARGS,    ""},
  {"pragma",              chill_pragma,              METH_VARARGS,    "insert a pragma"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
  
  //{"copy_to_constant",    chill_copy_to_constant,    METH_VARARGS,  "copy to constant mem"},
  
};
#else
static PyMethodDef ChillMethods[] = { 
  
  //python name           C routine                  parameter passing comment
  {"source",              chill_source,                    METH_VARARGS,     "set source file for chill script"},
  {"procedure",           chill_procedure,                 METH_VARARGS,     "set the name of the procedure"},
  {"destination",         chill_destination,               METH_VARARGS,     "set the destination file"},
  {"effort",              chill_effort,                    METH_VARARGS,     "set the effort to remove loop overhead"},
  {"loop",                chill_loop,                      METH_VARARGS,     "indicate which loop to optimize"},
  {"print_code",          chill_print_code,                METH_VARARGS,     "print generated code"},
  {"print_dep",           chill_print_dep,                 METH_VARARGS,     "print the dependencies graph"},
  {"print_space",         chill_print_space,               METH_VARARGS,     "print space"},
  {"known",               chill_known,                     METH_VARARGS,     "knwon"},
  {"remove_dep",          chill_remove_dep,                METH_VARARGS,     "remove dependency i suppose"},
  {"original",            chill_original,                  METH_VARARGS,     "original"},
  {"permute",             chill_permute,                   METH_VARARGS,     "permute"},
  {"pragma",              chill_pragma,                    METH_VARARGS,     "pragma"},
  {"prefetch",            chill_prefetch,                  METH_VARARGS,     "prefetch"},
  {"tile",                chill_tile,                      METH_VARARGS,     "tile"},
  {"datacopy",            chill_datacopy,                  METH_VARARGS,     "datacopy"},
  {"datacopy_privitized", chill_datacopy_privatized,       METH_VARARGS,     "datacopy_privatized"},
  {"unroll",              chill_unroll,                    METH_VARARGS,     "unroll"},
  {"unroll_extra",        chill_unroll_extra,              METH_VARARGS,     "unroll_extra"},
  {"split",               chill_split,                     METH_VARARGS,     "split"},
  {"nonsingular",         chill_nonsingular,               METH_VARARGS,     "nonsingular"},
  {"skew",                chill_skew,                      METH_VARARGS,     "skew"},
  {"scale",               chill_scale,                     METH_VARARGS,     "scale"},
  {"reverse",             chill_reverse,                   METH_VARARGS,     "reverse"},
  {"shift",               chill_shift,                     METH_VARARGS,     "shift"},
  {"shift_to",            chill_shift_to,                  METH_VARARGS,     "shift_to"},
  {"peel",                chill_peel,                      METH_VARARGS,     "peel"},
  {"fuse",                chill_fuse,                      METH_VARARGS,     "fuse"},
  {"distribute",          chill_distribute,                METH_VARARGS,     "distribute"},
  {"coalesce",            chill_flatten,                   METH_VARARGS,     "Convert a multidimentianal iteration space into a single dimensional one"},
  {"make_dense",          chill_make_dense,                METH_VARARGS,     "Convert a non-affine iteration space into an affine one to enable loop transformations"},
  {"compact",             chill_compact,                   METH_VARARGS,     "Call after make_dense to convert an affine iteration space back into a non-affine one"},
  {"stencil_temp",        chill_stencil_temp,              METH_VARARGS,     "???"},
  {"find_stencil_shape",  chill_find_stencil_shape,        METH_VARARGS,     "???"},
  {"num_statements",      chill_num_statements,            METH_VARARGS,     "number of statements in the current loop"},
  {"print_dep_ufs",       chill_print_dep_ufs,             METH_VARARGS,     "..."},
  {"pragma",              chill_pragma,                    METH_VARARGS,     "insert a pragma"},
  {"omp_for",             chill_omp_for,                   METH_VARARGS,     "insert a an omp for pragma"},
  {NULL, NULL, 0, NULL}
};
#endif

static void register_globals(PyObject* m) {
  // Preset globals
  PyModule_AddStringConstant(m, "VERSION", CHILL_BUILD_VERSION);
  PyModule_AddStringConstant(m, "dest", "C");
  PyModule_AddStringConstant(m, "C", "C");
  // Tile method
  PyModule_AddIntConstant(m, "strided", 0);
  PyModule_AddIntConstant(m, "counted", 1);
  // Memory mode
  PyModule_AddIntConstant(m, "global", 0);
  PyModule_AddIntConstant(m, "shared", 1);
  PyModule_AddIntConstant(m, "textured", 2);
  // Bool flags
  PyModule_AddIntConstant(m, "sync", 1);
} 

PyMODINIT_FUNC
initchill(void)    // pass C methods to python 
{
  debug_fprintf(stderr, "in C, initchill() to set up C methods to be called from python\n");
  PyObject* m = Py_InitModule("chill", ChillMethods);
  register_globals(m);

  //Operate on a single global IR_Code and Loop instance
  ir_code = NULL;
  myloop = NULL;
  omega::initializeOmega();
  debug_fprintf(stderr, "Omega initialized\n");

  effort = 3; // Set the initial value
  #ifdef FRONTEND_ROSE
    parser = new chill::parser::Rose();
  #else
    parser = new chill::parser::Clang();
  #endif
}
