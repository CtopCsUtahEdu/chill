/*****************************************************************************
 Copyright (C) 2010 University of Utah.
 All Rights Reserved.

 Purpose:
 Register variables and functions into the global Lua addres space to
 provide an environment for CHiLL scripts

 Notes:
 Contains Lua wrappers for the CHiLL Loop class and methods.

 History:
 01/2010 created by Gabe Rudy
 03/2014 added support for CHiLL without Cuda (Derick Huth)
*****************************************************************************/

#define lua_c
#include <lua.hpp> //All lua includes wrapped in extern "C"
#include "loop.hh"
#include "chill_env.hh"

#ifdef CUDACHILL
#include <mcheck.h>

// this defines LoopCuda, with chill internals no matter the front end 
#include "loop_cuda_chill.hh"

#ifdef FRONTEND_CLANG 
//#include "loop_cuda_clang.hh"
#endif 

// I think the LoopCuda definitions in loop_cuda_(frontend).hh collide 
#ifdef FRONTEND_ROSE
//#include "loop_cuda_rose.hh"
//#include "ir_rose.hh"
#include "ir_cudarose.hh"
#endif

#else

// CHILL, not CUDADACHILL
#include "chill_run_util.hh"
#include <omega.h>
#include "ir_code.hh"

#ifdef FRONTEND_ROSE
#include "ir_rose.hh"
#endif

#ifdef FRONTEND_CLANG
#include "ir_clang.hh"
#endif

#endif  // CHILL, NOT CUDACHILL 


using namespace omega;

#ifdef CUDACHILL
extern LoopCuda *myloop;
#else
extern Loop *myloop;
#endif
extern IR_Code *ir_code;
extern bool is_interactive;
extern bool repl_stop;

std::string procedure_name;
std::string source_filename;

extern std::vector<IR_Control *> ir_controls;
extern std::vector<int> loops;

//Macros for wrapping code to myloop-> that translates C++ exceptions to
//Lua stacktraced errors
#define REQUIRE_LOOP try{ if (myloop == NULL){ throw std::runtime_error("loop not initialized"); }
#define END_REQUIRE_LOOP  }catch (const std::exception &e) { return luaL_error(L, e.what()); }

#ifdef CUDACHILL
void register_v1(lua_State *L);
void register_v2(lua_State *L);

#endif
//Extra param checking
static bool luaL_checkboolean(lua_State *L, int narg) {
  if (!lua_isboolean(L,narg))
    luaL_typerror(L, narg, "boolean");
  return lua_toboolean(L, narg);
}

static bool luaL_optboolean(lua_State *L, int narg, bool def) {
  return luaL_opt(L, luaL_checkboolean, narg, def);
}

static bool tointvector(lua_State *L, int narg, std::vector<int>& v) {
  if (!lua_istable(L, narg))
    return false;
  
  //Iterate through array (table)
  lua_pushnil(L);  // first key
  while (lua_next(L, narg) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    v.push_back((int) luaL_checknumber(L, -1));
    //printf("added: %d", v[v.size()-1]);
    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);
  }
  return true;
}

static bool tointset(lua_State* L, int narg, std::set<int>& s) {
  if(!lua_istable(L, narg))
    return false;
  // iterate through array (lua table)
  lua_pushnil(L); // first key
  while (lua_next(L, narg) != 0) {
    int val = (int)luaL_checknumber(L, -1);
    //printf("added: %d\n", val);
    s.insert(val);
    lua_pop(L, 1);
  }
}

static bool tostringvector(lua_State *L, int narg,
                           std::vector<std::string>& v) {
  if (!lua_istable(L, narg))
    return false;
  
  //Iterate through array (table)
  lua_pushnil(L);  // first key
  while (lua_next(L, narg) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    v.push_back(luaL_checkstring(L,-1));
    //printf("added: %d", v[v.size()-1]);
    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);
  }
  return true;
}

static bool tostringmap(lua_State *L, int narg,
                        std::map<std::string, std::string>& v) {
  if (!lua_istable(L, narg))
    return false;
  
  //Iterate through array (table)
  lua_pushnil(L);  // first key
  while (lua_next(L, narg) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    v.insert(
             std::make_pair(luaL_checkstring(L,-2), luaL_checkstring(L,-1)));
    //printf("added: %d", v[v.size()-1]);
    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);
  }
  return true;
}

static bool tostringintmap(lua_State *L, int narg,
                           std::map<std::string, int>& v) {
  if (!lua_istable(L, narg))
    return false;
  
  //Iterate through array (table)
  lua_pushnil(L);  // first key
  while (lua_next(L, narg) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    v.insert(
             std::make_pair(luaL_checkstring(L,-2),
                            (int) luaL_checknumber(L, -1)));
    //printf("added: %d", v[v.size()-1]);
    // removes 'value'; keeps 'key' for next iteration
    lua_pop(L, 1);
  }
  return true;
}

static bool tostringintmapvector(lua_State *L, int narg, std::vector<std::map<std::string, int> >& v) {
  if(!lua_istable(L, narg))
    return false;
  lua_pushnil(L);
  // Iterate over table
  while(lua_next(L, narg) != 0) {
    std::map<std::string, int> map;
    // use 'value' (at index -1), discard key
    // try to parse table as a 'string to int' map.
    if(!tostringintmap(L, -1, map))
      return false;
    v.push_back(map);
    lua_pop(L, 1);
  }
  return true;
}

static bool tointmatrix(lua_State *L, int narg,
                        std::vector<std::vector<int> >& m) {
  if (!lua_istable(L, narg))
    return false;
  
  //Iterate through array (table)
  lua_pushnil(L);  // first key
  while (lua_next(L, narg) != 0) {
    // uses 'key' (at index -2) and 'value' (at index -1)
    if (!lua_istable(L,-1)) {
      lua_pop(L, 2);
      return false;
    }
    m.push_back(std::vector<int>());
    int i = m.size() - 1;
    //Now iterate over the keys of the second level table
    int l2 = lua_gettop(L); //Index of second level table
    lua_pushnil(L);  // first key
    while (lua_next(L, l2) != 0) {
      int k = (int) luaL_checknumber(L, -1);
      m[i].push_back(k);
      //printf("m[%d][%d] = %d\n", i,m[i].size()-1,k);
      lua_pop(L, 1);
    }
    lua_pop(L, 1);
    // removes 'value'; keeps 'key' for next iteration
  }
  return true;
}

static void strict_arg_num(lua_State *L, int num) {
  int n = lua_gettop(L); //Number of arguments
  if (n != num)
    throw std::runtime_error("incorrect number of arguments");
}

// -------------------------------------------------------------------
// Initialization and finalization functions
// -------------------------------------------------------------------
#ifdef CUDACHILL
/* The function we'll call from the lua script */
static int init(lua_State *L) {
  //mtrace(); fprintf(stderr, "malloc debugging ENABLED\n");  // turn on malloc debugging
  
  int n = lua_gettop(L); //Number of arguments
  if (n > 0) {
    //Expect one of the following forms
    //l1 = init("mm4.sp2",0,0) --input file, procedure 0, loop 0
    //or
    //l1 = init("mm4.sp2","NameFromPragma")
    
    const char* source_filename = luaL_optstring(L,1,0);
#ifdef  FRONTEND_ROSE
    if(lua_isstring(L,2)) {
      const char* procedure_name = luaL_optstring(L, 2, 0);
#endif
      int loop_num = luaL_optint(L, 3, 0);
      
      lua_getglobal(L, "dest");
      const char* dest_lang = lua_tostring(L,-1);
      lua_pop(L, 1);
#ifdef FRONTEND_ROSE
      ir_code = new IR_cudaroseCode(source_filename, procedure_name);
#elif defined( FRONTEND_CLANG) 
      ir_code = new IR_cudaclangCode(source_filename, procedure_name);      
#endif
      IR_Block *block = ir_code->GetCode();
      fprintf(stderr, "chill_env init()  L200  getting the control structure for the entire function(?)\n"); 
      ir_controls = ir_code->FindOneLevelControlStructure(block);
      fprintf(stderr, "%d controls\n\n", ir_controls.size()); 
      
#ifdef FRONTEND_ROSE

      int loop_count = 0;
      for (int i = 0; i < ir_controls.size(); i++) {
        if (ir_controls[i]->type() == IR_CONTROL_LOOP) {
          loops.push_back(i);
          loop_count++;
        }
      }
      fprintf(stderr, "FRONTEND_ROSE loop_count %d\n", loop_count); 
      delete block;
      
      fprintf(stderr, "chillenv.cc L277 loop count %d\n", loop_count); 
      
      std::vector<IR_Control *> parm;
      for(int j = 0; j < loop_count; j++)
        parm.push_back(ir_controls[loops[j]]);
      fprintf(stderr, "parm %d\n", parm.size()); 
      
      block = ir_code->MergeNeighboringControlStructures(parm);
      IR_roseBlock *RB = (IR_roseBlock *)block;
      fprintf(stderr, "block is now from merged\n"); 
      
     
      fprintf(stderr, "block %d statements, AST %p\n\n", RB->statements.size(), RB->chillAST);

#elif FRONTEND_CLANG
      //fprintf(stderr, "UHOH chill_env.cc  ~line 233\n"); 
      
      // find the loops in the ir_controls
      // make an array of statement indeces for the loops
      
      int loop_count = 0;
      for (int i = 0; i < ir_controls.size(); i++) {
        if (ir_controls[i]->type() == IR_CONTROL_LOOP) {
          loops.push_back(i);
          loop_count++;
        }
      }
      fprintf(stderr, "FRONTEND_CLANG loop_count %d\n", loop_count); 
      delete block;
      
      fprintf(stderr, "loop count %d\n", loop_count); 
      
      std::vector<IR_Control *> parm;
      for(int j = 0; j < loop_count; j++)
        parm.push_back(ir_controls[loops[j]]);
      fprintf(stderr, "parm %d\n", parm.size()); 

      block = ir_code->MergeNeighboringControlStructures(parm);
      
#endif


      fprintf(stderr, "chillenv.cc  making myloop from merged control structure\n"); 
      myloop = new LoopCuda(block, loop_num);
      delete block;
      
      fprintf(stderr, "myloop made, block gone\n"); 
      fprintf(stderr, "%d vecs\n", myloop->idxNames.size()); 
      int numvecs =  myloop->idxNames.size();
      for (int j=0; j<numvecs; j++) { 
        int numnames =  myloop->idxNames[j].size();
        fprintf(stderr, "vec %d has %d strings\n", j, numnames);
        for (int i=0; i<numnames; i++) fprintf(stderr, "name %d = '%s'\n", i, (myloop->idxNames[j])[i].c_str()); 
      }
      
      //end-protonu
      
    } else {
      //TODO: handle pragma lookup
    }
    //Also register a different set of global functions
    myloop->original();
    myloop->useIdxNames = true;           //Use idxName in code_gen
    register_v2(L);
    //TODO: return a reference to the intial array if that makes sense
    //still
    fprintf(stderr, "chill_env.cc init()  returning EARLY   BUH\n"); 
    return 0;
  }
  
  lua_getglobal(L, "source");
  const char* source_filename = lua_tostring(L,-1);
  lua_pop(L, 1);
  
  lua_getglobal(L, "dest");
  const char* dest_lang = lua_tostring(L,-1);
  lua_pop(L, 1);
  
  lua_getglobal(L, "procedure");
#if defined( FRONTEND_ROSE ) || defined( FRONTEND_CLANG )
  const char* procedure_name = lua_tostring(L , -1);
  fprintf(stderr, "procedure name %s\n", procedure_name); 
#endif
  lua_pop(L, 1);
  
  lua_getglobal(L, "loop");
  int loop_num = lua_tointeger(L, -1);
  lua_pop(L, 1);
  
#ifdef FRONTEND_ROSE
  ir_code = new IR_cudaroseCode(source_filename, procedure_name);
#elif FRONTEND_CLANG
  ir_code = new IR_cudaclangCode(source_filename, procedure_name);
#endif
  
  IR_Block *block = ir_code->GetCode();
  ir_controls = ir_code->FindOneLevelControlStructure(block);
  
  fprintf(stderr, "YET ANOTHER IFDEF\n"); // ??????????? exit(-1);  // TODO 
  
#ifdef FRONTEND_ROSE
  
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
  
  block = ir_code->MergeNeighboringControlStructures(parm);
#endif
  myloop = new LoopCuda(block, loop_num);
  delete block;
  
  //register_v1(L);
  register_v2 (L);
  return 0;
}
#else
static void strict_arg_num(lua_State* L, int min, int max) {
  int n = lua_gettop(L);
  if(n < min || n > max)
    throw std::runtime_error("incorrect number of arguments");
}

int get_loop_num_start(lua_State *L) {
  lua_getglobal(L, "loop_start");
  int loop_num_start = lua_tointeger(L, -1);
  lua_pop(L, 1);
  return loop_num_start;
}
int get_loop_num_end(lua_State* L) {
  lua_getglobal(L, "loop_end");
  int loop_num_end = lua_tointeger(L, -1);
  lua_pop(L, 1);
  return loop_num_end;
}

static int set_loop_num_start(lua_State *L, int start_num) {
  lua_pushinteger(L, start_num);
  lua_setglobal(L, "loop_start");
}
static int set_loop_num_end(lua_State *L, int end_num) {
  lua_pushinteger(L, end_num);
  lua_setglobal(L, "loop_end");
}

static int source(lua_State* L) {
  if(!source_filename.empty()) {
    fprintf(stderr, "only one file can be handled in a script");
    if(!is_interactive)
      exit(2);
  }
  source_filename = luaL_checkstring(L, 1);
  return 0;
}


static int procedure(lua_State* L) {
  if(!procedure_name.empty()) {
    fprintf(stderr, "only one procedure can be handled in a script");
    if(!is_interactive)
      exit(2);
  }
  procedure_name = luaL_checkstring(L, 1);
  return 0;
}

void finalize_loop(int loop_num_start, int loop_num_end) {
  if (loop_num_start == loop_num_end) {
    ir_code->ReplaceCode(ir_controls[loops[loop_num_start]], myloop->getCode());
    ir_controls[loops[loop_num_start]] = NULL;
  }
  else {
    std::vector<IR_Control *> parm;
    for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++)
      parm.push_back(ir_controls[i]);
    IR_Block *block = ir_code->MergeNeighboringControlStructures(parm);
    ir_code->ReplaceCode(block, myloop->getCode());
    for (int i = loops[loop_num_start]; i <= loops[loop_num_end]; i++) {
      delete ir_controls[i];
      ir_controls[i] = NULL;
    }
  }
  delete myloop;
}
void finalize_loop(lua_State* L) {
  int loop_num_start = get_loop_num_start(L);
  int loop_num_end = get_loop_num_end(L);
  finalize_loop(loop_num_start, loop_num_end);
}
static void init_loop(lua_State* L, int loop_num_start, int loop_num_end) {
  if (source_filename.empty()) {
    fprintf(stderr, "source file not set when initializing the loop");
    if (!is_interactive)
      exit(2);
  }
  else {
    if (ir_code == NULL) {
#ifdef FRONTEND_ROSE  
      if (procedure_name.empty())
        procedure_name = "main";
#endif
      
#ifdef FRONTEND_ROSE
      ir_code = new IR_roseCode(source_filename.c_str(), procedure_name.c_str());
#elif  FRONTEND_CLANG
      
#else
      
#endif
      
      IR_Block *block = ir_code->GetCode();
      ir_controls = ir_code->FindOneLevelControlStructure(block);
      for (int i = 0; i < ir_controls.size(); i++) {
        if (ir_controls[i]->type() == IR_CONTROL_LOOP)
          loops.push_back(i);
      }
      delete block;
    }
    if (myloop != NULL && myloop->isInitialized()) {
      finalize_loop(L);
    }
  }
  set_loop_num_start(L, loop_num_start);
  set_loop_num_end(L, loop_num_end);
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

static int loop(lua_State* L) {
  // loop (n)
  // loop (n:m)
  int nargs = lua_gettop(L);
  int start_num;
  int end_num;
  if(nargs == 1) {
    start_num = luaL_optint(L, 1, 0);
    end_num = start_num;
  }
  else if(nargs == 2) {
    start_num = luaL_optint(L, 1, 0);
    end_num = luaL_optint(L, 2, 0);
  }
  else {
    fprintf(stderr, "loop takes one or two arguments");
    if(!is_interactive)
      exit(2);
  }
  init_loop(L, start_num, end_num);
  return 0;
}
#endif

#ifdef CUDACHILL

static int exit(lua_State *L) {
  strict_arg_num(L, 0);
  repl_stop = true;
  return 0;
}

static int print_code(lua_State *L) {
  REQUIRE_LOOP;
  //strict_arg_num(L, 0);
  int n = lua_gettop(L); //Number of arguments
  //fprintf(stderr, "chill_env.cc line 557, calling myloop->printCode()\n"); 
  if (n==0) myloop->printCode(0);
  else { 
    int stmt = luaL_optint(L,1,0);
    myloop->printCode(stmt);
  }
  printf("\n");
  END_REQUIRE_LOOP;
  return 0;
}

static int print_ri(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  myloop->printRuntimeInfo();
  printf("\n");
  END_REQUIRE_LOOP;
  return 0;
}

static int print_idx(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  myloop->printIndexes();
  printf("\n");
  END_REQUIRE_LOOP;
  return 0;
}

static int print_dep(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  std::cout << myloop->dep;
  END_REQUIRE_LOOP;
  return 0;
}

static int print_space(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  for (int i = 0; i < myloop->stmt.size(); i++) {
    printf("s%d: ", i + 1);
    Relation r;
    if (!myloop->stmt[i].xform.is_null())
      r = Composition(copy(myloop->stmt[i].xform), copy(myloop->stmt[i].IS));
    else
      r = copy(myloop->stmt[i].IS);
    r.simplify(2, 4);
    r.print();
  }END_REQUIRE_LOOP;
  return 0;
}

static int num_statement(lua_State *L) {
  REQUIRE_LOOP;
  lua_pushinteger(L, myloop->stmt.size());
  END_REQUIRE_LOOP;
  return 1;
}

static int does_var_exists(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 1);
  std::string symName = luaL_optstring(L,1,"");
  lua_pushboolean(L, myloop->symbolExists(symName));
  END_REQUIRE_LOOP;
  return 1;
}

static int add_sync(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  int stmt = luaL_optint(L,1,0);
  std::string idxName = luaL_optstring(L,2,"");
  myloop->addSync(stmt, idxName);
  END_REQUIRE_LOOP;
  return 0;
}

static int rename_index(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  int stmt = luaL_optint(L,1,0);
  std::string idxName = luaL_optstring(L,2,"");
  std::string newName = luaL_optstring(L,3,"");
  myloop->renameIndex(stmt, idxName, newName);
  END_REQUIRE_LOOP;
  return 0;
}

//basic on index names
static int permute_v2(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  int stmt = luaL_optint(L,1,0);
  std::vector<std::string> order;
  if (!tostringvector(L, 2, order)) {
    throw std::runtime_error("second arg must be a string vector");
  }
  myloop->permute_cuda(stmt, order);
  END_REQUIRE_LOOP;
  return 0;
}

static int tile_v2(lua_State *L) {
  fprintf(stderr, "chill_env.cc tile_v2()\n"); 
  
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  if (n != 3 && n != 7)
    throw std::runtime_error("incorrect number of arguments");
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  fprintf(stderr, "%d args, stmt num %d,  level %d\n", n, stmt_num, level); 
  
  if (n == 3) {
    int outer_level = luaL_optint(L, 3, 1);
    myloop->tile_cuda(stmt_num, level, outer_level);
  } else {
    fprintf(stderr, "hey 6 or 7 args!\n"); 
    int tile_size = luaL_optint(L, 3, 0);
    int outer_level = luaL_optint(L, 4, 1);
    std::string idxName = luaL_optstring(L,5,"");
    std::string ctrlName = luaL_optstring(L,6,"");
    TilingMethodType method = StridedTile;
    if (n > 6) {
      int imethod = luaL_optint(L, 7, 2);
      if (imethod == 0)
        method = StridedTile;
      else if (imethod == 1)
        method = CountedTile;
      else {
        throw std::runtime_error(
                                 "7th argument must be either strided or counted");
      }
    }
    fprintf(stderr, "chill_env.cc tile_v2() calling tile_cuda()\n"); 
    myloop->tile_cuda(stmt_num, level, tile_size, outer_level, idxName,
                      ctrlName, method);
    fprintf(stderr, "chill_env.cc tile_v2() DONE calling tile_cuda()\n"); 
  }END_REQUIRE_LOOP;
  fprintf(stderr, "chill_env.cc tile_v2() DONE\n"); 
  
  return 0;
}

static int cur_indices(lua_State *L) {
  fprintf(stderr, "\ncur_indices()\n"); 
  REQUIRE_LOOP;
  strict_arg_num(L, 1);
  int stmt_num = luaL_optint(L, 1, 0);
  fprintf(stderr, "stmt_num = %d\n", stmt_num); 
  
  if (myloop->idxNames[stmt_num].empty()) {
    fprintf(stderr, "idxNames[%d] is EMPTY\n"); 
    exit(-1); 
  }
  fprintf(stderr, "%d idxnames\n",  myloop->idxNames[stmt_num].size()); 
  for (int i = 0; i < myloop->idxNames[stmt_num].size(); i++) {
    fprintf(stderr, "name %d   '%s'\n", i, myloop->idxNames[stmt_num][i].c_str()); 
  }
  
  //TODO: needs to be per stmt
  lua_createtable(L, myloop->idxNames[stmt_num].size(), 0);
  for (int i = 0; i < myloop->idxNames[stmt_num].size(); i++) {
    lua_pushinteger(L, i + 1);
    lua_pushstring(L, myloop->idxNames[stmt_num][i].c_str());
    lua_settable(L, -3);
  }END_REQUIRE_LOOP;
  return 1;
}

static int block_indices(lua_State *L) {
  fprintf(stderr, "\n\n\n*** chill_env.cc block_indices probably an error with cu_bx vs Vcu_bx\n");
  
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  lua_newtable(L);
  if (myloop->cu_bx > 1) {
    lua_pushinteger(L, 1);
    lua_pushstring(L, "bx");
    lua_settable(L, -3);
  }
  if (myloop->cu_by > 1) {
    lua_pushinteger(L, 2);
    lua_pushstring(L, "by");
    lua_settable(L, -3);
  }END_REQUIRE_LOOP;
  return 1;
}

static int thread_indices(lua_State *L) {
  REQUIRE_LOOP;
  fprintf(stderr, "\n\n\n*** chill_env.cc thread_indices probably an error with cu_bx vs Vcu_bx\n");
  strict_arg_num(L, 0);
  lua_newtable(L);
  if (myloop->cu_tx > 1) {
    lua_pushinteger(L, 1);
    lua_pushstring(L, "tx");
    lua_settable(L, -3);
  }
  if (myloop->cu_ty > 1) {
    lua_pushinteger(L, 2);
    lua_pushstring(L, "ty");
    lua_settable(L, -3);
  }
  if (myloop->cu_tz > 1) {
    lua_pushinteger(L, 3);
    lua_pushstring(L, "tz");
    lua_settable(L, -3);
  }END_REQUIRE_LOOP;
  return 1;
}

static int block_dims(lua_State *L) {
  fprintf(stderr, "\n\n\n*** chill_env.cc block_dims probably an error with cu_bx vs Vcu_bx\n");    
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  lua_pushinteger(L, myloop->cu_bx);
  lua_pushinteger(L, myloop->cu_by);
  END_REQUIRE_LOOP;
  return 2;
}

static int thread_dims(lua_State *L) {
  fprintf(stderr, "\n\n\n*** chill_env.cc thread_dims probably an error with cu_bx vs Vcu_bx\n");    
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  lua_pushinteger(L, myloop->cu_tx);
  lua_pushinteger(L, myloop->cu_ty);
  lua_pushinteger(L, myloop->cu_tz);
  END_REQUIRE_LOOP;
  return 3;
}

static int hard_loop_bounds(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  int stmt = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int upper, lower;
  myloop->extractCudaUB(stmt, level, upper, lower);
  lua_pushinteger(L, lower);
  lua_pushinteger(L, upper);
  END_REQUIRE_LOOP;
  return 2;
}



static int datacopy_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  
  //overload 1
  //examples:
  // datacopy(0,4,a,false,0,1,-16)
  // datacopy(0,3,2,1)
  if (n < 4 || n > 9)
    throw std::runtime_error("incorrect number of arguments");
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  const char* array_name = luaL_optstring(L, 3, 0);
  std::vector<std::string> new_idxs;
  if (!tostringvector(L, 4, new_idxs))
    throw std::runtime_error("fourth argument must be an array of strings");
  bool allow_extra_read = luaL_optboolean(L, 5, false);
  int fastest_changing_dimension = luaL_optint(L, 6, -1);
  int padding_stride = luaL_optint(L, 7, 1);
  int padding_alignment = luaL_optint(L, 8, 1);
  bool cuda_shared = luaL_optboolean(L, 9, false);
  myloop->datacopy_cuda(stmt_num, level, array_name, new_idxs, 
                        allow_extra_read, fastest_changing_dimension, 
                        padding_stride, 
                        padding_alignment,
                        cuda_shared);
  END_REQUIRE_LOOP;
  return 0;
}



static int ELLify_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  
  //overload 1
  //examples:
  // datacopy(0,4,a,false,0,1,-16)
  // datacopy(0,3,2,1)
  if (n != 3 && n != 5)
    throw std::runtime_error("incorrect number of arguments");
  int stmt_num = luaL_optint(L, 1, 0);
  std::vector<std::string> new_idxs;
  if (!tostringvector(L, 2, new_idxs))
    throw std::runtime_error("2nd argument must be an array of strings");
  int padding_stride = luaL_optint(L, 3, 0);
  bool dense_pad = luaL_optboolean(L, 4, false);
  std::string pos_array_name = luaL_optstring(L, 5, "");
  myloop->ELLify_cuda(stmt_num, new_idxs, padding_stride, dense_pad, pos_array_name);
  END_REQUIRE_LOOP;
  return 0;
}




static int datacopy_privatized_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  
  //example:
  //datacopy_privatized(0,3,"a",{4,5},false,-1,1,1)
  if (n < 4 || n > 9)
    throw std::runtime_error("incorrect number of arguments");
  int stmt_num = luaL_optint(L, 1, 0);
  std::string level_idx = luaL_optstring(L,2,"");
  int level = myloop->findCurLevel(stmt_num, level_idx);
  const char* array_name = luaL_optstring(L, 3, 0);
  
  std::vector<std::string> privatized_idxs;
  if (!tostringvector(L, 4, privatized_idxs))
    throw std::runtime_error("4th argument must be an array of index strings");
  std::vector<int> privatized_levels(privatized_idxs.size());
  for (int i = 0; i < privatized_idxs.size(); i++)
    privatized_levels[i] = myloop->findCurLevel(stmt_num, privatized_idxs[i]);
  
  bool allow_extra_read = luaL_optboolean(L, 5, false);
  int fastest_changing_dimension = luaL_optint(L, 6, -1);
  int padding_stride = luaL_optint(L, 7, 1);
  int padding_alignment = luaL_optint(L, 8, 1);
  bool cuda_shared = luaL_optboolean(L, 9, false);
  myloop->datacopy_privatized_cuda(stmt_num, level, array_name, privatized_levels,
                                   allow_extra_read, fastest_changing_dimension, padding_stride,
                                   padding_alignment, cuda_shared);
  END_REQUIRE_LOOP;
  return 0;
}

static int unroll_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level;
  if (lua_isnumber(L, 2)) {
    level = luaL_optint(L, 2, 0);
  } else {
    std::string level_idx = luaL_optstring(L,2,"");
    level = myloop->findCurLevel(stmt_num, level_idx);
  }
  int unroll_amount = luaL_optint(L, 3, 0);
  bool does_expand = myloop->unroll_cuda(stmt_num, level, unroll_amount);
  lua_pushboolean(L, does_expand);
  END_REQUIRE_LOOP;
  return 1;
}

#if 0
static int flatten_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 4);
  int stmt_num = luaL_optint(L, 1, 0);
  //std::vector<std::string> idxs;
  //if (!tostringvector(L, 2, idxs)) {
  //  throw std::runtime_error("second arg must be a string vector");
  //}
  
  std::string  idxs = luaL_optstring(L, 2, "");
  
  std::vector<int> loop_levels;
  
  if (!tointvector(L, 3, loop_levels)) {
    throw std::runtime_error("third arg must be a int vector");
  }
  
  std::string inspector_name = luaL_optstring(L,4,"");
  
  myloop->flatten_cuda(stmt_num, idxs, loop_levels, inspector_name);
  
  END_REQUIRE_LOOP;
  return 1;
}
#endif

static int distribute_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 2);
  
  std::vector<int> stmt_nums;
  if (!tointvector(L, 1, stmt_nums)) {
    throw std::runtime_error("1st arg must be an integer vector");
  }
  
  int loop_level = luaL_optint(L, 2, 0);
  
  myloop->distribute_cuda(stmt_nums, loop_level);
  
  END_REQUIRE_LOOP;
  return 1;
}

static int fuse_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 2);
  
  std::vector<int> stmt_nums;
  if (!tointvector(L, 1, stmt_nums)) {
    throw std::runtime_error("1st arg must be an integer vector");
  }
  
  int loop_level = luaL_optint(L, 2, 0);
  
  myloop->fuse_cuda(stmt_nums, loop_level);
  
  END_REQUIRE_LOOP;
  return 1;
}

static int peel_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 3);
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  int loop_level = luaL_optint(L, 2, 0);
  
  int amount = luaL_optint(L, 3, 0);
  
  myloop->peel_cuda(stmt_num, loop_level, amount);
  
  END_REQUIRE_LOOP;
  return 1;
}

static int scalar_expand_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  //strict_arg_num(L, 3);
  if (n != 3 && n != 4 && n != 5 && n != 6)
    throw std::runtime_error("wrong number of arguments");
  int stmt_num = luaL_optint(L, 1, 0);
  
  std::vector<int> level;
  if (!tointvector(L, 2, level)) {
    throw std::runtime_error("1st arg must be an integer vector");
  }
  std::string arrName = luaL_optstring(L,3,"");
  
  int cuda_shared = luaL_optint(L, 4, 0);
  int padding = luaL_optint(L, 5, 0);
  if (n != 6)
    myloop->scalar_expand_cuda(stmt_num, level, arrName, cuda_shared,
                               padding);
  else {
    int assign_then_accumulate = luaL_optint(L, 6, 0);
    myloop->scalar_expand_cuda(stmt_num, level, arrName, cuda_shared,
                               padding, assign_then_accumulate);
  }
  END_REQUIRE_LOOP;
  return 1;
}

static int shift_to_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 3);
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  int level = luaL_optint(L, 2, 0);
  
  int absolute_position = luaL_optint(L, 3, 0);
  
  myloop->shift_to_cuda(stmt_num, level, absolute_position);
  
  END_REQUIRE_LOOP;
  return 1;
}
static int split_with_alignment_v2(lua_State *L) {
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 3);
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  int level = luaL_optint(L, 2, 0);
  
  int alignment = luaL_optint(L, 3, 0);
  
  myloop->split_with_alignment_cuda(stmt_num, level, alignment);
  
  END_REQUIRE_LOOP;
  return 1;
}
static int compact_v2(lua_State *L){
  REQUIRE_LOOP;
  strict_arg_num(L, 5);
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  int level = luaL_optint(L, 2, 0);
  
  std::string new_array = luaL_optstring(L,3,"");
  
  int zero = luaL_optint(L, 4, 0);
  
  std::string data_array = luaL_optstring(L,5,"");
  myloop->compact_cuda(stmt_num,  level,  new_array,  zero,data_array);
  END_REQUIRE_LOOP;
  return 1;
}
static int  make_dense_v2(lua_State *L){
  
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  int loop_level = luaL_optint(L, 2, 0);
  
  std::string new_loop_index = luaL_optstring(L,3,"");
  
  
  myloop->make_dense_cuda(stmt_num,  loop_level,  new_loop_index);
  END_REQUIRE_LOOP;
  return 1;
  
}

static int  addKnown_v2(lua_State *L){
  
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  
  std::string var = luaL_optstring(L,1,"");
  int val = luaL_optint(L, 2, 0);
  
  
  
  
  
  myloop->addKnown_cuda(var,val);
  END_REQUIRE_LOOP;
  return 1;
  
}


static int  skew_v2(lua_State *L){
  
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  
  std::vector<int> stmt_num;
  if (!tointvector(L, 1, stmt_num)) {
    throw std::runtime_error("first arg must be a int vector");
  }
  int level = luaL_optint(L, 2, 0);
  
  
  std::vector<int> coefs;
  if (!tointvector(L, 3, coefs)) {
    throw std::runtime_error("third arg must be a int vector");
  }
  
  
  myloop->skew_cuda(stmt_num,level, coefs);
  END_REQUIRE_LOOP;
  return 1;
  
}

static int reduce_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  //strict_arg_num(L, 5);
  
  if(n != 5 && n != 6)
    throw std::runtime_error("wrong number of arguments");
  
  int stmt_num = luaL_optint(L, 1, 0);
  
  std::vector<int> level;
  if (!tointvector(L, 2, level)) {
    throw std::runtime_error("second arg must be a int vector");
  }
  
  int param = luaL_optint(L, 3, 0);
  
  std::string func_name = luaL_optstring(L,4,"");
  
  std::vector<int> seq_level;
  if (!tointvector(L, 5, seq_level)) {
    throw std::runtime_error("second arg must be a int vector");
  }
  
  if(n <= 5)
    myloop->reduce_cuda(stmt_num, level, param, func_name, seq_level);
  else{
    int bound = luaL_optint(L, 6, 0);
    myloop->reduce_cuda(stmt_num, level, param, func_name, seq_level,bound);
  }
  
  END_REQUIRE_LOOP;
  return 1;
}





static int cudaize_v2(lua_State *L) {
  REQUIRE_LOOP;
  int n = lua_gettop(L); //Number of arguments
  fprintf(stderr, "\n\ncudaize_v2(), %d arguments\n", n); 
  strict_arg_num(L, 3);
  
  std::string kernel_name = luaL_optstring(L, 1, 0);
  
  std::vector<std::string> blockIdxs;
  std::vector<std::string> threadIdxs;
  std::map<std::string, int> array_sizes;
  if (!tostringintmap(L, 2, array_sizes))
    throw std::runtime_error("second argument must be a map[string->int]");
  
  if (lua_istable(L, 3)) {
    //Iterate through array (table)
    lua_pushnil(L);   // first key
    while (lua_next(L, 3) != 0) {
      // uses 'key' (at index -2) and 'value' (at index -1)
      if (strcmp(luaL_checkstring(L,-2), "block") == 0) {
        if (!tostringvector(L, lua_gettop(L), blockIdxs))
          throw std::runtime_error(
                                   "third argument must have a string list for its 'block' key");
      } else if (strcmp(luaL_checkstring(L,-2), "thread") == 0) {
        if (!tostringvector(L, lua_gettop(L), threadIdxs))
          throw std::runtime_error(
                                   "third argument must have a string list for its 'thread' key");
      } else {
        goto v2NotTable;
      }
      lua_pop(L, 1);
    }
  } else {
  v2NotTable: throw std::runtime_error(
                                       "third argument must be a table with 'block' and 'thread' as potential keys and list of indexes as values");
  }
  myloop->cudaize_v2(kernel_name, array_sizes, blockIdxs, threadIdxs);
  END_REQUIRE_LOOP;
  return 0;
}



static int cudaize_v3(lua_State *L) {   // requires 5 args, starting with stmt num
  int n = lua_gettop(L); //Number of arguments
  fprintf(stderr, "\n\ncudaize_v3(), %d arguments\n", n); 
  REQUIRE_LOOP;
  //int n = lua_gettop(L); //Number of arguments
  strict_arg_num(L, 5);
  int stmt_num = luaL_optint(L, 1, 0);
  std::string kernel_name = luaL_optstring(L, 2, 0);
  
  std::vector<std::string> blockIdxs;
  std::vector<std::string> threadIdxs;
  std::vector<std::string> kernel_params;
  std::map<std::string, int> array_sizes;
  if (!tostringintmap(L, 3, array_sizes))
    throw std::runtime_error("second argument must be an map[string->int]");
  
  if (lua_istable(L, 4)) {
    //Iterate through array (table)
    lua_pushnil(L); // first key
    while (lua_next(L, 4) != 0) {
      // uses 'key' (at index -2) and 'value' (at index -1)
      if (strcmp(luaL_checkstring(L,-2), "block") == 0) {
        if (!tostringvector(L, lua_gettop(L), blockIdxs))
          throw std::runtime_error(
                                   "third argument must have a string list for it's 'block' key");
      } else if (strcmp(luaL_checkstring(L,-2), "thread") == 0) {
        if (!tostringvector(L, lua_gettop(L), threadIdxs))
          throw std::runtime_error(
                                   "third argument must have a string list for it's 'thread' key");
      } else {
        goto v2NotTable;
      }
      lua_pop(L, 1);
    }
  } else {
  v2NotTable: throw std::runtime_error(
                                       "third argument must be a table with 'block' and 'thread' as potential keys and list of indexes as values");
  }
  
  if (!tostringvector(L, 5, kernel_params))
    throw std::runtime_error(
                             "fifth argument must have a string list for it's kernel parameters");
  
  myloop->cudaize_v3(stmt_num, kernel_name, array_sizes, blockIdxs, // 5 args
                     threadIdxs, kernel_params);
  END_REQUIRE_LOOP;
  return 0;
}





int get_loop_num(lua_State *L) {
  lua_getglobal(L, "loop");
  int loop_num = lua_tointeger(L, -1);
  lua_pop(L, 1);
  return loop_num;
}


static int copy_to_texture(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 1);
  std::string array_name = luaL_optstring(L,1,0);
  myloop->copy_to_texture(array_name.c_str());
  END_REQUIRE_LOOP;
  return 0;
}

/*static int copy_to_texture_2d(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  std::string array_name = luaL_optstring(L, 1, 0);
  int width = luaL_optint(L, 2, 0);
  int height = luaL_optint(L, 3, 0); 
  myloop->copy_to_texture_2d(array_name.c_str(), width, height);
  END_REQUIRE_LOOP;
  return 0;
  }*/

//protonu-constant memory--place holder for now
static int copy_to_constant(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 1);
  std::string array_name = luaL_optstring(L,1,0);
  //call to loop->copy_to_texture goes here
  myloop->copy_to_constant(array_name.c_str());
  END_REQUIRE_LOOP;
  return 0;
  
}
#else

// CHILL, not CUDACHILL 
static int print_code(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  myloop->printCode();
  printf("\n");
  END_REQUIRE_LOOP;
  return 0;
}

static int print_dep(lua_State* L) {
  REQUIRE_LOOP;
  myloop->printDependenceGraph();
  END_REQUIRE_LOOP;
  return 0;
}

static int print_space(lua_State* L) {
  REQUIRE_LOOP;
  myloop->printIterationSpace();
  END_REQUIRE_LOOP;
  return 0;
}

static int exit(lua_State *L) {
  strict_arg_num(L, 0);
  repl_stop = true;
  return 0;
}

static int known(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 1);
  int num_dim = myloop->known.n_set();
  
  // parse expression from string
  std::vector<std::map<std::string, int> >* cond;
  std::string cond_expr = luaL_optstring(L,1,0);
  cond = parse_relation_vector(cond_expr.c_str());
  
  Relation rel(num_dim);
  F_And *f_root = rel.add_and();
  for (int j = 0; j < cond->size(); j++) {
    GEQ_Handle h = f_root->add_GEQ();
    for (std::map<std::string, int>::iterator it = (*cond)[j].begin(); it != (*cond)[j].end(); it++) {
      try {
        int dim = from_string<int>(it->first);
        if (dim == 0)
          h.update_const(it->second);
        else
          throw std::invalid_argument("only symbolic variables are allowed in known condition");
      }
      catch (std::ios::failure e) {
        Free_Var_Decl *g = NULL;
        for (unsigned i = 0; i < myloop->freevar.size(); i++) {
          std::string name = myloop->freevar[i]->base_name();
          if (name == it->first) {
            g = myloop->freevar[i];
            break;
          }
        }
        if (g == NULL)
          throw std::invalid_argument("symbolic variable " + it->first + " not found");
        else
          h.update_coef(rel.get_local(g), it->second);
      }
    }
  }
  myloop->addKnown(rel);
  END_REQUIRE_LOOP;
  return 0;
}

static int remove_dep(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  int from = luaL_optint(L, 1, 0);
  int to = luaL_optint(L, 2, 0);
  myloop->removeDependence(from, to);
  END_REQUIRE_LOOP;
  return 0;
}

static int original(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 0);
  myloop->original();
  END_REQUIRE_LOOP;
  return 0;
}

static int permute(lua_State *L) {
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 1) || (nargs > 3))
    throw std::runtime_error("incorrect number of arguments in permute");
  if(nargs == 1) {
    // premute ( vector )
    std::vector<int> pi;
    if(!tointvector(L, 1, pi))
      throw std::runtime_error("first arg in permute(pi) must be an int vector");
    myloop->permute(pi);
  }
  else if (nargs == 2) {
    // permute ( set, vector )
    std::set<int> active;
    std::vector<int> pi;
    if(!tointset(L, 1, active))
      throw std::runtime_error("the first argument in permute(active, pi) must be an int set");
    if(!tointvector(L, 2, pi))
      throw std::runtime_error("the second argument in permute(active, pi) must be an int vector");
    myloop->permute(active, pi);
  }
  else if (nargs == 3) {
    int stmt_num = luaL_optint(L, 1, 0);
    int level = luaL_optint(L, 2, 0);
    std::vector<int> pi;
    if(!tointvector(L, 3, pi))
      throw std::runtime_error("the third argument in permute(stmt_num, level, pi) must be an int vector");
    myloop->permute(stmt_num, level, pi);
  }
  END_REQUIRE_LOOP;
  return 0;
}

static int pragma(lua_State *L) { 
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  std::string pragmaText = luaL_optstring(L, 3, "");
  myloop->pragma(stmt_num, level, pragmaText);
  END_REQUIRE_LOOP;
  return 0;
}

static int prefetch(lua_State *L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  std::string prefetchText = luaL_optstring(L, 3, "");
  int hint = luaL_optint(L, 4, 0);
  myloop->prefetch(stmt_num, level, prefetchText, hint);
  END_REQUIRE_LOOP;
  return 0;
}

static int tile(lua_State* L) {
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 3) || (nargs > 7))
    throw std::runtime_error("incorrect number of arguments for tile");
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int tile_size = luaL_optint(L, 3, 0);
  if(nargs == 3) {
    myloop->tile(stmt_num, level, tile_size);
  }
  else if(nargs >= 4) {
    int outer_level = luaL_optint(L, 4, 0);
    if(nargs >= 5) {
      TilingMethodType method = StridedTile;
      int imethod = luaL_optint(L, 5, 2);
      // check method input against expected values
      if (imethod == 0)
        method = StridedTile;
      else if (imethod == 1)
        method = CountedTile;
      else
        throw std::runtime_error("5th argument must be either strided or counted");
      if(nargs >= 6) {
        int alignment_offset = luaL_optint(L, 6, 0);
        if(nargs == 7) {
          int alignment_multiple = luaL_optint(L, 7, 1);
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
  END_REQUIRE_LOOP;
  return 0;
}

static int datacopy(lua_State* L) {
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 3) || (nargs > 7))
    throw std::runtime_error("incorrect number of arguments for datacopy");
  // Overload 1: bool datacopy(const std::vector<std::pair<int, std::vector<int> > > &array_ref_nums, int level, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, int memory_type = 0);
  // Overload 2: bool datacopy(int stmt_num, int level, const std::string &array_name, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 4, int memory_type = 0);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  std::string array_name = std::string(luaL_optstring(L,3,0));
  bool allow_extra_read = luaL_optboolean(L, 4, false);
  int fastest_changing_dimension = luaL_optint(L, 5, -1);
  int padding_stride = luaL_optint(L, 6, 1);
  int padding_alignment = luaL_optint(L, 7, 4);
  int memory_type = luaL_optint(L, 8, 0);
  myloop->datacopy(stmt_num, level, array_name, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
  END_REQUIRE_LOOP;
  return 0;
}

static int datacopy_privatized(lua_State* L) {
  //  bool datacopy_privatized(int stmt_num, int level, const std::string &array_name, const std::vector<int> &privatized_levels, bool allow_extra_read = false, int fastest_changing_dimension = -1, int padding_stride = 1, int padding_alignment = 1, int memory_type = 0);
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 4) || (nargs > 9))
    throw std::runtime_error("incorrect number of arguments for datacopy_privatized");
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  std::string array_name = std::string(luaL_optstring(L, 3, 0));
  std::vector<int> privatized_levels;
  tointvector(L, 4, privatized_levels);
  bool allow_extra_read = luaL_optboolean(L, 5, false);
  int fastest_changing_dimension = luaL_optint(L, 6, -1);
  int padding_stride = luaL_optint(L, 7, 1);
  int padding_alignment = luaL_optint(L, 8, 1);
  int memory_type = luaL_optint(L, 9, 0);
  myloop->datacopy_privatized(stmt_num, level, array_name, privatized_levels, allow_extra_read, fastest_changing_dimension, padding_stride, padding_alignment, memory_type);
  END_REQUIRE_LOOP;
  return 0;
}

static int unroll(lua_State* L) {
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 3) || (nargs > 4))
    throw std::runtime_error("incorrect number of arguments for unroll");
  //std::set<int> unroll(int stmt_num, int level, int unroll_amount, std::vector< std::vector<std::string> >idxNames= std::vector< std::vector<std::string> >(), int cleanup_split_level = 0);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int unroll_amount = luaL_optint(L, 3, 0);
  std::vector< std::vector<std::string> > idxNames = std::vector< std::vector<std::string> >();
  int cleanup_split_level = luaL_optint(L, 4, 0);
  myloop->unroll(stmt_num, level, unroll_amount, idxNames, cleanup_split_level);
  END_REQUIRE_LOOP;
  return 0;
}

static int unroll_extra(lua_State* L) {
  REQUIRE_LOOP;
  int nargs = lua_gettop(L);
  if((nargs < 3) || (nargs < 4))
    throw std::runtime_error("incorrect number of arguments for unroll_extra");
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int unroll_amount = luaL_optint(L, 3, 0);
  int cleanup_split_level = luaL_optint(L, 4, 0);
  myloop->unroll_extra(stmt_num, level, unroll_amount, cleanup_split_level); 
  END_REQUIRE_LOOP;
  return 0;
}

static int split(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int num_dim = myloop->stmt[stmt_num].xform.n_out();
  
  // parse expression from string
  std::vector<std::map<std::string, int> >* cond;
  std::string cond_expr = luaL_optstring(L,3,0);
  cond = parse_relation_vector(cond_expr.c_str());
  
  Relation rel((num_dim-1)/2);
  F_And *f_root = rel.add_and();
  for (int j = 0; j < cond->size(); j++) {
    GEQ_Handle h = f_root->add_GEQ();
    for (std::map<std::string, int>::iterator it = (*cond)[j].begin(); it != (*cond)[j].end(); it++) {
      try {
        int dim = from_string<int>(it->first);
        if (dim == 0)
          h.update_const(it->second);
        else {
          if (dim > (num_dim-1)/2)
            throw std::invalid_argument("invalid loop level " + to_string(dim) + " in split condition");
          h.update_coef(rel.set_var(dim), it->second);
        }
      }
      catch (std::ios::failure e) {
        Free_Var_Decl *g = NULL;
        for (unsigned i = 0; i < myloop->freevar.size(); i++) {
          std::string name = myloop->freevar[i]->base_name();
          if (name == it->first) {
            g = myloop->freevar[i];
            break;
          }
        }
        if (g == NULL)
          throw std::invalid_argument("unrecognized variable " + to_string(it->first.c_str()));
        h.update_coef(rel.get_local(g), it->second);
      }
    }
  }
  myloop->split(stmt_num,level,rel);
  END_REQUIRE_LOOP;
  return 0;
}

static int nonsingular(lua_State* L) {
  REQUIRE_LOOP;
  std::vector< std::vector<int> > mat;
  tointmatrix(L, 1, mat);
  myloop->nonsingular(mat);
  END_REQUIRE_LOOP;
  return 0;
}

static int skew(lua_State* L) {
  REQUIRE_LOOP;
  std::set<int> stmt_nums;
  std::vector<int> skew_amounts;
  int level = luaL_optint(L, 2, 0);
  tointset(L, 1, stmt_nums);
  tointvector(L, 3, skew_amounts);
  myloop->skew(stmt_nums, level, skew_amounts);
  END_REQUIRE_LOOP;
  return 0;
}

static int scale(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  std::set<int> stmt_nums;
  int level = luaL_optint(L, 2, 0);
  int scale_amount = luaL_optint(L, 3, 0);
  tointset(L, 1, stmt_nums);
  myloop->scale(stmt_nums, level, scale_amount);
  END_REQUIRE_LOOP;
  return 0;
}

static int reverse(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  std::set<int> stmt_nums;
  int level = luaL_optint(L, 2, 0);
  tointset(L, 1, stmt_nums);
  myloop->reverse(stmt_nums, level);
  END_REQUIRE_LOOP;
  return 0;
}

static int shift(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  std::set<int> stmt_nums;
  int level = luaL_optint(L, 2, 0);
  int shift_amount = luaL_optint(L, 3, 0);
  tointset(L, 1, stmt_nums);
  myloop->shift(stmt_nums, level, shift_amount);
  END_REQUIRE_LOOP;
  return 0;
}

static int shift_to(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int absolute_pos = luaL_optint(L, 3, 0);
  myloop->shift_to(stmt_num, level, absolute_pos);
  END_REQUIRE_LOOP;
  return 0;
}

static int peel(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2, 3);
  int stmt_num = luaL_optint(L, 1, 0);
  int level = luaL_optint(L, 2, 0);
  int amount = luaL_optint(L, 3, 1);
  myloop->peel(stmt_num, level, amount);
  END_REQUIRE_LOOP;
  return 0;
}

static int fuse(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  std::set<int> stmt_nums;
  int level = luaL_optint(L, 2, 0);
  tointset(L, 1, stmt_nums);
  myloop->fuse(stmt_nums, level);
  END_REQUIRE_LOOP;
  return 0;
}

static int distribute(lua_State* L) {
  REQUIRE_LOOP;
  strict_arg_num(L, 2);
  std::set<int> stmts;
  int level = luaL_optint(L, 1, 0);
  tointset(L, 2, stmts);
  myloop->distribute(stmts, level);
  END_REQUIRE_LOOP;
  return 0;
}

static int num_statements(lua_State *L) {
  REQUIRE_LOOP;
  lua_pushinteger(L, myloop->stmt.size());
  END_REQUIRE_LOOP;
  return 1;
}
#endif

/**
 * Register global methods available to chill scripts
 */
void register_globals(lua_State *L) {
  //---
  //Preset globals
  //---
  lua_pushstring(L, CHILL_BUILD_VERSION);
  lua_setglobal(L, "VERSION");
  
  lua_pushstring(L, "C");
  lua_setglobal(L, "dest");
  lua_pushstring(L, "C");
  lua_setglobal(L, "C");
  
  //---
  //Enums for functions
  //---
  
  //TileMethod
  lua_pushinteger(L, 0);
  lua_setglobal(L, "strided");
  lua_pushinteger(L, 1);
  lua_setglobal(L, "counted");
  
  //MemoryMode
  lua_pushinteger(L, 0);
  lua_setglobal(L, "global");
  lua_pushinteger(L, 1);
  lua_setglobal(L, "shared");
  lua_pushinteger(L, 2);
  lua_setglobal(L, "texture");
  
  //Bool flags
  lua_pushboolean(L, 1);
  lua_setglobal(L, "sync");
  //...
}

#ifdef CUDACHILL
void register_functions(lua_State *L) {
  lua_register(L, "init", init);
  lua_register(L, "exit", exit);
  lua_register(L, "print_code", print_code);
  lua_register(L, "print_ri", print_ri);
  lua_register(L, "print_idx", print_idx);
  lua_register(L, "print_dep", print_dep);
  lua_register(L, "print_space", print_space);
  lua_register(L, "num_statement", num_statement);
}

void register_v2(lua_State *L) {
  lua_register(L, "cudaize5arg", cudaize_v3);
  lua_register(L, "cudaize", cudaize_v2);
  lua_register(L, "tile", tile_v2);
  lua_register(L, "permute", permute_v2);
  lua_register(L, "datacopy_privatized", datacopy_privatized_v2);
  lua_register(L, "datacopy", datacopy_v2);
  lua_register(L, "ELLify", ELLify_v2);
  lua_register(L, "unroll", unroll_v2);
  //lua_register(L, "coalesce", flatten_v2);
  lua_register(L, "distribute", distribute_v2);
  lua_register(L, "fuse", fuse_v2);
  lua_register(L, "peel", peel_v2);
  lua_register(L, "shift_to", shift_to_v2);
  lua_register(L, "scalar_expand", scalar_expand_v2);
  lua_register(L, "reduce", reduce_v2);
  lua_register(L, "skew", skew_v2);
  lua_register(L, "split_with_alignment", split_with_alignment_v2);
  lua_register(L, "compact", compact_v2);
  lua_register(L, "make_dense", make_dense_v2);
  lua_register(L, "known", addKnown_v2);
  
  
  lua_register(L, "cur_indices", cur_indices);
  lua_register(L, "block_indices", block_indices);
  lua_register(L, "thread_indices", thread_indices);
  lua_register(L, "block_dims", block_dims);
  
  lua_register(L, "thread_dims", thread_dims);
  lua_register(L, "hard_loop_bounds", hard_loop_bounds);
  lua_register(L, "num_statements", num_statement);
  
  lua_register(L, "does_exists", does_var_exists);
  lua_register(L, "add_sync", add_sync);
  lua_register(L, "rename_index", rename_index);
  
  lua_register(L, "copy_to_texture", copy_to_texture);
  lua_register(L, "copy_to_constant", copy_to_constant);
}

#else // CHiLL
void register_functions(lua_State* L) {
  lua_register(L, "source", source);
  lua_register(L, "procedure", procedure);
  lua_register(L, "loop", loop);
  lua_register(L, "print_code", print_code);
  lua_register(L, "print_dep", print_dep);
  lua_register(L, "print_space", print_space);
  lua_register(L, "exit", exit);
  lua_register(L, "known", known);
  lua_register(L, "remove_dep", remove_dep);
  lua_register(L, "original", original);
  lua_register(L, "permute", permute);
  lua_register(L, "pragma", pragma);
  lua_register(L, "prefetch", prefetch);
  lua_register(L, "tile", tile);
  lua_register(L, "datacopy", datacopy);
  lua_register(L, "datacopy_privatised", datacopy_privatized);
  lua_register(L, "unroll", unroll);
  lua_register(L, "unroll_extra", unroll_extra);
  lua_register(L, "split", split);
  lua_register(L, "nonsingular", nonsingular);
  lua_register(L, "skew", skew);
  lua_register(L, "scale", scale);
  lua_register(L, "reverse", reverse);
  lua_register(L, "shift", shift);
  lua_register(L, "shift_to", shift_to);
  lua_register(L, "peel", peel);
  lua_register(L, "fuse", fuse);
  lua_register(L, "distribute", distribute);
  lua_register(L, "num_statements", num_statements);
}
#endif
