#ifndef CHILL_ENV_H
#define CHILL_ENV_H

typedef struct lua_State lua_State;

void register_globals(lua_State *L);
void register_functions(lua_State *L);
#ifdef CUDACHILL
int get_loop_num(lua_State *L);
#else
void finalize_loop(int loop_num_start, int loop_num_end);
int get_loop_num_start(lua_State *L);
int get_loop_num_end(lua_State *L);
#endif
#endif
