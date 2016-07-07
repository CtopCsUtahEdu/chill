#include "chill_io.hh"

#ifdef DEBUGCHILL

#include <algorithm>
#include <vector>
#include <string>

static bool                         __debug_enabled = false;
static std::vector<std::string>     __debug_symbols;

void debug_enable(bool enable) {
    __debug_enabled = enable;
}

void debug_define(char* symbol) {
    __debug_symbols.push_back(std::string(symbol));
}

bool debug_isenabled() {
    return __debug_enabled;
}

bool debug_isdefined(char* symbol) {
    return __debug_symbols.end() != std::find(__debug_symbols.begin(), __debug_symbols.end(), std::string(symbol));
}

#endif
