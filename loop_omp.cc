/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009-2017 University of Utah
 All Rights Reserved.

 History:
 11/2017 Created by Derick Huth
*****************************************************************************/


#include <climits>
#include <cmath>
#include <omega/code_gen/include/codegen.h>
#include <code_gen/CG_utils.h>
#include <code_gen/CG_chillBuilder.h> // Manu   bad idea.  TODO
#include <code_gen/CG_stringRepr.h>
#include <code_gen/CG_chillRepr.h>   // Mark.  Bad idea.  TODO
#include <algorithm>
#include <map>
#include "loop.hh"
#include "omegatools.hh"
#include "irtools.hh"
#include "chill_error.hh"
#include <cstring>




void Loop::omp_mark_pragma(int stmt, int level, std::string name) {
    this->general_pragma_info.push_back(PragmaInfo(stmt, level, name));
}

void Loop::omp_mark_parallel_for(int stmt, int level, const std::vector<std::string>& privitized_vars, const std::vector<std::string>& shared_vars) {
    this->omp_pragma_info.push_back(OMPPragmaInfo(stmt, level, privitized_vars, shared_vars));
}

void Loop::omp_apply_pragmas() const {
    for(auto pinfo: this->general_pragma_info) {
        this->last_compute_cgr_->addPragma(pinfo.stmt, pinfo.loop_level, pinfo.name);
    }

    for(auto pinfo: this->omp_pragma_info) {
        this->last_compute_cgr_->addOmpPragma(pinfo.stmt, pinfo.loop_level, pinfo.privitized_vars, pinfo.shared_vars);
    }
}

