/*****************************************************************************
 Copyright (C) 1994-2000 University of Maryland
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2009 University of Utah
 All Rights Reserved.

 Purpose:
   omega holder for string implementation.

 Notes:

 History:
   04/17/96 - Lei Zhou - created
*****************************************************************************/

#include <code_gen/CG_stringRepr.h>
#include <stdio.h>

namespace omega {

CG_stringRepr::CG_stringRepr() {
}

CG_stringRepr::CG_stringRepr(const std::string& _s) : s(_s) {
}

CG_stringRepr::~CG_stringRepr() {
}

CG_outputRepr* CG_stringRepr::clone() {
  return new CG_stringRepr(s);
}


//-----------------------------------------------------------------------------
// basic operation
//-----------------------------------------------------------------------------
std::string CG_stringRepr::GetString() const { 
  return s;
}


//-----------------------------------------------------------------------------
// Dump operations
//-----------------------------------------------------------------------------
void CG_stringRepr::Dump() const {
  printf("%s\n", s.c_str());
}

void CG_stringRepr::DumpToFile(FILE *fp) const {
  fprintf(fp,"%s", s.c_str());
}

} // namespace
