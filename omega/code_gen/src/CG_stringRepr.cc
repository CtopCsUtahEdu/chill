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
#include <string.h>

namespace omega {
  
  //-----------------------------------------------------------------------------
  // Dump operations
  //-----------------------------------------------------------------------------
  void CG_stringRepr::Dump() const { dump(); } // TODO combine dump() and Dump()
  
  void CG_stringRepr::DumpToFile(FILE *fp) const {
    fprintf(fp,"%s", s_.c_str());
  }
  
} // namespace
