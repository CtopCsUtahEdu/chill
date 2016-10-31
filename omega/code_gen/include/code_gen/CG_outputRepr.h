/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   abstract base class of compiler IR code wrapper

 Notes:

 History:
   04/17/96 - Lei Zhou - created
*****************************************************************************/

#ifndef _CG_OUTPUTREPR_H
#define _CG_OUTPUTREPR_H

#include "chill_io.hh"

#include <string.h>

namespace omega {

class CG_outputRepr {
public:
  
  CG_outputRepr() {}
  virtual ~CG_outputRepr() { /* shallow delete */ }
  virtual CG_outputRepr *clone() const = 0;
  virtual void clear() { /* delete actual IR code wrapped inside */ }
  virtual void dump() const {}
  virtual char *type() const = 0; 
};

}

#endif
