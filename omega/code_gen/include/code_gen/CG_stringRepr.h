/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
   pseudo string code wrapper

 Notes:

 History:
   04/17/96 - Lei Zhou - created
*****************************************************************************/

#ifndef _CG_STRINGREPR_H
#define _CG_STRINGREPR_H

#include <code_gen/CG_outputRepr.h>
#include <string>
#include <iostream>
#include <stdio.h>

namespace omega {

class CG_stringRepr: public CG_outputRepr {
private:
  std::string s_;

public:
  char *type() const { return strdup("string"); }; 


  CG_stringRepr() {}; 
  CG_stringRepr(const std::string &s){ s_ = s; }
  ~CG_stringRepr() {} 
  CG_outputRepr *clone() const { return new CG_stringRepr(s_); }
  void dump() const { std::cout << s_ << std::endl; }
  void Dump() const;
  void DumpToFile(FILE *fp = stderr) const;

  //---------------------------------------------------------------------------
  // basic operation
  //---------------------------------------------------------------------------
  std::string GetString() const { return s_; }
};

}

#endif
