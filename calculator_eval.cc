#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_container.hpp>
#include <boost/spirit/include/phoenix_statement.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <omega/code_gen/include/codegen.h>
#include "loop.hh"
#include "ir_code.hh"
#include "chill_error.hh"

using namespace omega;

namespace client
{
  namespace qi = boost::spirit::qi;
  namespace phoenix = boost::phoenix;
  namespace ascii = boost::spirit::ascii;
  
  struct createIdentifier
  {
  public:
    
    createIdentifier(std::vector<CG_outputRepr *> &code, std::vector<std::string> &code_string, std::string & string_ref, CG_outputBuilder &builder_, std::vector<std::string> &index_names) :m_code(code), code_string_(code_string), global_string(string_ref), builder(builder_), index_names_(index_names){}
    template <typename IteratorT>
    void operator()( IteratorT, IteratorT ) const
    {
      
    }
    template <typename IteratorT>
    void operator()( IteratorT ) const
    {
      
    }
    
    //template <typename IteratorT>
    void operator()(qi::unused_type,qi::unused_type, qi::unused_type ) const
    { 
      m_code.push_back(builder.CreateIdent(global_string));
      code_string_.push_back(global_string);
      global_string.clear();
    }
    
    //template <typename IteratorT>
    void operator()(char &s ,qi::unused_type, qi::unused_type ) const
    { 
      global_string += s;
    }
    
    //template <typename IteratorT>
    void operator()(int &s ,qi::unused_type, qi::unused_type ) const
    { 
      
      assert(code_string_.size() >= 2);
      std::string member_name = index_names_[s];
      std::string function_name = code_string_[code_string_.size() -2];
      std::string argument_name = code_string_[code_string_.size() -1];

      code_string_.pop_back();
      code_string_.pop_back();
      if(function_name.find('_') != std::string::npos)
        {
          
          function_name = function_name.substr(0, function_name.find('_') );
          
        }
      
      std::cout<<"sanity check"<<std::endl;
      std::cout<< member_name<<std::endl;
      std::cout<< function_name<<std::endl;
      std::cout<< argument_name<<std::endl;
      m_code.pop_back();
      m_code.pop_back();
      m_code.push_back(builder.CreateArrayRefExpression(
                                                        
                                                        builder.ObtainInspectorData(
                                                                                    function_name, member_name),
                                                        builder.CreateIdent(argument_name)));
    }
    
  private:
    std::vector<CG_outputRepr *> &m_code;
    std::vector<std::string> &index_names_;
    std::vector<std::string> &code_string_;
    CG_outputBuilder &builder;
    std::string &global_string ;
  };  
  
  
  struct checkInvoke
  {
  public:
    
    checkInvoke(std::vector<CG_outputRepr *> &code, std::vector<std::string> &code_string, std::string & string_ref, CG_outputBuilder &builder_,std::vector<std::string> &index_names) :m_code(code), code_string_(code_string), global_string(string_ref), builder(builder_), index_names_(index_names){}
    
    //template <typename IteratorT>
    void operator()(qi::unused_type ,qi::unused_type, qi::unused_type ) const
    {
      
      if(m_code.size() == 2){
        auto func = m_code[0];
        auto param = m_code[1];
        auto fname = code_string_[0];
        m_code.clear();
        code_string_.clear();
        // function_name may actually be an array.   ???  TODO
        std::vector<CG_outputRepr*> p;
        p.push_back(param);
        m_code.push_back(builder.CreateInvoke(fname, p, true)); // <-- it IS an array
        std::cout<<"Success"<<std::endl;
      }
      
    }
    
  private:
    std::vector<CG_outputRepr *> &m_code;
    CG_outputBuilder &builder;
    std::string &global_string ;
    std::vector<std::string> &index_names_;
    std::vector<std::string> &code_string_;
  };

  template <typename Iterator>
  struct calculator : qi::grammar<Iterator, ascii::space_type>
  {
    calculator(std::vector<CG_outputRepr *>& code, std::vector<std::string> &code_string_, CG_outputBuilder &builder, std::vector<std::string> &index_names)
      : calculator::base_type(expression)
      , code(code), index_names_(index_names), builder_(builder), code_string(code_string_)
    {
      using namespace qi::labels;
      using qi::int_;
      using ascii::char_;
      using qi::on_error;
      using qi::fail;
      
      using phoenix::val;
      using phoenix::ref;
      using phoenix::push_back;
      using phoenix::construct;

      expression =
        term
            >> *(   ('+' >> term)
                |   ('-' >> term)
                |   ('*' >> term)
                |   ('/' >> term));

      term =
          (identifier >> -('(' > expression > ')') >> -('[' > int_[client::createIdentifier(code, code_string, global_string,builder_,index_names_)] >>  ']') [client::checkInvoke(code, code_string, global_string,builder_,index_names_)])
        | int_[std::cout<<_1<<std::endl] //int_[code.push_back(builder.CreateInt(static_cast<int>(_1)))]
        | '(' >> expression >> ')'
        ;

      identifier =
          identifyP    [client::createIdentifier(code, code_string, global_string,builder_,index_names_)];

      identifyP = char_("a-zA-Z") [client::createIdentifier(code, code_string, global_string,builder_,index_names_)]
        >> *(char_("_a-zA-Z0-9")[client::createIdentifier(code, code_string,global_string,builder_,index_names_)])
        ;     
      
      expression.name("expression");
      term.name("term");
      identifier.name("identifier");
      identifyP.name("identifyP");
      on_error<fail>
        (
         expression
         , std::cout
         << val("Error! Expecting ")
         << _4                               // what failed?
         << val(" here: \"")
         << construct<std::string>(_3, _2)   // iterators to error-pos, end
         << val("\"")
         << std::endl
         );
    }
    
    qi::rule<Iterator, ascii::space_type> expression, term, identifier, identifyP;
    std::vector<CG_outputRepr *>& code;
    std::vector<std::string>& code_string;
    std::vector<std::string> &index_names_ ;
    std::string global_string;
    CG_outputBuilder &builder_;
    
  };
  
  template <typename Grammar>
  bool compile(Grammar const& calc, std::string const& expr)
  {
    std::string::const_iterator iter = expr.begin();
    std::string::const_iterator end = expr.end();
    bool r = phrase_parse(iter, end, calc, ascii::space);
    
    return r && iter == end;
  }
}

///////////////////////////////////////////////////////////////////////////////
//  Main program
///////////////////////////////////////////////////////////////////////////////
CG_outputRepr * Loop::iegen_parser(std::string &str, std::vector<std::string> &index_names)
{
  
  debug_fprintf(stderr, "Loop::iegen_parser()\nstr = \"%s\"\n", str.c_str());
  debug_fprintf(stderr, "index_names: ");
  int n = index_names.size();
  for (int i=0; i<n; i++) { 
    debug_fprintf(stderr, "%s", index_names[i].c_str());
    if (i<(n-1)) debug_fprintf(stderr, ", ");
  }
  debug_fprintf(stderr, "\n\n");
  
  typedef std::string::const_iterator iterator_type;
  typedef client::calculator<iterator_type> calculator;
  
  
  std::vector<CG_outputRepr *> code;
  std::vector<std::string> code_string;
  calculator calc(code, code_string, *(ir->builder()), index_names);

  if (!client::compile(calc, str))
    throw loop_error("iegen string parsing failed");
  
  debug_fprintf(stderr, "Loop::iegen_parser DONE\n");
  for (auto i: code)
    i->dump();
  return code[0];
}
