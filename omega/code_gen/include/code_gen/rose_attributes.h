#ifndef ROSE_ATTRIBUTES_HH
#define ROSE_ATTRIBUTES_HH

#include "rose.h"
#include <algorithm>
#include <string>
#include <vector>

namespace omega {

class CodeInsertion;

typedef std::vector<CodeInsertion*> CodeInsertionPtrList;
typedef std::vector<CodeInsertion*>::iterator CodeInsertionPtrListItr;

class CodeInsertion {
public:
	int loop_level;
	bool marked;
	CodeInsertion(int looplevel) { this->loop_level = looplevel; marked = false; }
	~CodeInsertion() {}
	virtual SgStatement* getStatement(SgScopeStatement* scopeStmt = NULL) = 0;
};

class PragmaInsertion : public CodeInsertion {
private:
	std::string name;
public:
	PragmaInsertion(int loop_level, const std::string &pragma) : CodeInsertion(loop_level) { this->name = std::string(pragma); }
	~PragmaInsertion() { }
	virtual SgStatement* getStatement(SgScopeStatement* scopeStmt = NULL);
};

class MMPrefetchInsertion : public CodeInsertion {
private:
	std::string arrName;
	std::vector<std::string*> indecies;
	std::vector<int> offsets;
	int indexCount;
	int cacheHint;
	void initialize(const std::string& arrName, int hint);
	void addDim(int offset);
	void addDim(int offset, const std::string& indexer);
	SgExpression* buildArrArg(SgScopeStatement* scopeStmt);
	SgExpression* makeIndexExp(int dim, SgScopeStatement* scopeStmt);
public:
	MMPrefetchInsertion(int loop_level, const std::string &arr, int hint) : CodeInsertion(loop_level)
		{ this->initialize(arr, hint); }
	~MMPrefetchInsertion() { }
	virtual SgStatement* getStatement(SgScopeStatement* scopeStmt = NULL);
};

class CodeInsertionAttribute : public AstAttribute {
private:
	std::vector<CodeInsertion*> code_insertions;
public:
	CodeInsertionAttribute() { code_insertions = std::vector<CodeInsertion*>(); }
	~CodeInsertionAttribute() {}
	
	void add(CodeInsertion* ci) { code_insertions.push_back(ci); }
	CodeInsertionPtrListItr begin() { return code_insertions.begin(); }
	CodeInsertionPtrListItr end() { return code_insertions.end(); }
	void remove(CodeInsertion* ci) { std::remove(code_insertions.begin(), code_insertions.end(), ci); }
	int countCodeInsertions() { return code_insertions.size(); }
};

struct CodeInsertionMark {
public:
	SgStatement* stmt;
	CodeInsertion* ci;
};

class CodeInsertionVisitor : public AstPrePostProcessing {
private:
	int loop_level;
	std::vector<CodeInsertionMark*> ci_marks;
	void markStmt(SgStatement* stmt, CodeInsertion* ci);
public:
	void initialize();
	virtual void preOrderVisit(SgNode* n);
	virtual void postOrderVisit(SgNode* n);
	void insertCode();
};

void postProcessRoseCodeInsertion(SgProject* proj);
void copyAttributes(SgNode* s, SgNode* d);
CodeInsertionAttribute* getOrCreateCodeInsertionAttribute(SgNode* node);

}

#endif
