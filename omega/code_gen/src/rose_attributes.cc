
#ifdef FRONTEND_ROSE 

#include <code_gen/rose_attributes.h>

namespace omega {

CodeInsertionAttribute* getOrCreateCodeInsertionAttribute(SgNode* node) {
	CodeInsertionAttribute* attr;
	if(node->attributeExists("code_insertion"))
		return static_cast<CodeInsertionAttribute*>(node->getAttribute("code_insertion"));
	attr = new CodeInsertionAttribute();
	node->setAttribute("code_insertion", attr);
	return attr;
}

void postProcessRoseCodeInsertion(SgProject* proj) {
	//generatePDF(*proj);
	CodeInsertionVisitor visitor = CodeInsertionVisitor();
	visitor.initialize();
	visitor.traverseInputFiles(proj);
	visitor.insertCode();
}

// Swap a code insertion from one node (sn) to another (dn)
// -- note that this function does not currently remove the insertion from the sn node
void moveCodeInsertion(SgNode* sn, CodeInsertion* ci, SgNode* dn) {
	CodeInsertionAttribute* new_attr;
	// TODO in the near future: replace the above statement with 'new_attr = getOrCreateCodeInsertionAttribute(...)'
	CodeInsertionAttribute* old_attr = static_cast<CodeInsertionAttribute*>(sn->getAttribute("code_insertion"));
	if(dn->attributeExists("code_insertion")) {
		new_attr = static_cast<CodeInsertionAttribute*>(dn->getAttribute("code_insertion"));
	}
	else {
		new_attr = new CodeInsertionAttribute();
		dn->setAttribute("code_insertion", new_attr);
	}
	new_attr->add(ci);
}

// A function that copies a specific attribute from one node to another
// this function exists to get around a ROSE limitation that does not
// copy attributes
void copyAttribute(std::string attr_name, SgNode* s, SgNode* d) {
	if(s->attributeExists(attr_name)) {
		d->setAttribute(attr_name,s->getAttribute(attr_name));
	}
}

// TODO: find all existng attributes and iterate over them instead of doing them
//       individually
void copyAttributes(SgNode* s, SgNode* d) {
	copyAttribute("code_insertion", s, d);
	//...any other attributes...
}

void CodeInsertionVisitor::initialize() {
	this->loop_level = 0;
	this->ci_marks = std::vector<CodeInsertionMark*>();
}

void CodeInsertionVisitor::markStmt(SgStatement* stmt, CodeInsertion* ci) {
	// this check prevents multiple copies of stmts
	// -- may be changed in the future
	if(!ci->marked) {
		CodeInsertionMark* pos = new CodeInsertionMark();
		pos->stmt = stmt;
		pos->ci = ci;
		this->ci_marks.push_back(pos);
		ci->marked = true;
	}
}

// increase loop_level as the visitor descends
void CodeInsertionVisitor::preOrderVisit(SgNode* n) {
	if (isSgForStatement(n)) {
		this->loop_level++;
	}
}

void CodeInsertionVisitor::postOrderVisit(SgNode* n) {
	if(isSgForStatement(n)) {
		this->loop_level--;
	}
	if(isSgStatement(n)) {
		if(n->attributeExists("code_insertion")) {
			CodeInsertionAttribute *attr = static_cast<CodeInsertionAttribute*>(n->getAttribute("code_insertion"));
			for(CodeInsertionPtrListItr itr = attr->begin(); itr != attr->end(); ++itr) {
				CodeInsertion *insertion = *itr;
				// check loop level -- if it is equivelent, mark statement for insertion
				//                  -- else, move attribute up to parent
				if(insertion->loop_level != this->loop_level) {
					moveCodeInsertion(n, insertion, n->get_parent());
				}
				else {
					this->markStmt(isSgStatement(n), insertion);
				}
			}
		}
	}
}

// final stage of algorithm that inserts marked statements
void CodeInsertionVisitor::insertCode() {
	for(std::vector<CodeInsertionMark*>::iterator itr = this->ci_marks.begin(); itr != this->ci_marks.end(); ++itr) {
		CodeInsertionMark* mark = *itr;
		SgScopeStatement* scope = static_cast<SgScopeStatement*>(mark->stmt->get_parent());
		SageInterface::insertStatementBefore(mark->stmt, mark->ci->getStatement(scope));
	}
}

SgStatement* PragmaInsertion::getStatement(SgScopeStatement* scopeStmt) {
	SgStatement* stmt = SageBuilder::buildPragmaDeclaration(this->name);
	return stmt;
}

//SgStatement* MMPrefetchInsertion::getStatement(SgScopeStatement* scopeStmt) {
//	const SgName& name = SgName("_mm_prefetch");
//	SgType* rtype = SageBuilder::buildVoidType();
//	SgExpression* arr_arg = SageBuilder::buildVarRefExp(this->arrName);
//	SgExpression* hint_arg = SageBuilder::buildShortVal(this->cacheHint);
//	SgExprListExp* args = SageBuilder::buildExprListExp(arr_arg,hint_arg);
//	SgStatement* stmt = SageBuilder::buildFunctionCallStmt(name, rtype, args, scopeStmt);
//	return stmt;
//}

SgStatement* MMPrefetchInsertion::getStatement(SgScopeStatement* scopeStmt) {
	const SgName fname = SgName("_mm_prefetch");
	SgType* rtype = SageBuilder::buildVoidType();
	SgExpression* arr_arg = this->buildArrArg(scopeStmt);
	SgExpression* hint_arg = SageBuilder::buildShortVal(this->cacheHint);
	SgExprListExp* args = SageBuilder::buildExprListExp(arr_arg, hint_arg);
	return SageBuilder::buildFunctionCallStmt(fname, rtype, args, scopeStmt);
}

SgExpression* MMPrefetchInsertion::buildArrArg(SgScopeStatement* scopeStmt) {
	// if there are no index arguments given, just return a variable reference
	if(this->indexCount == 0) {
		const SgName aname = SgName(this->arrName);
		return SageBuilder::buildVarRefExp(aname, scopeStmt);
	}
	std::vector<SgExpression*> argList = std::vector<SgExpression*>();
	// foreach dimension
	for(int i = 0; i < this->indexCount; i++) {
		argList.push_back(this->makeIndexExp(i, scopeStmt));
	}
	return SageBuilder::buildExprListExp(argList);
}

SgExpression* MMPrefetchInsertion::makeIndexExp(int dim, SgScopeStatement* scopeStmt) {
	//(i + offset) or (offset) or (i)
	std::string* indexer = this->indecies.at(dim);
	int offset = this->offsets.at(dim);
	if(indexer == NULL) {
		return SageBuilder::buildIntVal(offset);
	}
	else {
		const SgName name = SgName(*indexer);
		SgVarRefExp* iref = SageBuilder::buildVarRefExp(name, scopeStmt);
		if(offset == 0) {
			return iref;
		}
		else {
			return SageBuilder::buildAddOp(iref, SageBuilder::buildIntVal(offset));
		}
	}
}

void MMPrefetchInsertion::initialize(const std::string& arrName, int hint) {
	this->arrName = std::string(arrName);
	this->cacheHint = hint;
	this->indecies = std::vector<std::string*>();
	this->offsets = std::vector<int>();
	this->indexCount = 0;
}
void MMPrefetchInsertion::addDim(int offset) {
	this->offsets.push_back(offset);
	this->indecies.push_back(NULL);
	this->indexCount++;
}
void MMPrefetchInsertion::addDim(int offset, const std::string& indexer) {
	this->offsets.push_back(offset);
	this->indecies.push_back(new std::string(indexer));
	this->indexCount++;
}
}


#endif 
