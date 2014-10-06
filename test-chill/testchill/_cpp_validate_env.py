import ast as _pyast
import collections as _pycollections
import functools as _pyfunctools
import itertools as _pyitertools
import random as _pyrandom
import struct as _pystruct
import types as _pytypes

from . import util as _chill_util

_pylambdatype = _pycollections.namedtuple('LambdaType', ['paramtypes','exprtype'])
_pyarraytype = _pycollections.namedtuple('ArrayType', ['dimensions','basetype'])

_runtime_globals = dict({
        '_pyitertools':_pyitertools,
        '_pyrandom':_pyrandom
    })

def _evalexpr(expr, target_type, bindings):
    glbls = dict(bindings)
    glbls.update(_runtime_globals)
    if target_type is None:
        pytype = None
    else:
        pytype = target_type.getpytype()
    expr = _pyast.Expression(expr.compile_expr(pytype))
    expr = _pyast.fix_missing_locations(expr)
    return eval(compile(expr, '<string>', 'eval'), glbls)

def _addbindings(expr, binding_frame):
    if hasattr(expr, 'binding_stack'):
        expr.binding_stack = [binding_frame] + expr.binding_stack
    return expr


class _TreeNode(object):
    def print_tree(self, stream=None, indent=0):
        strname = type(self).__name__
        stream.write(strname + ':\n')
        indent += 2
        for k,v in vars(self).items():
            if isinstance(v, _TreeNode):
                stream.write(('{}{}:'.format(' '*indent, k)))
                v.print_tree(stream, indent + len(k))
            elif isinstance(v, list):
                stream.write(('{}{}: [\n'.format(' '*indent, k)))
                for itm in v:
                    if isinstance(itm, _TreeNode):
                        stream.write(' '*indent)
                        itm.print_tree(stream, indent + len(k) + 1)
                    else:
                        stream.write('{}{}\n'.format(' '*(indent + 1), str(itm)))
            else:
                stream.write(('{}{}: {}\n'.format(' '*indent, k, str(v))))

class _CppType(_TreeNode):
    def __init__(self):
        pass
    
    def __repr__(self):
        return "{}".format(str(self))
    
    def statictype(self, bindings):
        return self
    
    def formatdata(self, data):
        raise NotImplementedError
    
    def get_cdecl_stmt(self, param_name):
        raise NotImplementedError
    
    def get_cread_stmt(self, param_name, istream_name, dims):
        raise NotImplementedError
    
    def get_cwrite_stmt(self, param_name, ostream_name, dims):
        raise NotImplementedError
    
    def getfreevars(self, glbls):
        raise NotImplementedError


class _CppPrimitiveType(_CppType):
    _bycppname = {
            'char':                 ('char', 'c', 1, False, False, True, False),
            'signed char':          ('signed char', 'b', 1, True, False, False, False),
            'unsigned char':        ('unsigned char', 'B', 1, True, False, False, False),
            'short':                ('short', 'h', 2, True, False, False, True),
            'unsigned short':       ('unsigned short', 'H', 2, True, False, False, False),
            'int':                  ('int', 'i', 4, True, False, False, True),
            'unsigned int':         ('unsigned int', 'I', 4, True, False, False, False),
            'long':                 ('long', 'l', 4, True, False, False, True),
            'unsigned long':        ('unsigned long', 'L', 4, True, False, False, False),
            'long long':            ('long long', 'q', 8, True, False, False, True),
            'unsigned long long':   ('unsigned long long', 'Q', 8, True, False, False, False),
            'float':                ('float', 'f', 4, False, True, False, True),
            'double':               ('double', 'd', 8, False, True, False, True)
        }
    def __init__(self, cppname, structfmt, size, isint, isfloat, ischar, issigned):
        _CppType.__init__(self)
        self.cppname = cppname
        self.size = size
        self.size_expr = 'sizeof(' + cppname + ')'
        self.structfmt = structfmt
        self.isint = isint
        self.isfloat = isfloat
        self.ischar = ischar
        self.issigned = issigned
    
    @staticmethod
    def get_from_cppname(cppname):
        return _CppPrimitiveType(*_CppPrimitiveType._bycppname[cppname])
    
    def getfreevars(self, glbls):
        return set()
    
    def getpytype(self):
        if self.ischar:
            return str
        elif self.isint:
            return int
        elif self.isfloat:
            return float
    
    def __str__(self):
        return self.cppname
    
    def formatdata(self, data):
        return [1], _pystruct.pack(self.structfmt, data)
    
    def get_cdecl_stmt(self, param_name):
        return '{} {};'.format(self.cppname, param_name)
    
    def get_cread_stmt(self, param_name, istream_name, dims):
        return '{}.read((const char*)&{}, {});'.format(istream_name, param_name, self.size_expr)
    
    def get_cwrite_stmt(self, param_name, ostream_name, dims):
        return '{}.write((const char*)&{}, {});'.format(ostream_name, param_name, self.size_expr)


class _CppVoidType(_CppType):
    def __init__(self):
        self.cppname = 'void'
    
    def getfreevars(self, glbls):
        return set()
    
    def getpytype(self):
        return type(None)
    
    def __str__(self):
        return 'void'


class _CppArrayType(_CppType):
    def __init__(self, basetype, dims=[None]):
        _CppType.__init__(self)
        self.basetype = basetype
        self.dimensions = dims
    
    def getfreevars(self, glbls):
        freevars = self.basetype.getfreevars(glbls)
        for fv in iter(d.getfreevars(glbls) for d in self.dimensions if hasattr(d, 'getfreevars')):
            freevars = freevars | fv
        return freevars
    
    def getpytype(self):
        return _pyarraytype(self.dimensions, self.basetype.getpytype())
    
    def __str__(self):
        return '{}[{}]'.format(str(self.basetype), ']['.join(map(str,self.dimensions)))
    
    def statictype(self, bindings):
        dim_list = list()
        for dim in self.dimensions:
            if dim is None:
                dim_list.append(None)
            else:
                dim_list.append(_evalexpr(dim, _CppPrimitiveType.get_from_cppname('int'), bindings))
        return _CppArrayType(self.basetype.statictype(bindings), dim_list)
    
    def _formatdata_array(self, unit_length, data):
        read_length = 0
        if _chill_util.python_version_major == 2:
            read_data = ''
        else:
            read_data = bytes()
        while read_length < len(data):
            for i in range(unit_length):
                _, b = self.basetype.formatdata(data[read_length+i])
                read_data += b
            read_length += unit_length
        return read_data
    
    def formatdata(self, data):
        prod = lambda l: _pyfunctools.reduce(lambda a,v: a*v, l, 1)
        if self.dimensions[0] is None:
            return self.dimensions, self._formatdata_array(prod(self.dimensions[1:]), data)
        else:
            return self.dimensions, self._formatdata_array(prod(self.dimensions), data)
    
    def get_cdecl_stmt(self, param_name):
        return '{} {}[{}];'.format(str(self.basetype), param_name, ']['.join(map(str,self.dimensions)))
    
    def get_cread_stmt(self, param_name, istream_name, dims):
        length = _pyfunctools.reduce(lambda a,v: a*v, self.dimensions)
        #TODO: use dims
        if isinstance(self.basetype, _CppPrimitiveType):
            size_expr = '{}*{}'.format(length, self.basetype.size_expr)
            return '{}.read((char*){}, {});'.format(istream_name, param_name, size_expr)
        else:
            raise NotImplementedError
    
    def get_cwrite_stmt(self, param_name, ostream_name, dims):
        length = _pyfunctools.reduce(lambda a,v: a*v, self.dimensions)
        #TODO: use dims
        if isinstance(self.basetype, _CppPrimitiveType):
            size_expr = '{}*{}'.format(length, self.basetype.size_expr)
            return '{}.write((char*){}, {});'.format(ostream_name, param_name, size_expr)
        else:
            raise NotImplementedError


class _CppPointerType(_CppType):
    def __init__(self, basetype):
        _CppType.__init__(self)
        self.basetype = basetype
    
    def getfreevars(self, glbls):
        return self.basetype.getfreevars(glbls)
    
    def getpytype(self):
        return self.basetype.getpytype()
    
    def __str__(self):
        return '{}*'.format(str(self.basetype))
    
    def statictype(self, bindings):
        return _CppPointerType(self.basetype.statictype(bindings))
    
    def formatdata(self, data):
        if isinstance(data, list):
            if _chill_util.python_version_major == 2:
                read_data = ''
            else:
                read_data = bytes()
            for data_item in data:
                next_dims, b = self.basetype.formatdata(data_item)
                read_data += b
            return [len(data)] + next_dims, read_data
        else:
            dims, fmt_data = self.basetype.formatdata(data)
            return [1] + dims, fmt_data


class _CppReferenceType(_CppType):
    def __init__(self, basetype):
        _CppType.__init__(self)
        self.basetype = basetype
    
    def getfreevars(self, glbls):
        return self.basetype.getfreevars(glbls)
    
    def getpytype(self):
        return self.basetype.getpytype()
    
    def __str__(self):
        return '{}&'.format(str(self.basetype))
    
    def statictype(self, bindings):
        return _CppReferenceType(self.basetype.statictype(bindings))
    
    def formatdata(self, data):
        dims, fmt_data = self.basetype.formatdata(data)
        return dims, fmt_data


class _Parameter(_TreeNode):
    def __init__(self, name, cpptype, direction, init_expr=None):
        self.name = name
        self.direction = direction
        self.cpptype = cpptype
        self.init_expr = init_expr
        self._generated = None
    
    @staticmethod
    def order_by_freevars(param_list, glbls=set()):
        defined_names = set()
        parameter_names = set(p.name for p in param_list)
        param_queue = _pycollections.deque(param_list)
        while len(param_queue):
            param = param_queue.popleft()
            freevars = (parameter_names & param.getfreevars(glbls)) - defined_names
            if not len(freevars):
                defined_names.add(param.name)
                yield param
            else:
                param_queue.append(param)
    
    def getfreevars(self, glbls=set()):
        freevars = set()
        if self.init_expr is not None:
            freevars = freevars | self.init_expr.getfreevars(glbls)
        freevars = freevars | self.cpptype.getfreevars(glbls)
        return freevars
    
    def generatedata(self, bindings=dict()):
        if self._generated is None:
            if self.init_expr is None:
                py_data = None
            else:
                py_data = _evalexpr(self.init_expr, self.cpptype, bindings)
            static_type = self.cpptype.statictype(bindings)
            dims, data = static_type.formatdata(py_data)
            self._generated = (self.name, static_type, dims, data)
            return self.name, static_type, dims, data
        else:
            return self._generated


class _Procedure(_TreeNode):
    def __init__(self, name, rtype, parameters):
        self.name = name
        self.rtype = rtype
        self.parameters = parameters
        self.binding_stack = []
        self._bindings = None
        self._params_orderd = None
        self._invoke_str = '{}({});'.format(self.name, ','.join([p.name for p in parameters]))
    
    def _order_params(self):
        if not self._params_orderd:
            self._params_orderd = list(_Parameter.order_by_freevars(self.parameters))
    
    def _compute_bindings(self, global_bindings):
        local_bindings = dict(global_bindings)
        if self._bindings is None:
            new_bindings = dict()
            for binding_frame in self.binding_stack:
                for name, (ctype, expr) in binding_frame.items():
                    value = _evalexpr(expr, ctype, local_bindings)
                    new_bindings[name] = value
                    local_bindings[name] = value
            self._bindings = new_bindings
        local_bindings.update(self._bindings)
        return local_bindings
    
    def generatedata(self, direction_list, global_bindings=None):
        self._order_params()
        if global_bindings is None:
            global_bindings = dict()
        bindings = self._compute_bindings(global_bindings)
        for param in (p for p in self._params_orderd if p.direction in direction_list):
            p_name, p_statictype, p_dims, p_data = param.generatedata(bindings)
            #TODO: add binding
            yield p_name, p_statictype, p_dims, p_data
    
    def generatedecls(self, bindings):
        for p_name, p_statictype, p_dims, p_data in self.generatedata(['in','out','inout'], bindings):
            yield p_statictype.get_cdecl_stmt(p_name)
        #for p_name, p_statictype, p_dims, p_data in self.generatedata('out', bindings):
        #    yield p_statictype.get_cdecl_stmt(p_name)
    
    def generatereads(self, direction_list, stream, bindings):
        for p_name, p_statictype, p_dims, p_data in self.generatedata(direction_list, bindings):
            yield p_statictype.get_cread_stmt(p_name, stream, p_dims)
    
    def generatewrites(self, stream, bindings):
        for p_name, p_statictype, p_dims, p_data in self.generatedata(['inout', 'out'], bindings):
            yield p_statictype.get_cwrite_stmt(p_name, stream, p_dims)
    
    def getinvokestr(self):
        return self._invoke_str


class _Expr(_TreeNode):
    def __init__(self):
        pass
    
    def getfreevars(self, glbls):
        raise NotImplementedError
    
    def compile_to_lambda(self, glbls, target_type):
        args = _pyast.arguments(list(_pyast.Name(n, _pyast.Param()) for n in self.getfreevars(self, glbls)), None, None, [])
        expr = _pyast.Expression(_pyast.Lambda(args, self.compile_expr(target_type)))
        expr = _pyast.fix_missing_locations(expr)
        return eval(compile(expr, '<string>', 'eval'))
    
    def compile_expr(self, target_type):
        raise NotImplementedError


class _ConstantExpr(_Expr):
    def __init__(self, value):
        self.value = value
    
    def compile_expr(self, target_type):
        if target_type is None:
            return _pyast.parse(self.value, '<string>', 'eval').body
        elif target_type == chr:
            return _pyast.Str(chr(self.value))
        elif target_type == int:
            return _pyast.Num(int(self.value))
        elif target_type == str:
            return _pyast.Str(str(self.value))
        elif target_type == float:
            return _pyast.Num(float(self.value))
    
    def getfreevars(self, glbls):
        return set()
    
    def __str__(self):
        return self.value


class _NameExpr(_Expr):
    def __init__(self, name):
        self.name = name
    
    def compile_expr(self, target_type):
        return _pyast.Name(self.name, _pyast.Load())
    
    def getfreevars(self, glbls):
        if self.name not in glbls:
            return set([self.name])
        else:
            return set()
    
    def __str__(self):
        return self.name


class _AttributeExpr(_Expr):
    def __init__(self, expr, name):
        self.expr = expr
        self.name = name
    
    def compile_expr(self, target_type):
        return _pyast.Attribute(
            self.expr.compile_expr(None),
            self.name,
            _pyast.Load())
    
    def getfreevars(self, glbls):
        return self.expr.getfreevars(glbls)
    
    def __str__(self):
        return '{}.{}'.format(str(self.expr), self.name)


class _BinExpr(_Expr):
    _optypes = {
            '+':  _pyast.Add,
            '-':  _pyast.Sub,
            '*':  _pyast.Mult,
            '**': _pyast.Pow,
            '/':  _pyast.Div
        }
    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.op = op
    
    def compile_expr(self, target_type):
        return _pyast.BinOp(
                self.left.compile_expr(target_type),
                _BinExpr._optypes[self.op](),
                self.right.compile_expr(target_type))
    
    def getfreevars(self, glbls):
        return self.left.getfreevars(glbls) | self.right.getfreevars(glbls)
    
    def __str__(self):
        return '({}{}{})'.format(str(self.left),self.op,str(self.right))


class _UnaryExpr(_Expr):
    _optypes = {
            '-': _pyast.USub
        }
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr
    
    def compile_expr(self, target_type):
        return _pyast.UnaryOp(
                _UnaryExpr._optypes[self.op](),
                self.expr.compile_expr(target_type))
    
    def getfreevars(self, glbls):
        return self.expr.getfreevars(glbls)
    
    def __str__(self):
        return '({}{})'.format(self.op, str(self.expr))


class _LambdaExpr(_Expr):
    def __init__(self, params, expr):
        self.params = params
        self.expr = expr
    
    def compile_expr(self, target_type):
        if target_type is None:
            exprtype = None
        else:
            assert hasattr(target_type, 'paramtypes')
            assert hasattr(target_type, 'exprtype')
            exprtype = target_type.exprtype
        if _chill_util.python_version_major == 2:
            return _pyast.Lambda(
                _pyast.arguments([_pyast.Name(p, _pyast.Param()) for p in self.params], None, None, []),
                self.expr.compile_expr(exprtype))
        else:
            return _pyast.Lambda(
                _pyast.arguments([_pyast.arg(p, None) for p in self.params], None, None, [], None, None, [], []),
                self.expr.compile_expr(exprtype))
    
    def getfreevars(self, glbls):
        new_glbls = set(glbls)
        new_glbls = new_glbls | set(self.params)
        return self.expr.getfreevars(new_glbls)
    
    def __str__(self):
        return 'lambda {}:{}'.format(','.join(map(str,self.params)), str(self.expr))


class _InvokeExpr(_Expr):
    def __init__(self, func, parameters):
        self.func = func
        self.parameters = parameters
    
    def compile_expr(self, target_type):
        if target_type is None:
            lt = None
        else:
            lt = _pylambdatype([None for p in self.parameters], target_type)
        return _pyast.Call(
                self.func.compile_expr(lt),
                [p.compile_expr(None) for p in self.parameters],
                [],
                None,
                None)
    
    def getfreevars(self, glbls):
        return set(
            self.func.getfreevars(glbls) |
            _pyfunctools.reduce(lambda a,v: a | v.getfreevars(glbls), self.parameters, set()))
    
    def __str__(self):
        return '{}({})'.format(str(self.func),','.join(map(str,self.parameters)))


class _Generator(_Expr):
    def __init__(self):
        _Expr.__init__(self)
    
    
class _MatrixGenerator(_Generator):
    def __init__(self, dims, genexpr):
        self.dimensions = dims
        self.genexpr = genexpr
    
    def _compile_dims(self, target_type):
        if hasattr(target_type, 'dimensions'):
            dim_exprs = list()
            assert len(target_type.dimensions) == len(self.dimensions)
            for i, d in enumerate(target_type.dimensions):
                if d is None:
                    d = self.dimensions[i]
                dim_exprs += [d.compile_expr(int)]
        else:
            dim_exprs = [d.compile_expr(int) for d in self.dimensions]
        return _pyast.List(dim_exprs, _pyast.Load())
    
    def _lambda_type(self, target_type):
        if hasattr(target_type, 'dimensions'):
            return _pylambdatype([int for d in target_type.dimensions], target_type.basetype)
        else:
            return _pylambdatype([int for d in self.dimensions], target_type)
    
    def compile_expr(self, target_type):
        assert target_type is not None
        dims = self._compile_dims(target_type)
        ltype = self._lambda_type(target_type)
        
        #def array(func,dims):
        #    return [func(*d) for d in itertools.product(*(map(range,dims))]
        elt_expr = _pyast.Call(self.genexpr.compile_expr(ltype), [], [], _pyast.Name('_d', _pyast.Load()), None)                  # func(*d)
        # elt_expr = _pyast.Call(_pyast.Name('tuple', _pyast.Load()), [_pyast.Name('_d', _pyast.Load()), elt_expr], [], None, None) # tuple(d, func(*d))
        pdt_expr = _pyast.Attribute(_pyast.Name('_pyitertools', _pyast.Load()), 'product', _pyast.Load())                            # itertools.product
        itr_expr = _pyast.Call(_pyast.Name('map', _pyast.Load()), [_pyast.Name('range', _pyast.Load()), dims], [], None, None)    # map(range,dims)
        itr_expr = _pyast.Call(pdt_expr, [], [], itr_expr, None)                                                                  # itertools.product(*(map(range,dims)))
        return _pyast.ListComp(
            elt_expr,
            [_pyast.comprehension(_pyast.Name('_d', _pyast.Store()), itr_expr, [])])
    
    def getfreevars(self, glbls):
        return set(
            self.genexpr.getfreevars(glbls) |
            _pyfunctools.reduce(lambda a,v: a | v.getfreevars(glbls), filter(lambda x: x is not None, self.dimensions), set()))
    
    def __str__(self):
        return 'matrix([{}],{})'.format(','.join(map(str,self.dimensions)),str(self.genexpr))


class _RandomExpr(_Expr):
    def __init__(self, minexpr, maxexpr):
        self.minexpr = minexpr
        self.maxexpr = maxexpr
        self.expr = _BinExpr(
            _BinExpr(
                _InvokeExpr(_AttributeExpr(_NameExpr('_pyrandom'),'random'),[]),
                '*',
                _BinExpr(maxexpr, '-', minexpr)),
            '+',
            minexpr)
    
    def getfreevars(self, glbls):
        return self.minexpr.getfreevars(glbls) | self.maxexpr.getfreevars(glbls)
    
    def compile_expr(self, target_type):
        if target_type == int:
            return _pyast.Call(_pyast.Name('int', _pyast.Load()),[self.expr.compile_expr(float)],[],None,None)
        elif target_type == float:
            return self.expr.compile_expr(target_type)
        elif target_type is None:
            return self.expr.compile_expr(None)
        assert False
    
    def __str__(self):
        return 'random({},{})'.format(str(self.minexpr),str(self.maxexpr))


### What to import from * ###
addbindings = _addbindings

CppType = _CppType
CppPrimitiveType = _CppPrimitiveType
CppVoidType = _CppVoidType
CppArrayType = _CppArrayType
CppPointerType = _CppPointerType

ConstantExpr = _ConstantExpr
NameExpr = _NameExpr
AttributeExpr = _AttributeExpr
BinExpr = _BinExpr
UnaryExpr = _UnaryExpr
LambdaExpr = _LambdaExpr
InvokeExpr = _InvokeExpr
MatrixGenerator = _MatrixGenerator
RandomExpr = _RandomExpr

Procedure = _Procedure
Parameter = _Parameter

