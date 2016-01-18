import ast
import functools
import itertools
import pylang.debug
import random
import struct
import unittest

import testchill
import testchill._cpp_validate_env as validate_env
import testchill.cpp_validate
import testchill.util

## Support functions ##
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

def _compile_and_run(expr, target_type, bindings):
    t = ast.fix_missing_locations(ast.Expression(expr.compile_expr(target_type)))
    return eval(compile(t, '<string>', 'eval'), bindings)

def _compile_and_invoke(expr, target_type, bindings, args):
    t = ast.fix_missing_locations(ast.Expression(expr.compile_expr(target_type)))
    return (eval(compile(t, '<string>', 'eval'), bindings))(*args)

def _expr_test(tc, expr, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value):
    freevars = expr.getfreevars(fv_bindings)
    value = _compile_and_run(expr, target_type, rt_bindings)
    tc.assertEqual(exp_freevars, freevars)
    tc.assertEqual(exp_value, value)
    tc.assertEqual(target_type, type(value))

def _expr_test_list(tc, expr, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value):
    freevars = expr.getfreevars(fv_bindings)
    value = _compile_and_run(expr, target_type, rt_bindings)
    tc.assertEqual(exp_freevars, freevars)
    tc.assertEqual(exp_value, value)
    tc.assertEqual(list, type(value))

def _expr_test_invoke(tc, expr, fv_bindings, rt_bindings, target_type, exp_freevars, invoke_args, exp_value):
    freevars = expr.getfreevars(fv_bindings)
    value = _compile_and_invoke(expr, target_type, rt_bindings, invoke_args)
    tc.assertEqual(exp_freevars, freevars)
    tc.assertEqual(exp_value, value)
    tc.assertEqual(target_type.exprtype, type(value))

def lambdatype(param_types, etype):
    return validate_env._pylambdatype(param_types, etype)

def arraytype(dims, etype):
    return validate_env._pyarraytype(dims, etype)


## Test case class ##
class Test_CppValidateEnv(unittest.TestCase):
    def setUp(self):
        ### data for the abstract syntax tree ###
        _const_4 = validate_env._ConstantExpr('4')
        _const_3 = validate_env._ConstantExpr('3')
        _const_2 = validate_env._ConstantExpr('2')
        _const_0 = validate_env._ConstantExpr('0')
        _name_x = validate_env._NameExpr('x')
        _name_y = validate_env._NameExpr('y')
        _name_p = validate_env._NameExpr('p')
        _name_pow = validate_env._NameExpr('pow')
        _attr_px = validate_env._AttributeExpr(_name_p, 'x')
        _attr_py = validate_env._AttributeExpr(_name_p, 'y')
        _add_3_2 = validate_env._BinExpr(_const_3, '+', _const_2)
        _add_x_2 = validate_env._BinExpr(_name_x, '+', _const_2)
        _pow_x_2 = validate_env._BinExpr(_name_x, '**', _const_2)
        
        _name_i = validate_env._NameExpr('i')
        _lambda_i = validate_env._LambdaExpr(['i'],_name_i)
        
        _name_j = validate_env._NameExpr('j')
        _const_10 = validate_env._ConstantExpr('10')
        _mul_i_10 = validate_env._BinExpr(_name_i, '*', _const_10)
        _add_mul_i_10_j = validate_env._BinExpr(_mul_i_10, '+', _name_j)
        _lambda_ij = validate_env._LambdaExpr(['i','j'],_add_mul_i_10_j)
        
        self._ConstantExpr_test_data = [
                (('3',), set(), dict(), int, set(), int(3)),
                (('3',), set(), dict(), float, set(), float(3))
            ]
        self._NameExpr_test_data = [
                (('x',), set(), {'x':3}, int, {'x'}, int(3)),
                (('x',), {'x'}, {'x':3}, int, set(), int(3))
            ]
        self._AttributeExpr_test_data = [
                ((validate_env._NameExpr('p'),'x'), set(), {'p':Point(3,0)}, int, {'p'}, int(3)),
                ((validate_env._NameExpr('p'),'x'), {'p'}, {'p':Point(3,0)}, int, set(), int(3))
            ]
        self._BinExpr_test_data = [
                ((_const_3, '+', _const_2), set(), dict(), int, set(), int(5)),
                ((_const_3, '+', _const_2), set(), dict(), float, set(), float(5)),
                ((_name_x, '+', _const_2), set(), {'x':3}, int, {'x'}, int(5)),
                ((_name_x, '+', _const_2), {'x'}, {'x':3}, int, set(), int(5)),
                ((_const_3, '+', _name_x), set(), {'x':2}, int, {'x'}, int(5)),
                ((_const_3, '+', _name_x), {'x'}, {'x':2}, int, set(), int(5)),
                ((_const_3, '-', _const_2), set(), dict(), int, set(), int(1)),
                ((_const_3, '*', _const_2), set(), dict(), int, set(), int(6)),
                ((_const_3, '/', _const_2), set(), dict(), int, set(), int(1)),
                ((_const_3, '**', _const_2), set(), dict(), int, set(), int(9))
            ]
        self._UnaryExpr_test_data = [
                (('-', _const_3), set(), dict(), int, set(), int(-3)),
                (('-', _add_3_2), set(), dict(), int, set(), int(-5)),
                (('-', _add_x_2), set(), {'x':3}, int, {'x'}, int(-5)),
                (('-', _add_x_2), {'x'}, {'x':3}, int, set(), int(-5))
            ]
        self._LambdaExpr_test_data = [
                (([],_const_3), set(), dict(), lambdatype([],int), set(), tuple(), int(3)),
                (([],_name_x), set(), {'x':3}, lambdatype([],int), {'x'}, tuple(), int(3)),
                ((['x'],_pow_x_2), set(), dict(), lambdatype([int],int), set(), (int(4),), int(16))
            ]
        self._InvokeExpr_test_data = [
                ((_name_pow,[_const_3, _const_2]), set(), dict(), int, {'pow'}, int(9)),
            ]
        self._MatrixGenerator_test_data = [
                (([_const_2],_lambda_i), set(), {'_pyitertools': itertools}, arraytype([None],int), set(), [0, 1]),
                (([None],_lambda_i), set(), {'_pyitertools': itertools}, arraytype([_const_2],int), set(), [0, 1]),
                (([_const_2,_const_3],_lambda_ij), set(), {'_pyitertools': itertools}, arraytype([_const_2,_const_3], int), set(), [0, 1, 2, 10, 11, 12]),
                (([_const_2,_const_3],_lambda_ij), set(), {'_pyitertools': itertools}, arraytype([None,None], int), set(), [0, 1, 2, 10, 11, 12]),
                (([_const_2,None],_lambda_ij), set(), {'_pyitertools': itertools}, arraytype([None,_const_3], int), set(), [0, 1, 2, 10, 11, 12]),
                (([None,_const_3],_lambda_ij), set(), {'_pyitertools': itertools}, arraytype([_const_2,None], int), set(), [0, 1, 2, 10, 11, 12]),
                (([None,None],_lambda_ij), set(), {'_pyitertools': itertools}, arraytype([_const_2,_const_3], int), set(), [0, 1, 2, 10, 11, 12]),
                (([_name_x],_lambda_i), set(), {'_pyitertools': itertools, 'x':2}, arraytype([None],int), {'x'}, [0, 1]),
                (([None],_lambda_i), set(), {'_pyitertools': itertools, 'x':2}, arraytype([_name_x],int), set(), [0, 1]),
            ]
        self._RandomExpr_test_state = random.getstate()
        self._RandomExpr_test_data = [
                ((_const_0,_const_4), set(), {'_pyrandom': random}, int, set(), int(random.random()*4)),
                ((_const_0,_name_x), set(), {'_pyrandom': random, 'x':4}, int, {'x'}, int(random.random()*4)),
                ((_name_x,_const_4), set(), {'_pyrandom': random, 'x':0}, int, {'x'}, int(random.random()*4)),
            ]
        ### data for data generating ###
        _name_ambn = validate_env._NameExpr('ambn')
        _name_an   = validate_env._NameExpr('an')
        _name_bm   = validate_env._NameExpr('bm')
        _name_even2 = validate_env._NameExpr('evendist2')
        _lambda_ij_0 = validate_env._LambdaExpr(['i','j'],_const_0)
        _matrix_2_an_ambn_even2 = validate_env._MatrixGenerator([_name_an,_name_ambn],_name_even2)
        _matrix_2_ambn_bm_even2 = validate_env._MatrixGenerator([_name_ambn,_name_bm],_name_even2)
        _matrix_2_an_bm_lambda_ij_0 = validate_env._MatrixGenerator([_name_an,_name_bm],_lambda_ij_0)
        _add_an_bm = validate_env._BinExpr(_name_an, '+', _name_bm)
        _int_type = validate_env._CppPrimitiveType.get_from_cppname('int')
        _float_type = validate_env._CppPrimitiveType.get_from_cppname('float')
        _float_ptr_type = validate_env._CppPointerType(_float_type)
        _param_A    = validate_env._Parameter('A',   _float_ptr_type,'in', _matrix_2_an_ambn_even2)
        _param_B    = validate_env._Parameter('B',   _float_ptr_type,'in', _matrix_2_ambn_bm_even2)
        _param_C    = validate_env._Parameter('C',   _float_ptr_type,'out',_matrix_2_an_bm_lambda_ij_0)
        _param_ambn = validate_env._Parameter('ambn',_int_type,      'in', _add_an_bm)
        _param_an   = validate_env._Parameter('an',  _int_type,      'in', _const_2)
        _param_bm   = validate_env._Parameter('bm',  _int_type,      'in', _const_3)
        self._Parameter_order_by_freevars_test_data = [
                ([_param_A, _param_B, _param_C, _param_ambn, _param_an, _param_bm], ['an','bm','C','ambn','A','B'])
            ]
        _float_3_type = validate_env._CppArrayType(_float_type, [_const_3])
        _float_3_2_type = validate_env._CppArrayType(_float_type, [_const_3,_const_2])
        _name_N = validate_env._NameExpr('N')
        _float_N_type = validate_env._CppArrayType(_float_type, [_name_N])
        _float_N_2_type = validate_env._CppArrayType(_float_type, [_name_N,_const_2])
        self._CppType_statictype_test_data = [
                ((_int_type, dict()), 'int'),
                ((_float_ptr_type, dict()), 'float*'),
                ((_float_3_type, dict()), 'float[3]'),
                ((_float_N_type, {'N': 3}), 'float[3]'),
                ((_float_N_2_type, {'N': 3}), 'float[3][2]')
            ]
        _int_ptr_type = validate_env._CppPointerType(_int_type)
        _int_ptr_ptr_type = validate_env._CppPointerType(_int_ptr_type)
        _int_3_type = validate_env._CppArrayType(_int_type, [_const_3])
        _int_N_type = validate_env._CppArrayType(_int_type, [_name_N])
        _int_3_2_type = validate_env._CppArrayType(_int_type, [_const_3, _const_2])
        joinbytes = lambda b: functools.reduce(lambda a,v: a+v,b)
        self._CppType_formatdata_test_data = [
                ((_int_type,         dict(), 3),                 ([1],     struct.pack('i',3))),
                ((_float_type,       dict(), float(3)),          ([1],     struct.pack('f',float(3)))),
                ((_int_3_type,       dict(), list(range(3))),    ([3],     joinbytes([struct.pack('i',i) for i in range(3)]))),
                ((_int_3_2_type,     dict(), list(range(6))),    ([3,2],   joinbytes([struct.pack('i',i) for i in range(6)]))),
                ((_int_ptr_type,     dict(), 3),                 ([1,1],   struct.pack('i',3))),
                ((_int_ptr_type,     dict(), list(range(3))),    ([3,1],   joinbytes([struct.pack('i',i) for i in range(3)]))),
                ((_int_ptr_ptr_type, dict(), list(range(3))),    ([3,1,1], joinbytes([struct.pack('i',i) for i in range(3)]))),
                ((_int_ptr_ptr_type, dict(), [[0,1,2],[3,4,5]]), ([2,3,1], joinbytes([struct.pack('i',i) for i in range(6)]))),
            ]
        evendist2 = lambda i,j: random.random()
        random.seed(0)
        self._Parameter_generatedata_test_state = random.getstate()
        if testchill.util.python_version_major == 2:
            self._Parameter_generatedata_test_data = [
                    ((_param_A,    {'an':2, 'ambn':5,'evendist2':evendist2}), ('A', 'float*', [10, 1], '\x08,X?M\tB?)U\xd7>\xbc\x90\x84>\xe6\xe2\x02?\x87S\xcf>\x06\xa7H?\xceK\x9b>\x84\x04\xf4>\x86X\x15?')),
                    ((_param_B,    {'ambn':5, 'bm':3,'evendist2':evendist2}), ('B', 'float*', [15, 1], '\x16zh?(3\x01?\rM\x90>b|A?nM\x1e?^B\x80>!\xe5h?\xd4\x97{?fjO?Y\xf4f?\xaa\xcb\x9e>A\xd6:?D\x1af?\x92\x19/?\xb1\xbc\xf1>')),
                    ((_param_C,    {'an':2, 'bm':3, 'evendist2':evendist2}),  ('C', 'float*', [6, 1], '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')),
                    ((_param_ambn, {'an':2, 'bm':3}),                         ('ambn', 'int', [1], '\x05\x00\x00\x00')),
                    ((_param_an,   dict()),                                   ('an', 'int', [1], '\x02\x00\x00\x00')),
                    ((_param_bm,   dict()),                                   ('bm', 'int', [1], '\x03\x00\x00\x00'))
                ]
        else:
            self._Parameter_generatedata_test_data = [
                    ((_param_A,    {'an':2, 'ambn':5,'evendist2':evendist2}), ('A', 'float*', [10, 1], b'\x08,X?M\tB?)U\xd7>\xbc\x90\x84>\xe6\xe2\x02?\x87S\xcf>\x06\xa7H?\xceK\x9b>\x84\x04\xf4>\x86X\x15?')),
                    ((_param_B,    {'ambn':5, 'bm':3,'evendist2':evendist2}), ('B', 'float*', [15, 1], b'\x16zh?(3\x01?\rM\x90>b|A?nM\x1e?^B\x80>!\xe5h?\xd4\x97{?fjO?Y\xf4f?\xaa\xcb\x9e>A\xd6:?D\x1af?\x92\x19/?\xb1\xbc\xf1>')),
                    ((_param_C,    {'an':2, 'bm':3, 'evendist2':evendist2}),  ('C', 'float*', [6, 1], b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')),
                    ((_param_ambn, {'an':2, 'bm':3}),                         ('ambn', 'int', [1], b'\x05\x00\x00\x00')),
                    ((_param_an,   dict()),                                   ('an', 'int', [1], b'\x02\x00\x00\x00')),
                    ((_param_bm,   dict()),                                   ('bm', 'int', [1], b'\x03\x00\x00\x00'))
                ]
        ### data for parsing ###
        self.parse_procedure_test_data = [
                (('procedure void q()',), ('void', 'q', 0)),
                (('procedure int q()',), ('int', 'q', 0)),
                (('procedure float q()',), ('float', 'q', 0)),
                (('procedure unsigned int q()',), ('unsigned int', 'q', 0)),
                (('procedure void q(in int x)',), ('void', 'q', 1)),
                (('procedure void q(in int x, in int y)',), ('void', 'q', 2)),
            ]
        _mm_proc_expr = '''
            procedure void mm(
                in  float* A    = matrix([an,ambn],evendist2),
                in  float* B    = matrix([ambn,bm],evendist2),
                out float* C    = matrix([an,bm],lambda i,j: 0),
                in  int    ambn = an + bm,
                in  int    an   = 2,
                in  int    bm   = 3)
            '''
        self.parse_parameter_test_data = [
                (('procedure void mm(in int x)',), [(0, 'x', 'int', 'in', False, set())]),
                (('procedure void mm(out int* x = 10)',), [(0, 'x', 'int*', 'out', True, set())]),
                ((_mm_proc_expr,),[
                        (0, 'A', 'float*', 'in', True, set(['an','ambn','evendist2'])),
                        (1, 'B', 'float*', 'in', True, set(['ambn','bm','evendist2'])),
                        (2, 'C', 'float*', 'out', True, set(['an','bm'])),
                        (3, 'ambn', 'int', 'in', True, set(['an','bm'])),
                        (4, 'an', 'int', 'in', True, set([])),
                        (5, 'bm', 'int', 'in', True, set([]))
                    ]),
            ]
        ### data for code generation ###
        _float_2d_type = validate_env._CppArrayType(_float_type, [_name_an,_name_ambn])
        self._CppType_statictype_test_data = [
                ((_float_2d_type, {'an':2,'ambn':5}), 'float[2][5]')
            ]
        self._CppType_get_cdecl_stmt_test_data = [
                ((_float_2d_type, 'A', {'an':2,'ambn':5}), 'float A[2][5];')
            ]
        self._CppType_get_cread_stmt_test_data = [
                ((_float_2d_type, 'A', {'an':2,'ambn':5}, 'datafile_initialize', [10,1]), 'datafile_initialize.read((char*)A, 10*sizeof(float));')
            ]
        self._CppType_get_cwrite_stmt_test_data = [
                ((_float_2d_type, 'A', {'an':2,'ambn':5}, 'datafile_out', [10,1]), 'datafile_out.write((char*)A, 10*sizeof(float));')
            ]
    
    def run_expr_test_data(self, ctor, test_data):
        for ctor_args, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value in test_data:
            expr = ctor(*ctor_args)
            _expr_test(self, expr, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value)
    
    def run_expr_test_data_list(self, ctor, test_data):
        for ctor_args, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value in test_data:
            expr = ctor(*ctor_args)
            _expr_test_list(self, expr, fv_bindings, rt_bindings, target_type, exp_freevars, exp_value)
    
    def run_expr_test_data_invoke(self, ctor, test_data):
        for ctor_args, fv_bindings, rt_bindings, target_type, exp_freevars, invoke_args, exp_value in test_data:
            expr = ctor(*ctor_args)
            _expr_test_invoke(self, expr, fv_bindings, rt_bindings, target_type, exp_freevars, invoke_args, exp_value)
    
    def test__ConstantExpr(self):
        self.run_expr_test_data(validate_env._ConstantExpr, self._ConstantExpr_test_data)
    
    def test__NameExpr(self):
        self.run_expr_test_data(validate_env._NameExpr, self._NameExpr_test_data)
    
    def test__AttributeExpr(self):
        self.run_expr_test_data(validate_env._AttributeExpr, self._AttributeExpr_test_data)
    
    def test__UnaryExpr(self):
        self.run_expr_test_data(validate_env._UnaryExpr, self._UnaryExpr_test_data)
    
    def test__LambdaExpr(self):
        self.run_expr_test_data_invoke(validate_env._LambdaExpr, self._LambdaExpr_test_data)
    
    def test__InvokeExpr(self):
        self.run_expr_test_data(validate_env._InvokeExpr, self._InvokeExpr_test_data)
    
    def test__MatrixGenerator(self):
        self.run_expr_test_data_list(validate_env._MatrixGenerator, self._MatrixGenerator_test_data)
    
    def test__RandomExpr(self):
        random.setstate(self._RandomExpr_test_state)
        self.run_expr_test_data(validate_env._RandomExpr, self._RandomExpr_test_data)
    
    def test_parse_procedure(self):
        parse_func = testchill.cpp_validate._parse_testproc_script
        for args, expected in self.parse_procedure_test_data:
            rtype_exp, name_exp, param_count_exp = expected
            proc = parse_func(*args)
            self.assertEqual(str(proc.rtype), rtype_exp)
            self.assertEqual(proc.name, name_exp)
            self.assertEqual(len(proc.parameters), param_count_exp)
    
    def test_parse_parameter(self):
        #pylang.debug.enable(['pylang.parser.BaseTextParser.parse'])
        parse_func = testchill.cpp_validate._parse_testproc_script
        for args, expected in self.parse_parameter_test_data:
            proc = parse_func(*args)
            for param_exp in expected:
                index, name_exp, ctype_exp, direction_exp, has_init_exp, freevars_exp = param_exp
                param = proc.parameters[index]
                self.assertEqual(param.name, name_exp)
                self.assertEqual(str(param.cpptype), ctype_exp)
                self.assertEqual(param.direction, direction_exp)
                self.assertEqual(param.init_expr is not None, has_init_exp)
                self.assertEqual(param.getfreevars(), freevars_exp)
        #pylang.debug.enable(['pylang.parser.BaseTextParser.parse'], False)
    
    def test__Parameter_order_by_freevars(self):
        def testfunc(param_list):
            return [p.name for p in validate_env._Parameter.order_by_freevars(param_list)]
        for arg, expected in self._Parameter_order_by_freevars_test_data:
            self.assertEqual(testfunc(arg),expected)
    
    def test__CppType_statictype(self):
        def testfunc(ctype, glbls):
            return str(ctype.statictype(glbls))
        for args, expected in self._CppType_statictype_test_data:
            self.assertEqual(testfunc(*args), expected)
    
    def test__CppType_formatdata(self):
        def testfunc(ctype, glbls, data):
            return ctype.statictype(glbls).formatdata(data)
        for args, expected in self._CppType_formatdata_test_data:
            dim_exp, bytes_exp = expected
            dim_val, bytes_val = testfunc(*args)
            self.assertEqual(dim_val, dim_exp)
            self.assertEqual(bytes_val, bytes_exp)
    
    def test__CppType_statictype(self):
        def testfunc(t, bindings):
            return str(t.statictype(bindings))
        for args, typename in self._CppType_statictype_test_data:
            self.assertEqual(testfunc(*args), typename)
    
    def test__CppType_get_cdecl_stmt(self):
        def testfunc(t, param_name, bindings):
            return t.statictype(bindings).get_cdecl_stmt(param_name)
        for args, decl_exp in self._CppType_get_cdecl_stmt_test_data:
            decl_val = testfunc(*args)
            self.assertEqual(decl_val, decl_exp)
    
    def test__CppType_get_cread_stmt(self):
        def testfunc(t, param_name, bindings, stream, dims):
            return t.statictype(bindings).get_cread_stmt(param_name, stream, dims)
        for args, decl_exp in self._CppType_get_cread_stmt_test_data:
            decl_val = testfunc(*args)
            self.assertEqual(decl_val, decl_exp)
    
    def test__CppType_get_cwrite_stmt(self):
        def testfunc(t, param_name, bindings, stream, dims):
            return t.statictype(bindings).get_cwrite_stmt(param_name, stream, dims)
        for args, decl_exp in self._CppType_get_cwrite_stmt_test_data:
            decl_val = testfunc(*args)
            self.assertEqual(decl_val, decl_exp)
    
    def test__Parameter_generatedata(self):
        def testfunc(param, glbls):
            return param.generatedata(glbls)
        for args, expected in self._Parameter_generatedata_test_data:
            name_val, type_val, dims_val, data_val = testfunc(*args)
            name_exp, type_exp, dims_exp, data_exp = expected
            #print((name_val,type_val,dims_val,data_val))
            self.assertEqual(name_val, name_exp)
            self.assertEqual(str(type_val), type_exp)
            self.assertEqual(dims_val, dims_exp)
            self.assertEqual(data_val, data_exp)
    
