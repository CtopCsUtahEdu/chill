import functools
import os
import pprint
import struct
import unittest

import testchill
import testchill.util as util
import testchill._cpp_validate_env as cpp_validate_env
import testchill.cpp_validate as cpp_validate


def listtodata(flist):
    data = [struct.pack('f',n) for n in flist]
    return functools.reduce(lambda a,v: a+v, data)

class TestCppValidate(unittest.TestCase):
    def setUp(self):
        self.staging_dir_wd = os.getenv("STAGING_DIR_WD")
        self.cpp_validate_dir = os.path.join(os.getcwd(),'unit-tests/cpp_validate_prog/')
        self._parse_testproc_script_test_data = [
                (('mm_one.testproc',),      None),
                (('mm_one_with.testproc',), None)
            ]
        self._parse_testproc_python_test_data = [
            ]
        #self._generate_data_test_data = [
        #        (('mm_one.cc','in'),               None),
        #        (('mm_one.cc','out'),              None),
        #        (('mm_one_with.cc','in'),          None),
        #        (('mm_one_with.cc','out'),         None),
        #        (('mm_one_defines.cc','in'),       None),
        #        (('mm_one_defines.cc','out'),      None),
        #        (('mm_one_with_defines.cc','in'),  None),
        #        (('mm_one_with_defines.cc','out'), None),
        #    ]
        self._parse_testproc_iter_test_data = [
                (('mm_one.cc',),
                    [({'lang': 'script', 'name': 'mm_small', 'define':'dict()'},)]),
                (('mm_one_with.cc',),
                    [({'lang': 'script', 'name': 'mm_small', 'define':'dict()'},)]),
                (('mm_one_defines.cc',),
                    [({'lang': 'script', 'name': 'mm_small', 'define': "{'AN':3, 'BM':2, 'AMBN':5}"},)]),
                (('mm_one_with_defines.cc',),
                    [({'lang': 'script', 'name': 'mm_small', 'define': "{'AN':3, 'BM':2, 'AMBN':5}"},)])
            ]
        self._compile_gpp_test_data = [
                ('mm_one_main.cc', 'mm_one')
            ]
        self._test_time_test_data = [
                ((0.0034, 0.0025), True),
                ((0.0025, 0.0034), False)
            ]
        self._test_validate_test_data = [
                (('asdf', 'asdf'), True),
                (('asdf', 'sdfg'), False)
            ]
        self._run_test_validate_time_test_data = [
                (('mm_one_main.cc', 'mm_control', 'mm_one_longer_main.cc', 'mm_test', list(range(15)) + list(range(10))), (True, False)),
                (('mm_one_longer_main.cc', 'mm_control', 'mm_one_main.cc', 'mm_test', list(range(15)) + list(range(10))), (True, True)),
                (('mm_one_main.cc', 'mm_control', 'mm_one_longer_wrong_main.cc', 'mm_test', list(range(15)) + list(range(10))), (False, False))
            ]
        self._compile_run_test_validate_time_test_data = [
                (('mm_one_main.cc', 'mm_one_longer_main.cc', list(range(15)) + list(range(10))), (True, False)),
                (('mm_one_longer_main.cc', 'mm_one_main.cc', list(range(15)) + list(range(10))), (True, True)),
                (('mm_one_main.cc', 'mm_one_longer_wrong_main.cc', list(range(15)) + list(range(10))), (False, False))
            ]
        self._generate_initial_data_test_data = [
                (('mm_one.testproc', 'mm.cc', {}),      listtodata(list(range(15)) + list(range(10)) + [0]*6)),
                (('mm_one_with.testproc', 'mm.cc', {}), listtodata(list(range(15)) + list(range(10)) + [0]*6)),
            ]
        self._format_insertion_dict_test_data = [
                (('mm_one.testproc', 'mm_one.cc', {}),
                    {
                        'run': 'mm(A,B,C);',
                        'read-out': 'datafile_initialize.read((char*)C, 6*sizeof(float));',
                        'declarations': 'float A[3][5];\nfloat B[5][2];\nfloat C[3][2];',
                        'write-out': 'datafile_out.write((char*)C, 6*sizeof(float));',
                        'defines': '',
                        'read-in': 'datafile_initialize.read((char*)A, 15*sizeof(float));\ndatafile_initialize.read((char*)B, 10*sizeof(float));'
                    }),
                (('mm_one_with.testproc', 'mm_one.cc', {}),
                    {
                        'run': 'mm(A,B,C);',
                        'read-out': 'datafile_initialize.read((char*)C, 6*sizeof(float));',
                        'declarations': 'float A[3][5];\nfloat B[5][2];\nfloat C[3][2];',
                        'write-out': 'datafile_out.write((char*)C, 6*sizeof(float));',
                        'defines': '',
                        'read-in': 'datafile_initialize.read((char*)A, 15*sizeof(float));\ndatafile_initialize.read((char*)B, 10*sizeof(float));'
                    }),
            ]
        self._write_generated_code_test_data = [
                (('mm_one.testproc', 'mm_one.cc', 'control.cc', {}), 'mm_one_out.cc')
            ]
        self.run_from_src_test_data = [
                (('mm_three_basic.cc', 'mm_three_slow.cc', self.staging_dir_wd), [('small', (True, False)), ('medium', (True, False)), ('big', (True, False))]),
                (('mm_three_slow.cc', 'mm_three_basic.cc', self.staging_dir_wd), [('small', (True, True)), ('medium', (True, True)), ('big', (True, True))]),
            ]
    
    def tearDown(self):
        util.rmtemp()
    
    def test__get_script_parser(self):
        cpp_validate._script_parser = None
        self.assertIsNotNone(cpp_validate._get_script_parser())
        self.assertIsNotNone(cpp_validate._get_script_parser())
    
    def _test_parse_src(self, parsefunc, test_data):
        def parse_file(filename):
            path = os.path.join(self.cpp_validate_dir, filename)
            with open(path, 'r') as f:
                src = f.read()
            return parsefunc(src)
        for args, expected in test_data:
            srcfile, = args
            val = parse_file(srcfile)
            #TODO: make some assertions
    
    def test__parse_testproc_script(self):
        self._test_parse_src(
                cpp_validate._parse_testproc_script,
                self._parse_testproc_script_test_data)
    
    @unittest.skip("not yet supported")
    def test__parse_testproc_python(self):
        self._test_parse_src(
                cpp_validate._parse_testproc_python,
                self._parse_testproc_python_test_data)
    
    def test__parse_testproc_iter(self):
        def testfunc(filename):
            path = os.path.join(self.cpp_validate_dir, filename)
            util.shell('cp', [path, '.'], wd=self.staging_dir_wd)
            return list(cpp_validate._parse_testproc_iter(filename, wd=self.staging_dir_wd))
        for args, expected_list in self._parse_testproc_iter_test_data:
            val_list = testfunc(*args)
            for val, expected in zip(val_list, expected_list):
                _, attr_val = val
                attr_exp, = expected
                self.assertEqual(attr_val, attr_exp)
            #TODO: make some more assertions
    
    #def test__generate_data(self):
    #    def testfunc(filename, direction):
    #        path = os.path.join(self.cpp_validate_dir, filename)
    #        util.shell('cp', [path, '.'], wd=self.staging_dir_wd)
    #        for proc, attrs in cpp_validate._parse_testproc_iter(filename, wd=self.staging_dir_wd):
    #            defines = eval(attrs['define'])
    #            yield cpp_validate._generate_initial_data(proc, direction, filename, defines, wd=self.staging_dir_wd)
    #        
    #    for args, expected in self._generate_data_test_data:
    #        for filename in testfunc(*args):
    #            self.assertTrue(os.path.exists(filename))
    #        #TODO: make some more assertions
    
    def test__compile_gpp(self):
        def testfunc(src, obj):
            src = os.path.join(self.cpp_validate_dir, src)
            obj = os.path.join(self.staging_dir_wd, obj)
            cpp_validate._compile_gpp(src, obj)
        
        for src, obj in self._compile_gpp_test_data:
            testfunc(src, obj)
            obj_path = os.path.join(self.staging_dir_wd, obj)
            self.assertTrue(os.path.exists(obj_path))
    
    def test__test_time(self):
        def testfunc(control_time, test_time):
            return cpp_validate._test_time(control_time, test_time)
        
        for args, exp in self._test_time_test_data:
            val = testfunc(*args)
            self.assertEqual(val, exp)
    
    def test__test_validate(self):
        def testfunc(control_data, test_data):
            if util.python_version_major == 3:
                control_data = bytes(map(ord,control_data))
                test_data = bytes(map(ord,test_data))
            control_file, control_path = util.mktemp('wb')
            control_file.write(control_data)
            control_file.close()
            test_file, test_path = util.mktemp('wb')
            test_file.write(test_data)
            test_file.close()
            return cpp_validate._test_validate(control_path, test_path)
        
        for args, exp in self._test_validate_test_data:
            val = testfunc(*args)
            self.assertEqual(val, exp)
    
    def test__run_test_validate_time(self):
        def makeobj(src, obj):
            src_path = os.path.join(self.cpp_validate_dir, src)
            obj_path = os.path.join(self.staging_dir_wd, obj)
            util.shell('g++', ['-o', obj_path, src_path, '-lrt'])
            util.set_tempfile(obj_path)
            return src_path, obj_path
        
        def testfunc(control_src, control_obj, test_src, test_obj, in_data):
            control_src, control_obj = makeobj(control_src, control_obj)
            test_src, test_obj = makeobj(test_src, test_obj)
            inpath = os.path.join(self.staging_dir_wd, 'test.in.data')
            with open(inpath, 'wb') as infile:
                infile.write(listtodata(in_data))
            util.set_tempfile(inpath)
            return cpp_validate._run_test_validate_time(control_obj, test_obj, inpath)
        
        for args, expected in self._run_test_validate_time_test_data:
            validate_val, time_val = testfunc(*args)
            validate_exp, time_exp = expected
            self.assertEqual(validate_val, validate_exp)
            self.assertEqual(time_val, time_exp)
    
    def test__compile_run_test_validate_time(self):
        def testfunc(control_src, test_src, in_data):
            control_src = os.path.join(self.cpp_validate_dir, control_src)
            test_src = os.path.join(self.cpp_validate_dir, test_src)
            inpath = os.path.join(self.staging_dir_wd, 'test.in.data')
            with open(inpath, 'wb') as infile:
                infile.write(listtodata(in_data))
            util.set_tempfile(inpath)
            return cpp_validate._compile_run_test_validate_time(control_src, test_src, inpath)
        
        for args, expected in self._compile_run_test_validate_time_test_data:
            validate_val, time_val = testfunc(*args)
            validate_exp, time_exp = expected
            self.assertEqual(validate_val, validate_exp)
            self.assertEqual(time_val, time_exp)
    
    def test__generate_initial_data(self):
        def testfunc(testprocfile, srcfile, defines):
            testprocpath = os.path.join(self.cpp_validate_dir, testprocfile)
            with open(testprocpath, 'r') as f:
                srcpath = os.path.join(self.cpp_validate_dir, srcfile)
                testproc = cpp_validate._parse_testproc_script(f.read())
                return cpp_validate._generate_initial_data(testproc, srcpath, defines, wd=self.staging_dir_wd)
        
        for args, expected in self._generate_initial_data_test_data:
            datafile = testfunc(*args)
            with open(datafile, 'rb') as f:
                self.assertEqual(len(f.read()), len(expected))
    
    def test__format_insertion_dict(self):
        def testfunc(testprocfile, srcfile, defines):
            testprocpath = os.path.join(self.cpp_validate_dir, testprocfile)
            srcpath = os.path.join(self.cpp_validate_dir, srcfile)
            with open(testprocpath, 'r') as f:
                testproc = cpp_validate._parse_testproc_script(f.read())
                #testproc.generatedata('in', defines)
                #testproc.generatedata('out', defines)
            return cpp_validate._format_insertion_dict(testproc, srcpath, defines)
                
        for args, exp in self._format_insertion_dict_test_data:
            val = testfunc(*args)
            for k,v in exp.items():
                self.assertEqual(val[k], v)
    
    def test__write_generated_code(self):
        def testfunc(testprocfile, srcname, destname, defines):
            srcpath = os.path.join(self.cpp_validate_dir, srcname)
            with open(os.path.join(self.cpp_validate_dir, testprocfile),'r') as f:
                testproc = cpp_validate._parse_testproc_script(f.read())
            return cpp_validate._write_generated_code(testproc, srcpath, defines, destname, self.staging_dir_wd)
        for args, exp_path in self._write_generated_code_test_data:
            val_path = testfunc(*args)
            util.set_tempfile(val_path)
            exp_path = os.path.join(self.cpp_validate_dir, exp_path)
            with open(val_path, 'r') as valfile:
                with open(exp_path, 'r') as expfile:
                    self.assertEqual(valfile.read().splitlines(), expfile.read().splitlines())
    
    def test_run_from_src(self):
        for args, expected in self.run_from_src_test_data:
            control_src, test_src, wd = args
            control_src = os.path.join(self.cpp_validate_dir, control_src)
            test_src = os.path.join(self.cpp_validate_dir, test_src)
            val = list(cpp_validate.run_from_src(control_src,test_src,wd))
            self.assertEqual(val, expected)
            
