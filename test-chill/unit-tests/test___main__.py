import os
import unittest

import testchill.gcov as gcov
import testchill.__main__ as main


def runtest(tc):
    tc.setUp()
    tc.run()
    tc.tearDown()

class TestMain(unittest.TestCase):
    def setUp(self):
        self.chill_dev_src = os.getenv('CHILL_DEV_SRC')
        self.chill_release_src = os.getenv('CHILL_RELEASE_SRC')
        self.omega_dev_src = os.getenv('OMEGA_DEV_SRC')
        self.omega_release_src = os.getenv('OMEGA_RELEASE_SRC')
        self.staging_dir_bin = os.getenv('STAGING_DIR_BIN')
        self.staging_dir_wd = os.getenv('STAGING_DIR_WD')
    
    def test_main_parse_chillbuild(self):
        pass
    
    def test_main_parse_chill_dev(self):
        tclist = main.args_to_tclist('-b {} chill-testcase path/to/somescript.script path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        
        self.assertEqual(tc.config.chill_dir, None)
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.build_cuda, False)
        self.assertEqual(tc.config.version, 'dev')
        self.assertEqual(tc.config.script_lang, 'script')
        
        self.assertEqual(tc.name, 'chill:somescript.script')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'chill'))
        self.assertEqual(tc.chill_script, 'somescript.script')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.script'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.c')
    
    def test_main_parse_chill_lua_dev(self):
        tclist = main.args_to_tclist('-b {} chill-testcase path/to/somescript.lua path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        
        self.assertEqual(tc.config.chill_dir, None)
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.build_cuda, False)
        self.assertEqual(tc.config.version, 'dev')
        self.assertEqual(tc.config.script_lang, 'lua')
        
        self.assertEqual(tc.name, 'chill-lua:somescript.lua')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'chill-lua'))
        self.assertEqual(tc.chill_script, 'somescript.lua')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.lua'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.c')
    
    def test_main_parse_chill_python_dev(self):
        tclist = main.args_to_tclist('-b {} chill-testcase path/to/somescript.py path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        
        self.assertEqual(tc.config.chill_dir, None)
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.build_cuda, False)
        self.assertEqual(tc.config.version, 'dev')
        self.assertEqual(tc.config.script_lang, 'python')
        
        self.assertEqual(tc.name, 'chill-python:somescript.py')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'chill-python'))
        self.assertEqual(tc.chill_script, 'somescript.py')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.py'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.c')
    
    def test_main_parse_cudachill_dev(self):
        tclist = main.args_to_tclist('-b {} chill-testcase -u path/to/somescript.lua path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        
        self.assertEqual(tc.config.chill_dir, None)
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.build_cuda, True)
        self.assertEqual(tc.config.version, 'dev')
        self.assertEqual(tc.config.script_lang, 'lua')
        
        self.assertEqual(tc.name, 'cuda-chill:somescript.lua')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'cuda-chill'))
        self.assertEqual(tc.chill_script, 'somescript.lua')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.lua'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.cu')
    
    def test_main_parse_cudachill_python_dev(self):
        tclist = main.args_to_tclist('-b {} chill-testcase -u path/to/somescript.py path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        
        self.assertEqual(tc.config.chill_dir, None)
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.build_cuda, True)
        self.assertEqual(tc.config.version, 'dev')
        self.assertEqual(tc.config.script_lang, 'python')
        
        self.assertEqual(tc.name, 'cuda-chill-python:somescript.py')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'cuda-chill-python'))
        self.assertEqual(tc.chill_script, 'somescript.py')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.py'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.cu')
    
    def test_main_parse_chill_release(self):
        tclist = main.args_to_tclist('-b {} chill-testcase -v release path/to/somescript.script path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'chill-release:somescript.script')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'chill-release'))
        self.assertEqual(tc.chill_script, 'somescript.script')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.script'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.c')
    
    def test_main_parse_chill_release(self):
        tclist = main.args_to_tclist('-b {} chill-testcase -uv release path/to/somescript.lua path/to/somesrc.c'.format(self.staging_dir_bin).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'cuda-chill-release:somescript.lua')
        self.assertEqual(tc.wd, os.getcwd())
        self.assertEqual(tc.chill_bin, os.path.join(self.staging_dir_bin, 'cuda-chill-release'))
        self.assertEqual(tc.chill_script, 'somescript.lua')
        self.assertEqual(tc.chill_src, 'somesrc.c')
        self.assertEqual(tc.chill_script_path, os.path.join(os.getcwd(), 'path/to/somescript.lua'))
        self.assertEqual(tc.chill_src_path, os.path.join(os.getcwd(), 'path/to/somesrc.c'))
        self.assertEqual(tc.chill_gensrc, 'rose_somesrc.cu')
    
    def test_main_parse_chillbuild_dev(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'chill')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'script')
    
    def test_main_parse_chillbuild_lua_dev(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -i lua'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'chill-lua')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'lua')
    
    def test_main_parse_chillbuild_python_dev(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -i python'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'chill-python')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'python')
    
    def test_main_parse_chillbuild_cuda_dev(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -u'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'cuda-chill')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'lua')
    
    def test_main_parse_chillbuild_cuda_python_dev(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -u -i python'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'cuda-chill-python')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'python')
    
    def test_main_parse_chillbuild_release(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -v release'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'chill-release')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'script')
    
    def test_main_parse_chillbuild_cuda_release(self):
        tclist = main.args_to_tclist('-b {} -C {} build-chill-testcase -u -v release'.format(self.staging_dir_bin, self.chill_dev_src).split(), coverage_set=gcov.GcovSet())
        tc = tclist[0]
        self.assertEqual(tc.name, 'cuda-chill-release')
        self.assertEqual(tc.config.bin_dir, self.staging_dir_bin)
        self.assertEqual(tc.config.chill_dir, self.chill_dev_src)
        self.assertEqual(tc.config.script_lang, 'lua')
    
    def test_main_tctree(self):
        tclist = main.args_to_tclist('batch test-cases/unit/chill-basic.tclist'.split(), coverage_set=gcov.GcovSet())
        for tc in tclist:
            runtest(tc)

    
