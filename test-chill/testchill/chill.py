#TODO: Re-Document
#TODO: highlight test implementation hooks

import os
import os.path

from . import gcov
from . import test
from . import util
from . import cpp_validate


class ChillConfig(object):
    _config_map = dict(('-'.join(map(str,k)),v) for k,v in [
            (('dev',False,'script'),     ('chill',              'depend-chill',      'chill',      '')),
            (('dev',False,'lua'),        ('chill-lua',          'depend-chill',      'chill',      'SCRIPT_LANG=lua')),
            (('dev',False,'python'),     ('chill-python',       'depend-chill',      'chill',      'SCRIPT_LANG=python')),
            (('dev',True,'lua'),         ('cuda-chill',         'depend-cuda-chill', 'cuda-chill', '')),
            (('dev',True,'python'),      ('cuda-chill-python',  'depend-cuda-chill', 'cuda-chill', 'SCRIPT_LANG=python')),
            (('release',False,'script'), ('chill-release',      'depend',            'chill',      '')),
            (('release',True,'lua'),     ('cuda-chill-release', 'depend-cuda-chill', 'cuda-chill', ''))
        ])
    
    def __init__(self, omega_dir=None, chill_dir=None, bin_dir=None, build_cuda=False, script_lang=None, version='dev'):
        self.version = version
        self.build_cuda = build_cuda
        self.script_lang = script_lang
        self.omega_dir = omega_dir
        self.chill_dir = chill_dir
        self.bin_dir = bin_dir
        if self.script_lang is None:
            self.script_lang = self.default_script_lang()
    
    def _get(self, index):
        return ChillConfig._config_map[self.version + '-' + str(self.build_cuda) + '-' + self.script_lang][index]
    
    def default_script_lang(self):
        if self.build_cuda:
            return 'lua'
        else:
            return 'script'
    
    def name(self):
        return self._get(0)
    
    def make_depend_target(self):
        return self._get(1)
    
    def make_target(self):
        return self._get(2)
    
    def make_args(self):
        return self._get(3)
    
    def _buildfunc(self, cc, link=True):
        if not link:
            compile_args = ['-c -Wuninitialized']
        elif link and cc == 'nvcc':
            compile_args = ['-L/usr/local/cuda/lib64/lib', '-lcuda', '-lcudart', '-lstdc++', '-lrt', '-Wuninitialized']
        else:
            compile_args = ['-lstdc++', '-lrt', '-Wuninitialized']
        
        def build(src, dest, args=[], defines={}, wd=None):
            if wd is None:
                wd = os.path.dirname(src)
            args += ['-D{}={}'.format(k,v) for k, v in defines.items()]
            dest = os.path.join(wd, dest)
            stdout = util.shell(cc, args + [src, '-o', dest] + compile_args, wd=wd)
            return dest, stdout
        return build
    
    def compile_src_func(self):
        return self._buildfunc('gcc', False)
    
    def compile_gensrc_func(self):
        if self.build_cuda:
            return self._buildfunc('nvcc', False)
        else:
            return self._buildfunc('gcc', False)
    
    def build_src_func(self):
        return self._buildfunc('gcc')
    
    def build_gensrc_func(self):
        if self.build_cuda:
            return self._buildfunc('nvcc')
        else:
            return self._buildfunc('gcc')
    
    def env(self):
        chill_env = {'OMEGAHOME':self.omega_dir}
        if self.version == 'release' and self.build_cuda:
            chill_env['CUDACHILL']='true'
        return chill_env
    
    @staticmethod
    def ext_to_script_lang(ext):
        return {'script':'script', 'lua':'lua', 'py':'python'}[ext]
    
    @staticmethod
    def configs(omega_dir, chill_dir, bin_dir, build_cuda=None, script_lang=None, version=None):
        all_configs = [
                (False, 'script', 'dev'),
                (False, 'script', 'release'),
                (False, 'lua', 'dev'),
                (False, 'python', 'dev'),
                (True, 'lua', 'dev'),
                (True, 'lua', 'release'),
                (True, 'python', 'dev')]
                
        pred_list = [lambda x: True]
        if not build_cuda is None:
            pred_list += [lambda x: x[0] == build_cuda]
        if not script_lang is None:
            pred_list += [lambda x: x[1] == script_lang]
        if not version is None:
            pred_list += [lambda x: x[2] == version]
        
        cond = lambda x: all(p(x) for p in pred_list)
        
        return iter(ChillConfig(omega_dir, chill_dir, bin_dir, *conf) for conf in filter(cond, all_configs))


# -                               - #
# -  Test case for building chill - #
# -                               - #
class BuildChillTestCase(test.TestCase):
    """
    Test case for building chill.
    """
    
    default_options = {
            'coverage': False   # compile for coverage
        }
    
    def __init__(self, config, options={}, coverage_set=None):
        """
        @param config chill configuration object
        @param options options for building chill and testing the build process
        @param coverage_set GcovSet object to record coverage
        """
        assert isinstance(config, ChillConfig)
        if config.script_lang == None:
            config.script_lang = config.default_script_lang()
        self.config = config
        super(BuildChillTestCase,self).__init__(self.config.name())
        self._set_options(options, coverage_set)
    
    def _set_options(self, options, coverage_set):
        self.options = dict(BuildChillTestCase.default_options)
        self.options.update(options)
        
        self.build_env = self.config.env()
        self.build_args = self.config.make_args()
        if self.options['coverage']:
            self.build_args += ' "TEST_COVERAGE=1"'
            coverage_set.addprogram(self.config.name(), self.config.chill_dir)
    
    def setUp(self):
        """
        Called before run, outside of the context of a test case
        """
        # clean up any coverage files from a previous build
        util.shell('rm', ['-f', '*.gcno'], wd=self.config.chill_dir)
        util.shell('rm', ['-f', '*.gcov'], wd=self.config.chill_dir)
        util.shell('rm', ['-f', '*.gcda'], wd=self.config.chill_dir)
        
        util.shell('make clean', wd=self.config.chill_dir)
        util.shell('make veryclean', wd=self.config.chill_dir)
    
    def run(self):
        """
        Build chill
        """
        depend_target = self.config.make_depend_target()
        target = self.config.make_target()
        util.shell('make', ['clean'], wd=self.config.chill_dir)
        util.shell('make', ['veryclean'], wd=self.config.chill_dir)
        util.shell('make', [depend_target] + [self.build_args], env=self.build_env, wd=self.config.chill_dir)
        util.shell('make', [target] + [self.build_args], env=self.build_env, wd=self.config.chill_dir)
        return self.make_pass()
        
    def tearDown(self):
        """
        Called after run, outside of the context of a test case.
        If a binary directory is specified, rename and move the executable there, otherwise, just rename it.
        """
        if self.test_result.passed():
            if self.config.bin_dir:
                util.shell('mv', [os.path.join(self.config.chill_dir, self.config.make_target()), os.path.join(self.config.bin_dir, self.config.name())])
            else:
                util.shell('mv', [os.path.join(self.config.chill_dir, self.config.make_target()), os.path.join(self.config.chill_dir, self.config.name())])


# -                              - #
# -  Test case for running chill - #
# -                              - #
class RunChillTestCase(test.SequencialTestCase):
    """
    Test case for running and testing chill.
    """
    
    default_options={
            'compile-src':True,              # Compile original source file
            'run-script':True,               # Run chill script
            'compile-gensrc':True,           # Compile generated source file
            'check-run-script-stdout':False, # Diff stdout from run_script() against an expected value (from a .stdout file)
            'coverage':False,                # Record coverage
            
            'fail-compile-src':False,        # Expect compile_src to fail (TODO: not implemented)
            'fail-run-script':False,         # Expect run_script to fail  (TODO: not implemented)
        }
    
    def __init__(self, config, chill_script, chill_src, wd=None, options={}, coverage_set=None):
        """
        @param config Chill configuration object
        @param chill_script The path to the chill script.
        @param chill_src The path to the source file that the script uses.
        @param wd The working directory. Where the script will be executed, compiled, and tested.
        @param options Additional testing options.
        @param coverage_set GcovSet object to record coverage
        """
        if config.script_lang == None:
            config.script_lang = ChillConfig.ext_to_script_lang(chill_script.split('.')[-1])
        
        assert isinstance(config, ChillConfig)
        
        super(RunChillTestCase,self).__init__(config.name() + ':' + os.path.basename(chill_script))
        
        self.config = config
        self.wd = wd if (wd != None) else os.getcwd()
        
        self.chill_src_path = os.path.abspath(chill_src)
        self.chill_script_path = os.path.abspath(chill_script)
        self.chill_bin = os.path.join(self.config.bin_dir, self.config.name())
        self.chill_src = os.path.basename(self.chill_src_path)
        self.chill_script = os.path.basename(self.chill_script_path)
        self.chill_gensrc = self._get_gensrc(self.chill_src)
        self.chill_gensrc_path = os.path.join(self.wd, self.chill_gensrc)
        
        self.compile_src_func = self.config.compile_src_func()
        self.compile_gensrc_func = self.config.compile_gensrc_func()
        self.build_src_func = self.config.build_src_func()
        self.build_gensrc_func = self.config.build_gensrc_func()
        
        self._set_options(options, coverage_set)

    def _set_options(self, options, coverage_set=None):
        self.options = dict(RunChillTestCase.default_options)
        self.options.update(options)
        
        self.out = dict()
        self.expected = dict()
        
        if self.options['compile-src']:
            self.add_subtest('compile-src', self.compile_src)
        if self.options['run-script']:
            self.add_subtest('run-script', self.run_script)
        if self.options['compile-gensrc']:
            self.add_subtest('compile-generated-src', self.compile_gensrc)
        self.add_subtest('check-run-script-validate', self.check_run_script_validate)
        if self.options['check-run-script-stdout']:
            self.add_subtest('check-run-script-stdout', self.check_run_script_stdout)
            with open('.'.join(self.chill_script_path.split('.')[0:-1] + ['stdout']), 'r') as f:
                self.expected['run_script.stdout'] = f.read()
        self.coverage_set = coverage_set
    
    def _get_gensrc(self, src):
        """
        The name of the generated source file.
        """
        if not self.config.build_cuda:
            return 'rose_' + src
        else:
            return 'rose_' + '.'.join(src.split('.')[0:-1]) + '.cu'
    
    def setUp(self):
        """
        Called before any tests are performed. Moves source and script files into the working directory
        and removes any gcov data files
        """
        util.shell('cp', [self.chill_src_path, self.chill_src], wd=self.wd)
        util.shell('cp', [self.chill_script_path, self.chill_script], wd=self.wd)
        #TODO: check for chill binary
    
    def tearDown(self):
        """
        Called when the test is complete
        """
        util.shell('rm', ['-f', self.chill_src], wd=self.wd)
        util.shell('rm', ['-f', self.chill_script], wd=self.wd)
        util.shell('rm', ['-f', self.chill_gensrc], wd=self.wd)
        if self.options['coverage'] and self.coverage_set is not None:
            self.coverage_set.addcoverage(self.config.name(), self.name)
    
    # -             - #
    # - Chill Tests - #
    # -             - #
    
    def compile_src(self, tc):
        """
        Attempts to compile the source file before any transformation is performed. Fails if gcc fails.
        """
        #self.out['compile_src.stdout'] = util.shell('gcc', ['-c', self.chill_src], wd=self.wd)
        _, self.out['compile_src.stdout'] = self.compile_src_func(self.chill_src, util.mktemp(), wd=self.wd)
        return tc.make_pass()
    
    def run_script(self, tc):
        """
        Attempts to run the script file. Fails if chill exits with a non-zero result.
        """
        # look for cudaize.lua for cuda-chill
        if self.config.build_cuda and not os.path.exists(os.path.join(self.wd, 'cudaize.lua')):
            return test.TestResult.make_error(test.FailedTestResult, tc, reason='cudaize.lua was missing from the working directory.')
        self.out['run_script.stdout'] = util.shell(self.chill_bin, [self.chill_script], wd=self.wd)
        return tc.make_pass()
    
    def compile_gensrc(self, tc):
        """
        Attempts to compile the generated source file. Fails if gcc fails.
        """
        #self.out['compile_gensrc.stdout'] = util.shell('gcc', ['-c', self.chill_gensrc], wd=self.wd)
        _, self.out['compile_gensrc.stdout'] = self.compile_gensrc_func(self.chill_gensrc_path, util.mktemp(), wd=self.wd)
        return tc.make_pass()
    
    def check_run_script_validate(self, tc):
        """
        Generate test data and run both the original source and generated source against it.
        Fail if any test procedure generates different output.
        """
        for name, (is_valid, is_faster) in cpp_validate.run_from_src(self.chill_src, self.chill_gensrc, self.build_src_func, self.build_gensrc_func, wd=self.wd):
            self.out['check_run_script_validate.{}'.format(name)] = (is_valid, is_faster)
            if not is_valid:
                return tc.make_fail('test procedure {} returned invalid results.'.format(name))
        return tc.make_pass()
    
    def check_run_script_stdout(self, tc):
        """
        Diff stdout from run_script against an expected stdout
        """
        isdiff, diff = util.isdiff(self.out['run_script.stdout'], self.expected['run_script.stdout'])
        if isdiff:
            return test.TestResult.make_fail(test.FailedTestResult, tc, reason='Diff:\n' + diff)
        return tc.make_pass()
    
