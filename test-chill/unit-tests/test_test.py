import io
import pickle
import pprint
import unittest
import textwrap

import testchill.test as test
import testchill.util as util


class Named(object):
    def __init__(self, name):
        self.name = name
    
    def setresult(self, res):
        pass

def make_tc(rfunc=None, sufunc=None, tdfunc=None, name=None):
    class SomeTestCase(test.TestCase):
        def setUp(self):
            if sufunc:
                sufunc(self)
            
        def run(self):
            if rfunc != None:
                return rfunc(self)
            
        def tearDown(self):
            if tdfunc:
                tdfunc(self)
    
    return SomeTestCase(name)

def make_seqtc(subtests, sufunc=None, tdfunc=None, name=None):
    class SomeSeqTestCase(test.SequencialTestCase):
        def __init__(self, name):
            test.SequencialTestCase.__init__(self, name)
            for fn_name, func in subtests:
                self.add_subtest(fn_name, func)
        
        def setUp(self):
            if sufunc:
                sufunc(self)
                
        def tearDown(self):
            if tdfunc:
                tdfunc(self)
        
    return SomeSeqTestCase(name)


class TestTest(unittest.TestCase):
    
    def flip_n_switch(self, n, value=True):
        '''
        Return a function that sets switches[n] to value (True by default)
        '''
        def flipswitch(tc):
            self.switches[n] = value
        return flipswitch
    
    def flip_n_switch_if_m(self, n, m, value=True):
        '''
        Returns a function that sets switches[n] to value (True by default) if switches[m] is True
        '''
        def flipswitch(tc):
            if self.switches[m]:
                self.switches[n] = value
        return flipswitch
    
    def allways_raise(self, exc=Exception('Expected exception')):
        '''
        Returns a function that raises an exception
        '''
        def throwexc(tc):
            raise exc
        return throwexc
    
    def allways_fail(self):
        '''
        Returns a function that returns an explicit failure
        '''
        def fail(tc):
            return test.TestResult.make_fail(test.TestResult, tc)
        return fail
    
    def allways_skip(self):
        '''
        Returns a function that skips
        '''
        def skip(tc):
            return test.TestResult.make_skipped(test.TestResult, tc)
        return skip
    
    def allways_pass(self):
        '''
        Returns a function that passes
        '''
        def notfail(tc):
            return test.TestResult.make_pass(test.TestResult, tc)
        return notfail
    
    def donothing(self):
        '''
        Returns a function that does nothing
        '''
        def foo(tc):
            pass
        return foo
    
    def setUp(self):
        self.switches = dict((n, False) for n in range(3))
    
    def test_TestResult_make_pass(self):
        self.assertTrue(test.TestResult.make_pass(test.TestResult, Named('i-pass')).passed())
        self.assertFalse(test.TestResult.make_pass(test.TestResult, Named('i-pass')).errored())
        self.assertFalse(test.TestResult.make_pass(test.TestResult, Named('i-pass')).failed())
        self.assertFalse(test.TestResult.make_pass(test.TestResult, Named('i-pass')).skipped())
    
    def test_TestResult_make_error(self):
        self.assertFalse(test.TestResult.make_error(test.TestResult, Named('i-error')).passed())
        self.assertTrue(test.TestResult.make_error(test.TestResult, Named('i-error')).errored())
        self.assertFalse(test.TestResult.make_error(test.TestResult, Named('i-error')).failed())
        self.assertFalse(test.TestResult.make_error(test.TestResult, Named('i-error')).skipped())
    
    def test_TestResult_make_fail(self):
        self.assertFalse(test.TestResult.make_fail(test.TestResult, Named('i-fail')).passed())
        self.assertFalse(test.TestResult.make_fail(test.TestResult, Named('i-fail')).errored())
        self.assertTrue(test.TestResult.make_fail(test.TestResult, Named('i-fail')).failed())
        self.assertFalse(test.TestResult.make_fail(test.TestResult, Named('i-fail')).skipped())
    
    def test_TestResult_make_skipped(self):
        self.assertFalse(test.TestResult.make_skipped(test.TestResult, Named('i-skip')).passed())
        self.assertFalse(test.TestResult.make_skipped(test.TestResult, Named('i-skip')).errored())
        self.assertFalse(test.TestResult.make_skipped(test.TestResult, Named('i-skip')).failed())
        self.assertTrue(test.TestResult.make_skipped(test.TestResult, Named('i-skip')).skipped())
    
    def test__result(self):
        result_passed = test.TestResult.make_pass(test.TestResult, Named('i-pass'))
        result_failed = test.TestResult.make_fail(test.TestResult, Named('i-fail'))
        self.assertTrue(result_passed is test._result(result_passed, Named('i-pass')))
        self.assertTrue(test._result(result_failed, Named('i-fail')).failed())
        self.assertTrue(test._result(Exception(), Named('i-error')).errored())
    
    def test_run_empty(self):
        test.run([])
    
    def test_run_run(self):
        test.run([make_tc(
            rfunc=self.flip_n_switch(0))])
        self.assertTrue(self.switches[0])
    
    def test_run_setupfirst(self):
        test.run([make_tc(
                rfunc = self.flip_n_switch_if_m(0,1),
                sufunc = self.flip_n_switch(1))])
        self.assertTrue(self.switches[0])
    
    def test_run_teardownlast(self):
        test.run([make_tc(
                rfunc = self.flip_n_switch(1),
                tdfunc = self.flip_n_switch_if_m(0,1))])
        self.assertTrue(self.switches[0])
    
    def test_run_teardown_allways(self):
        test.run([make_tc(
                rfunc = self.allways_raise(),
                tdfunc = self.flip_n_switch(0))])
        self.assertTrue(self.switches[0])
    
    def test_run_pass_result(self):
        result_set = test.run([make_tc(
                rfunc = self.donothing(),
                name='pass')])
        result = result_set[0]
        self.assertTrue(result.passed())
        self.assertFalse(result.errored())
        self.assertFalse(result.failed())
        self.assertFalse(result.skipped())
    
    def test_run_error_result(self):
        result_set = test.run([make_tc(
                rfunc = self.allways_raise(),
                name='error')])
        result = result_set[0]
        self.assertFalse(result.passed())
        self.assertTrue(result.errored())
        self.assertFalse(result.failed())
        self.assertFalse(result.skipped())
    
    def test_run_fail_result(self):
        result_set = test.run([make_tc(
                rfunc = self.allways_fail(),
                name='fail')])
        result = result_set[0]
        self.assertFalse(result.passed())
        self.assertFalse(result.errored())
        self.assertTrue(result.failed())
        self.assertFalse(result.skipped())

    def test_run_skipped_result(self):
        result_set = test.run([make_tc(
                rfunc = self.allways_skip(),
                name='skipped')])
        result = result_set[0]
        self.assertFalse(result.passed())
        self.assertFalse(result.errored())
        self.assertFalse(result.failed())
        self.assertTrue(result.skipped())
    
    def test_run_seq_empty(self):
        test.run([make_seqtc([])])
    
    def test_run_seq_allrun(self):
        result_set = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', self.flip_n_switch(1)),
                ('three', self.flip_n_switch(2))],
                name='seq')])
        self.assertTrue(result_set[0].passed())
        self.assertTrue(self.switches[0])
        self.assertTrue(self.switches[1])
        self.assertTrue(self.switches[2])
    
    def test_run_seq_until_fail(self):
        result_set = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', self.allways_fail()),
                ('trhee', self.flip_n_switch(1))],
                name='seq')])
        self.assertTrue(result_set[0].failed())
        self.assertTrue(self.switches[0])
        self.assertFalse(self.switches[1])
    
    def test_run_seq_until_error(self):
        result_set = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', self.allways_raise()),
                ('trhee', self.flip_n_switch(1))],
                name='seq')])
        self.assertTrue(result_set[0].errored())
        self.assertTrue(self.switches[0])
        self.assertFalse(self.switches[1])
    
    def test_persistance_one_pass(self):
        result_set = test.run([make_tc(self.allways_pass(), name='tc-name')])
        read_result_set = util.withtmp(
            lambda f: pickle.dump(result_set, f),
            lambda f: pickle.load(f))
        self.assertEqual(list(map(vars,result_set)), list(map(vars,read_result_set)))
    
    def test_persistance_seq(self):
        result_set = test.run([make_seqtc([
            ('one', self.flip_n_switch(0)),
            ('two', self.flip_n_switch(1))],
            name = 'seq')])
        read_result_set = util.withtmp(
            lambda f: pickle.dump(result_set, f),
            lambda f: pickle.load(f))
        
        for i in range(len(result_set)):
            self.assertEqual(result_set[i].status, read_result_set[i].status)
            self.assertEqual(result_set[i].testcase_name, read_result_set[i].testcase_name)
            for j in range(len(result_set[i].sub_results)):
                self.assertEqual(result_set[i].sub_results[j].status, read_result_set[i].sub_results[j].status)
                self.assertEqual(result_set[i].sub_results[j].testcase_name, read_result_set[i].sub_results[j].testcase_name)
    
    def test_persistance_seq_error(self):
        result_set = test.run([make_seqtc([
            ('one', self.flip_n_switch(0)),
            ('two', self.allways_raise())],
            name = 'seq')])
        read_result_set = util.withtmp(
            lambda f: pickle.dump(result_set, f),
            lambda f: pickle.load(f))
        
        for i in range(len(result_set)):
            self.assertEqual(result_set[i].status, read_result_set[i].status)
            self.assertEqual(result_set[i].testcase_name, read_result_set[i].testcase_name)
            for j in range(len(result_set[i].sub_results)):
                self.assertEqual(result_set[i].sub_results[j].status, read_result_set[i].sub_results[j].status)
                self.assertEqual(result_set[i].sub_results[j].testcase_name, read_result_set[i].sub_results[j].testcase_name)
        
    def test_FailedTestResult_init(self):
        result = test.TestResult.make_fail(test.FailedTestResult, Named('i-fail'), reason='testing')
        self.assertFalse(result.passed())
        self.assertTrue(result.failed())
        self.assertFalse(result.errored())
        self.assertFalse(result.skipped())
        self.assertEqual(result.testcase_name, 'i-fail')
        self.assertEqual(result.reason, 'testing')
    
    def test_pretty_print(self):
        def pretty_print_to_string(results_iter):
            sio = util.StringIO()
            test.pretty_print_results(results_iter, outfile=sio)
            return sio.getvalue()
            
        results_iter = iter([
                test.TestResult.make_pass(test.TestResult, Named('i-pass')),
                test.TestResult.make_error(test.UnhandledExceptionTestResult, Named('i-error'), Exception, Exception(), None),
                test.TestResult.make_fail(test.FailedTestResult, Named('i-fail'), reason='Oops'),
                test.TestResult.make_skipped(test.TestResult, Named('i-skip'))
            ])
        
        self.assertEqual(pretty_print_to_string(results_iter), textwrap.dedent('''\
            Passed: 1
            Errors: 1
            Failed: 1
            Skipped: 1
            ============================================================
            error: i-error
            ------------------------------------------------------------
            <class 'Exception'>: 
            ------------------------------------------------------------
            ============================================================
            fail: i-fail
            ------------------------------------------------------------
            Oops
            ------------------------------------------------------------
            '''))
        result_set = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', self.flip_n_switch(1)),
                ('three', self.flip_n_switch(2))],
                name='seq')])
        result_set_fail = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', lambda s: test.TestResult.make_fail(test.FailedTestResult, s, 'Oops')),
                ('trhee', self.flip_n_switch(1))],
                name='seq')])
        result_set_error = test.run([make_seqtc([
                ('one', self.flip_n_switch(0)),
                ('two', lambda s: test.TestResult.make_error(test.UnhandledExceptionTestResult, s, Exception, Exception(), None))],
                name = 'seq')])
        
        
        
        compound_set = pretty_print_to_string(result_set)
        compound_set_fail = pretty_print_to_string(result_set_fail)
        compound_set_error = pretty_print_to_string(result_set_error)
        
        self.assertEqual(compound_set, textwrap.dedent('''\
            Passed: 1
            Errors: 0
            Failed: 0
            Skipped: 0
            '''))
        self.assertEqual(compound_set_fail, textwrap.dedent('''\
            Passed: 0
            Errors: 0
            Failed: 1
            Skipped: 0
            ============================================================
            fail: seq
            ------------------------------------------------------------
            pass: one
            fail: two
            Oops
            ------------------------------------------------------------
            '''))
        self.assertEqual(compound_set_error, textwrap.dedent('''\
            Passed: 0
            Errors: 1
            Failed: 0
            Skipped: 0
            ============================================================
            error: seq
            ------------------------------------------------------------
            pass: one
            error: two
            <class 'Exception'>: 
            ------------------------------------------------------------
            '''))
    
    def tearDown(self):
        util.rmtemp()

if __name__ == '__main__':
    unittest.main()
