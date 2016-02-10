from __future__ import print_function
#TODO: test dependencies
#TODO: expected failures
import itertools
import io
import logging
import pprint
import sys
import traceback

from . import util


class TestResult(object):
    """
    The base class for all test results.
    """
    _pass = 'pass'
    _error = 'error'
    _fail = 'fail'
    _skipped = 'skipped'
    
    def __init__(self, testcase, status):
        self.testcase_name = testcase.name
        self.status = status
        testcase.setresult(self)
    
    @staticmethod
    def make_pass(result_type, testcase, *args, **kwargs):
        """
        Create and return a passing test result of type result_type.
        @param result_type A class that extends TestResult
        @param testcase The test case that generated the result
        @param *args Additional positional arguments to be passed to result_type.__init__
        @param *kwargs Keyword arguments to be passed to result_type.__init__
        """
        return result_type(testcase, TestResult._pass, *args, **kwargs)
    
    @staticmethod
    def make_error(result_type, testcase, *args, **kwargs):
        """
        Create and return a errored test result of type result_type.
        @param result_type A class that extends TestResult
        @param testcase The test case that generated the result
        @param *args Additional positional arguments to be passed to result_type.__init__
        @param *kwargs Keyword arguments to be passed to result_type.__init__
        """
        return result_type(testcase, TestResult._error, *args, **kwargs)
    
    @staticmethod
    def make_fail(result_type, testcase, *args, **kwargs):
        """
        Create and return a failed test result of type result_type.
        @param result_type A class that extends TestResult
        @param testcase The test case that generated the result
        @param *args Additional positional arguments to be passed to result_type.__init__
        @param *kwargs Keyword arguments to be passed to result_type.__init__
        """
        return result_type(testcase, TestResult._fail, *args, **kwargs)
    
    @staticmethod
    def make_skipped(result_type, testcase, *args, **kwargs):
        """
        Create and return a skipped test result of type result_type.
        @param result_type A class that extends TestResult
        @param testcase The test case that generated the result
        @param *args Additional positional arguments to be passed to result_type.__init__
        @param *kwargs Keyword arguments to be passed to result_type.__init__
        """
        return result_type(testcase, TestResult._skipped, *args, **kwargs)
    
    def passed(self):
        """ Return true iff the testcase passed. """
        return self.status == TestResult._pass
    
    def errored(self):
        """ Return true iff the testcase passed. """
        return self.status == TestResult._error

    def failed(self):
        """ Return true iff the testcase passed. """
        return self.status == TestResult._fail
    
    def skipped(self):
        """ Return true iff the testcase was skipped """
        return self.status == TestResult._skipped
        
    def pprint_dict(self):
        """
        Return a dict that is ideal for passing to pprint.
        """
        return {'testcase_name': self.testcase_name, 'status':self.status}
    
    def pretty_print(self, width=60, outfile=sys.stdout):
        """
        Print result to a file in a human readable way.
        """
        print('='*width, end='\n', file=outfile)
        print("{}: {}".format(self.status, self.testcase_name), end='\n', file=outfile)
        print('-'*width, end='\n', file=outfile)
        print(self.pretty_message(), end='\n', file=outfile)
        print('-'*width, end='\n', file=outfile)
    
    def pretty_message(self):
        """ Return a message to be printed by pretty_print. Returns an empyt string if not overriden. """
        return ''
        


class FailedTestResult(TestResult):
    """
    A basic implementation of TestResult for failed tests.
    """
    def __init__(self, testcase, status=TestResult._fail, reason=None):
        super(FailedTestResult, self).__init__(testcase, status)
        self.reason = reason
    
    def pprint_dict(self):
        """
        Return a dict that is ideal for passing to pprint.
        """
        ppdict = super(FailedTestResult, self).pprint_dict()
        ppdict['reason'] = self.reason
        return ppdict
    
    def pretty_message(self):
        return self.reason


class CompoundTestResult(TestResult):
    """
    A TestResult returned by running a sequencial test case
    """
    def __init__(self, testcase, results):
        super(CompoundTestResult, self).__init__(testcase, None)
        self.sub_results = results
        status_list = [r.status for r in results]
        if TestResult._fail in status_list:
            self.status = TestResult._fail
        elif TestResult._error in status_list:
            self.status = TestResult._error
        elif TestResult._pass in status_list:
            self.status = TestResult._pass
        else:
            self.status = TestResult._skipped
    
    def pprint_dict(self):
        """
        Returns a dict that is ideal for passing to pprint.
        """
        ppdict = super(CompoundTestResult, self).pprint_dict()
        ppdict['sub_results'] = list(s.pprint_dict() for s in self.sub_results)
        return ppdict
    
    def pretty_message(self):
        return '\n'.join(
                "{}: {}{}".format(
                    st.status,
                    st.testcase_name,
                    '\n' + st.pretty_message() if st.status in [TestResult._fail, TestResult._error] else '')
                for st in self.sub_results)


class SubTestResult(TestResult):
    """
    A TestResult for a subtest in a sequencial test case.
    """
    def __init__(self, subtest_name, inner_result):
        """
        @param subtest_name The name of the subtest.
        @param inner_result The result returned from running the subtest.
        """
        super(SubTestResult, self).__init__(inner_result.testcase, inner_result.status)
        self.inner_result = inner_result
    
    def pprint_dict(self):
        """
        Return a dict that is ideal for passing to pprint.
        """
        ppdict = super(CompoundTestResult, self).pprint_dict()
        ppdict['inner_result'] = self.inner_result.pprint_dict()
        return ppdict


class UnhandledExceptionTestResult(TestResult):
    """
    A TestResult returned for exceptions that the test case failed to handle.
    """
    def __init__(self, testcase, status, exc_type, exc_value, exc_traceback):
        super(UnhandledExceptionTestResult, self).__init__(testcase, status)
        self.exception_type = exc_type
        self.exception_value = exc_value
        if not exc_traceback is None:
            sio = util.StringIO()
            traceback.print_exception(self.exception_type, self.exception_value, exc_traceback, file=sio)
            self.exception_message = sio.getvalue()
        else:
            self.exception_message = "{}: {}".format(str(exc_type), str(exc_value))
    
    def pprint_dict(self):
        """
        Return a dict that is ideal for passing to pprint.
        """
        ppdict = super(UnhandledExceptionTestResult, self).pprint_dict()
        ppdict['exception_type'] = self.exception_type
        ppdict['exception_value'] = self.exception_value
        ppdict['exception_message'] = self.exception_message
        return ppdict
    
    def pretty_message(self):
        return self.exception_message


class TestCase(object):
    """
    Base class for all test cases
    """
    def __init__(self, name=None):
        """
        @param name A unique test case name.
        """
        self.name = name    
    
    def setUp(self):
        """
        Called imediately before a testcase is executed.
        """
        pass
    
    def run(self):
        """
        Run the test case, and return its result.
        """
        raise NotImplementedError
    
    def tearDown(self):
        """
        Called imediately after a testcase is executed.
        """
        pass
    
    def catch(self, exc):
        """
        Called when run raises an exception. If the test case
        knows how to handle it, it should return it's own result or None.
        Otherwise, return the original exception.
        """
        return exc
    
    def setresult(self, test_result):
        """
        Called after a test issues a result and before tearDown is called.
        """
        self.test_result = test_result
    
    def make_pass(self, result_type=TestResult, *args, **kwargs):
        """
        Make a passed result for this testcase.
        """
        return TestResult.make_pass(result_type, self, *args, **kwargs)
    
    def make_fail(self, result_type=FailedTestResult, *args, **kwargs):
        """
        Make a failed result for this testcase.
        """
        return TestResult.make_fail(result_type, self, *args, **kwargs)


class SequencialTestCase(TestCase):
    """
    A test case that executes a sequence of subtests until
    one fails.
    """
    def __init__(self, name):
        super(SequencialTestCase, self).__init__(name)
        self.tests = []
    
    def add_subtest(self, subtest_name, subtest_func):
        """
        Add a subtest.
        """
        self.tests.append((subtest_name, subtest_func))
    
    def run(self):
        return CompoundTestResult(self, list(self._runall()))
    
    def _runall(self):
        return _rungen([SubTestCase(name, func) for name, func in self.tests], failfast=True)


class SubTestCase(TestCase):
    """
    A subtest of a sequncial test.
    """
    def __init__(self, name, func):
        super(SubTestCase, self).__init__(name)
        self.run = lambda: func(self)


def run(tclist, failfast=False):
    """
    Run all test cases in tclist and return a list of thier results.
    """
    return list(_rungen(tclist, failfast))

def _rungen(tclist, failfast=False):
    """
    A generator for running tests internally.
    """
    for tc in tclist:
        result = None
        tc.setUp()
        try:
            result = _result(tc.run(), tc)
        except Exception as ex:
            result = _result(tc.catch(ex), tc)
        tc.tearDown()
        yield result
        if failfast and (result.failed() or result.errored()):
            break

def _result(res, tc):
    """
    Convert res to a TestResult object.
    If res is a TestResult object, give it back.
    If res is an Exception, return an UnandledExceptionTestResult.
    If res is something else, discard it and return a passed TestResult.
    """
    if isinstance(res, TestResult):
        return res
    elif isinstance(res, Exception):
        logging.info('uncaught exception: {}'.format(str(res)))
        return TestResult.make_error(UnhandledExceptionTestResult, tc, *(sys.exc_info()))
    else:
        return TestResult.make_pass(TestResult, tc)

def pprint_results(result_iter, outfile=sys.stdout):
    """
    Print pprint version of test results to a file-like object.
    @param result_iter An iterator of results to print.
    @param outfile An opened file-like object to print to (defaults to stdout).
    """
    status_func = lambda r: r.status
    result_iter = sorted(result_iter, key=status_func)
    status_dict = dict(iter((k, list(map(lambda tc: tc.pprint_dict(), g))) for k, g in itertools.groupby(result_iter, status_func)))
    pprint.pprint(status_dict, stream=outfile)

def pretty_print_results(
        result_iter,
        count_by_status=True, exclude_passed=True, exclude_skipped=True, exclude_failed=False,
        exclude_errored=False, sort_by_status=True, width=60, outfile=sys.stdout):
    """
    Print iterator of TestResults in a human readable format to a file-like object.
    @param result_iter An iterator of TestResult objects to print.
    @param count_by_status Print the number of tests for each status (defaults to True).
    @param exclude_passed Exclude passed test results from printing (defaults to True).
    @param exclude_skipped Exclude skipped test results from printing (defaults to True).
    @param exclude_failed Exclude failed test results from printing (defaults to False).
    @param exclude_errored Exclude errored test results from printing (defaults to False).
    @param sort_by_status Print test results in order of status: passed, errored, failed, then skipped (defaults to True).
    @param width Printing width (defaults to 60).
    @param outfile A file-like object to print to (defaults to stdout).
    """
    result_list = list(result_iter)
    status_func = lambda r: r.status
    if sort_by_status:
        #TODO: printing order
        result_iter = sorted(result_iter, key=status_func)
    
    if count_by_status:
        print('Passed: {}'.format(len([tr for tr in result_list if tr.passed()])), file=outfile)
        print('Errors: {}'.format(len([tr for tr in result_list if tr.errored()])), file=outfile)
        print('Failed: {}'.format(len([tr for tr in result_list if tr.failed()])), file=outfile)
        print('Skipped: {}'.format(len([tr for tr in result_list if tr.skipped()])), file=outfile)
    #TODO: something that doesn't expose TestResult._*
    print_status = set(itertools.compress([TestResult._pass, TestResult._error, TestResult._fail, TestResult._skipped],
            map(lambda n: not n, [exclude_passed, exclude_errored, exclude_failed, exclude_skipped])))
    for tr in (r for r in result_list if r.status in print_status):
        tr.pretty_print(width=width, outfile=outfile)
    
    
