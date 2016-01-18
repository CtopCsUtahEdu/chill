import os
import subprocess
import tempfile
import unittest

import testchill.util as util

### Most of these are sanity checks. ###

class TestUtil(unittest.TestCase):
    def setUp(self):
        self.tempfiles = []
    
    def maketempfiles(self, n=1):
        files = tuple([tempfile.mkstemp(text=True) for i in range(n)])
        self.tempfiles += list(map(lambda f: f[1], files))
        return files
        
    def test_shell(self):
        sbla = subprocess.check_output(['ls', '-la', 'test-cases/chill'])
        
        if util.python_version_major == 3:
            sbla = sbla.decode()
        
        shla = util.shell('ls', ['-la', 'test-cases/chill'])
        self.assertEqual(sbla, shla)
    
    def test_shell_env(self):
        env = {'STRING_VAR':'string','NUMBER_VAR':3,'DEFINED_VAR':1}
        
        self.assertEqual(util.shell('echo', ['$STRING_VAR'], env=env), env['STRING_VAR'] + '\n')
        self.assertEqual(util.shell('echo', ['$NUMBER_VAR'], env=env), str(env['NUMBER_VAR']) + '\n')
        self.assertEqual(util.shell('echo', ['$DEFINED_VAR'], env=env), str(env['DEFINED_VAR']) + '\n')
    
    def test_shell_tofile(self):
        tfile = self.maketempfiles(1)
        fname = tfile[0][1]
        
        with open(fname, 'w') as f:
            util.shell('ls', ['-la', 'test-cases/chill'], stdout=f)
        with open(fname, 'r') as f:
            self.assertEqual(util.shell('ls', ['-la', 'test-cases/chill']), f.read())
    
    def test_copy(self):
        class C(object):
            pass
        c = C()
        c.x = 'x'
        a = util.copy(c)
        b = util.copy(c)
        a.x = 'y'
        self.assertEqual(c.x,'x')
        self.assertEqual(b.x,'x')
        self.assertEqual(a.x,'y')
    
    def test_callonce(self):
        def foo():
            return 3
        foo_once = util.callonce(foo)
        self.assertEqual(foo_once(), 3)
        self.assertRaises(Exception, foo_once)
    
    def test_isdiff(self):
        testdata = [
                (('aaa','aaa'),(False,'  aaa')),
                (('aab','aaa'),(True, '- aab\n+ aaa')),
                (('a\nb','a\nb\nc'),(True, '  a\n  b\n+ c')),
                (('a\na\nc','a\nb\nc'),(True, '  a\n- a\n+ b\n  c'))
            ]
        for args, expected in testdata:
            isdiff_exp, diff_exp = expected
            isdiff_val, diff_val = util.isdiff(*args)
            self.assertEqual(isdiff_val, isdiff_exp)
            self.assertEqual(diff_val, diff_exp)
    
    def test_filterext(self):
        testdata = [
                ((['.c','.py'],['a.c','b.txt','c.py']),['a.c','c.py'])
            ]
        for args, expected in testdata:
            self.assertEqual(list(util.filterext(*args)), expected)
    
    #TODO:
    #def test_extract_tag(self):
    #    testdata = [
    #            (('a', 'abc<a>def</a>ghi<b>jkl</b>mno<c>pqr</c>stu<b>zwx</b>yz'), ['def']),
    #            (('b', 'abc<a>def</a>ghi<b>jkl</b>mno<c>pqr</c>stu<b>zwx</b>yz'), ['jkl','zwx']),
    #            (('c', 'abc<a>def</a>ghi<b>jkl</b>mno<c>pqr</c>stu<b>zwx</b>yz'), ['pqr']),
    #            (('d', 'abc<a>def</a>ghi<b>jkl</b>mno<c>pqr</c>stu<b>zwx</b>yz'), []),
    #        ]
    #    for args, expected in testdata:
    #        self.assertEqual(list(util.extract_tag(*args)), expected)
    
    def test_textstream(self):
        testdata = [
                (('asdf',),'asdf')
            ]
        for args, expected in testdata:
            stream = util.textstream(*args)
            self.assertTrue(hasattr(stream,'read'))
            self.assertEqual(stream.read(), expected)
    
    def tearDown(self):
        for f in self.tempfiles:
            os.remove(f)
    

