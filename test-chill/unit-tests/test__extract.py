import ast
import unittest

import testchill._extract as _extract
import testchill.util as util

class TestExtraction(unittest.TestCase):
    def setUp(self):
        self._TagExtractor_parse_test_data = [
                (('a',''),                      []),
                (('a','x<a>yy</a>z'),           [('yy', {})]),
                (('a','x<a>yy</a>z<a>ww</a>g'), [('yy', {}), ('ww',{})]),
                (('a','x<a>yy</a>z<b>ww</b>g'), [('yy', {})])
            ]
        self._commented_test_data = [
                (('no comment here','cc'), []),
                (('one comment //xxx\n','cc'), ['xxx']),
                (('two comments //xxx\nunrelated//yyy\n', 'cc'), ['xxx','yyy']),
                (('two comments //xxx\nunrelated//yyy', 'cc'), ['xxx','yyy']),
                (('ss/*x\ny\n*/z','cc'),['x\ny\n']),
                (('ss/*x\ny\n*/z//q\nc','cc'),['x\ny\n','q']),
                (('ss###x#\n','py'),['x#']),
                (('ss"""x"""\n','py'),['x'])
            ]
    
    def test__commented(self):
        def run(txt, ext):
            return list(_extract._TagExtractor._commented(txt, ext))
        for args, res in self._commented_test_data:
            self.assertEqual(run(*args), res)
    
    #def test_extract(self):
    #    def testfunc(tag, txt):
    #        temp = util.mktemp()
    #        with open(temp, 'w') as f:
    #            f.write(txt)
    #        extracted = _extract._TagExtractor.extract_tag(tag, temp)
    #        util.rmtemp()
    #        return extracted
    #        
    #    for args, res in self.test_extract_data:
    #        self.assertEqual(testfunc(*args), res)
    
    def test__TagExtractor_parse(self):
        def testfunc(tag, txt):
            return _extract._TagExtractor._parse(tag, txt)
        for args, exp in self._TagExtractor_parse_test_data:
            self.assertEqual(testfunc(*args), exp)
