import itertools
import pprint
import os
import textwrap
import unittest

import testchill.util as util
import testchill.gcov as gcov


class TestGCov(unittest.TestCase):
    def setUp(self):
        self.cprog_dir = os.path.join(os.getcwd(), 'unit-tests/cprog')
        self.cprog_bin = os.path.join(self.cprog_dir, 'bin/sorter')
    
    def build_prog(self):
        self.clean_prog()
        util.shell('make', [], wd=self.cprog_dir)
    
    def clean_prog(self):
        util.shell('make', ['clean'], wd=self.cprog_dir)
    
    def run_prog(self, alg, lst):
        util.shell(self.cprog_bin, [alg] + list(map(str,lst)))
    
    def test_GcovLine_mrege_lines(self):
        '''
           56:   14:        while((index < pivot_index) && (list[index] >= pivot_value)) {
            6:   15:            swap(list, index, pivot_index);
            6:   16:            pivot_index--;
            -:   17:        }
        And
            78:   14:        while((index < pivot_index) && (list[index] >= pivot_value)) {
            18:   15:            swap(list, index, pivot_index);
            18:   16:            pivot_index--;
            -:   17:        }
        '''
        lines_proc_one = list(itertools.starmap(gcov.GcovLine,[ 
                (14, {'proc_one':   56},'        while((index < pivot_index) && (list[index] >= pivot_value)) {'),
                (15, {'proc_one':    6},'            swap(list, index, pivot_index);'),
                (16, {'proc_one':    6},'            pivot_index--;'),
                (17, {'proc_one': None},'        }')]))
        lines_proc_two = list(itertools.starmap(gcov.GcovLine,[
                (14, {'proc_two':   78},'        while((index < pivot_index) && (list[index] >= pivot_value)) {'),
                (15, {'proc_two':   18},'            swap(list, index, pivot_index);'),
                (16, {'proc_two':   18},'            pivot_index--;'),
                (17, {'proc_two': None},'        }')]))
        gcov.GcovLine.merge_lines(lines_proc_one, lines_proc_two)
        self.assertEqual(lines_proc_one[0].lineno, 14)
        self.assertEqual(lines_proc_one[1].lineno, 15)
        self.assertEqual(lines_proc_one[2].lineno, 16)
        self.assertEqual(lines_proc_one[3].lineno, 17)
    
    def test_GcovLine_merge_and_count(self):
        lines_proc_one = list(itertools.starmap(gcov.GcovLine,[ 
                (14, {'proc_one':   56},'        while((index < pivot_index) && (list[index] >= pivot_value)) {'),
                (15, {'proc_one':    6},'            swap(list, index, pivot_index);'),
                (16, {'proc_one':    6},'            pivot_index--;'),
                (17, {'proc_one': None},'        }')]))
        lines_proc_two = list(itertools.starmap(gcov.GcovLine,[
                (14, {'proc_two':   78},'        while((index < pivot_index) && (list[index] >= pivot_value)) {'),
                (15, {'proc_two':   18},'            swap(list, index, pivot_index);'),
                (16, {'proc_two':   18},'            pivot_index--;'),
                (17, {'proc_two': None},'        }')]))
        gcov.GcovLine.merge_lines(lines_proc_one, lines_proc_two)
        self.assertEqual(lines_proc_one[0].count(), 134)
        self.assertEqual(lines_proc_one[1].count(), 24)
        self.assertEqual(lines_proc_one[2].count(), 24)
        self.assertEqual(lines_proc_one[3].count(), None)
    
    def test_GcovFile_parse_lines(self):
        lines = textwrap.dedent(
            '''-:0:SomeProperty:SomeValue
               56:   14:        while((index < pivot_index) && (list[index] >= pivot_value)) {
                6:   15:            swap(list, index, pivot_index);
                6:   16:            pivot_index--;
                -:   17:        }''').splitlines()
        lines, properties = gcov.GcovFile.parse_lines(lines, 'proc')
        self.assertEqual(lines[0].lineno, 14)
        self.assertEqual(lines[0].count_by_process, {'proc': 56})
        self.assertEqual(lines[0].code, '        while((index < pivot_index) && (list[index] >= pivot_value)) {')
        self.assertEqual(lines[3].count_by_process, dict())
    
    def test_Gcov_parse(self):
        self.build_prog()
        self.run_prog('quicksort', [9, 4, 10, 6, 11, 0, 3, 7, 2, 1, 8, 5])
        cov = gcov.Gcov.parse(self.cprog_dir, 'unsorted')
        self.build_prog()
        self.run_prog('quicksort', [5, 4, 3, 2, 1])
        #pprint.pprint(vars(cov.files['QuickSorter.cc']))
        cov.merge(gcov.Gcov.parse(self.cprog_dir, 'reverse'))
        #pprint.pprint(vars(cov.files['QuickSorter.cc']))
        #TODO: assert something
        #cov.pretty_print()
    
    def tearDown(self):
        self.clean_prog()
        
