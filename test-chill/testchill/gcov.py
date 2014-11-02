from __future__ import print_function
import functools
import itertools
import os
import os.path
import pickle
import sys

from . import util

class GcovFile(object):
    def __init__(self, src_file_name, cov_file_path, lines, properties):
        """
        @param src_file_name Name of the source file.
        @param cov_file_path Full path to the coverage file.
        @param lines List of GcovLine objects.
        @param properties Properties from the coverage file.
        """
        self.src_file_name = src_file_name
        self.cov_file_path = cov_file_path
        self.lines = lines
        self.properties = properties
    
    @staticmethod
    def parse_file(gcov, fname, process=None):
        """
        Parse a file into a GcovFile object.
        @param gcov Gcov object that tis file is a part of.
        @param gname File name.
        @param process Process name
        """
        util.shell('gcov', [fname], wd=gcov.srcdir)
        cov_file_path = os.path.join(gcov.srcdir, fname + '.gcov')
        src_file_name = fname
        if os.path.exists(cov_file_path):
            with open(cov_file_path, 'r') as f:
                lines, properties = GcovFile.parse_lines(f.readlines(), process)
            return GcovFile(src_file_name, cov_file_path, lines, properties)
        else:
            return None
    
    @staticmethod
    def parse_lines(str_lines, process):
        """
        Parse a string from a coverage file into a list of GcovLine objects.
        @param str_lines Full text of a coverage file.
        @param process Name of the process that executed the code.
        """
        properties = dict()
        lines = []
        for line in str_lines:
            if line[-1] == '\n':
                line = line[0:-1]
            pline = line.split(':')
            pline = list(map(str.strip, pline[0:2])) + pline[2:]
            if pline[1] == '0':
                properties[pline[2]] = pline[3].strip()
            elif pline[0][0] == '-':
                lines.append(GcovLine(int(pline[1]), dict(), ':'.join(pline[2:])))
            elif pline[0][0] == '#':
                lines.append(GcovLine(int(pline[1]), {process : 0}, ':'.join(pline[2:])))
            else:
                lines.append(GcovLine(int(pline[1]), {process : int(pline[0])}, ':'.join(pline[2:])))
        return lines, properties
    
    @staticmethod
    def union(left, right):
        """
        Merge two different coverages of the same file into a single coverage object.
        """
        return left | right
    
    def __or__(self, right):
        """
        Merge two different coverages of the same file into a single coverage object.
        """
        new_file = self.clone()
        new_file.merge(right)
        return new_file
    
    def __ior__(self, right):
        """
        Merge two different coverages of the same file into a single coverage object.
        """
        self.merge(right)
        return self
    
    def merge(self, other):
        """
        Merge another coeverage into self.
        """
        assert self.src_file_name == other.src_file_name
        GcovLine.merge_lines(self.lines, other.lines)
        self.properties.update(other.properties)
    
    def clone(self):
        """
        Create a shallow clone.
        """
        return GcovFile(self.src_file_name, self.cov_file_path, list(self.lines), dict(self.properties))


class GcovLine(object):
    def __init__(self, lineno, count_by_process, code):
        """
        @param lineno Line number.
        @param count_by_prcess A dictionary of execution counts by name of the process that executed them.
        @param code Source code from this line.
        """
        self.lineno = lineno
        self.count_by_process = count_by_process
        self.code = code
    
    @staticmethod
    def merge_lines(lines, other_lines):
        """
        Merge lines from other_line into lines.
        """
        for line, other_line in zip(lines, other_lines):
            assert line.lineno == other_line.lineno
            assert line.code == other_line.code
            line.count_by_process.update(other_line.count_by_process)
    
    def count(self):
        """
        The total number of times this line was executed.
        """
        runable_list = [l for l in self.count_by_process.values() if l is not None]
        if len(runable_list) == 0:
            return None
        else:
            return sum(runable_list)
    
    def __repr__(self):
        return str((self.lineno, self.count_by_process, self.code))


class Gcov(object):
    def __init__(self, srcdir):
        self.srcdir = srcdir
        self.files = dict()
    
    @staticmethod
    def parse(srcdir, process=None):
        gcov = Gcov(srcdir)
        gcov._append(filter(lambda f: f is not None, map(functools.partial(GcovFile.parse_file, gcov, process=process),
                util.filterext(['cc','c','cpp','h','hh'], os.listdir(srcdir)))))
        return gcov
    
    def _append(self, files):
        for f in files:
            if f.src_file_name in self.files:
                self.files[f.src_file_name].merge(f)
            else:
                self.files[f.src_file_name] = f
    
    def __or__(self, right):
        new_cov = self.clone()
        new_cov.merge(right)
        return new_cov
    
    def __ior__(self, right):
        self.merge(right)
        return self
    
    @staticmethod
    def union(left, right):
        return left | right
    
    def merge(self, other):
        self._append(other.files.values())
    
    def clone(self):
        new_cov = Gcov(self.srcdir)
        new_cov._append(iter(f.clone() for f in self.files.values()))
        return new_cov


class GcovSet(object):
    def __init__(self):
        self.coverage_by_program = dict()
    
    def addprogram(self, prog_name, src_dir):
        self.coverage_by_program[prog_name] = Gcov(src_dir)
    
    def addcoverage(self, prog_name, process_name):
        cov = self.coverage_by_program[prog_name]
        cov.merge(Gcov.parse(cov.srcdir, process_name))
    
    def unexecuted_lines(self):
        covlist = sorted(self.coverage_by_program.values(), key=lambda c: c.srcdir)
        for src, grp in itertools.groupby(covlist, lambda c: c.srcdir):
            files = functools.reduce(lambda a, c: a | c, grp).files.values()
            file_lines = iter((f.src_file_name, iter(l for l in f.lines if l.count() == 0)) for f in files)
            yield src, file_lines
    
    #def pretty_print(self, outfile=sys.stdout, width=60, stats=['unexecuted', 'unexecuted.bysrc']):
    #    print('='*width, file=outfile)
    #    print('  CODE COVERAGE', file=outfile)
    #    
    #    if 'unexecuted' in stats:
    #        print('='*width, file=outfile)
    #        print('    unexecuted lines', file=outfile)
    #        if 'unexecuted.bysrc' in stats:
    #            for src, file_lines in self.unexecuted_lines():
    #                print((src + ':'), file=outfile)
    #                print('-'*width, file=outfile)
    #                for src_file_name, lines in file_lines:
    #                    print('  ' + src_file_name + ':', file=outfile)
    #                    for line in lines:
    #                        print("{}:{}".format(str(line.lineno).rjust(5), line.code), file=outfile)
    #    #print('='*width, file=outfile)
    #    #print(prog, file=outfile)
    #    #print('-'*width, file=outfile)
    
    def _get_coverage_by_file(self):
        return functools.reduce(lambda a,b: a|b, self.coverage_by_program.values()).files
    
    coverage_by_file = property(_get_coverage_by_file)


def load(filename = 'coverage.pickle'):
    with open(filename) as f:
        return pickle.load(f)


def lines(covset, filename, predicate=None):
    if predicate is None:
        predicate = lambda l: True
    for line in filter(predicate, covset.coverage_by_file[filename].lines):
        yield line.lineno, line.count(), line.code


def nonexecuted(covset, filename):
    return lines(covset, filename, lambda line: line.count() == 0)


def commented(covset, filename):
    return lines(covset, filename, lambda line: line.count() is None)


