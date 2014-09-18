import collections
import os
import os.path
import itertools
import re

from . import util

if util.python_version_major == 2:
    from HTMLParser import HTMLParser
else:
    from html.parser import HTMLParser

class _TagExtractor(HTMLParser):
    _comment_style_expr = {
            'c':      [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'cc':     [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'cpp':    [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'h':      [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'hh':     [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'hpp':    [('/(/)+',r'[\n]'),(r'/\*',r'\*/')],
            'py':     [('#+',r'[\n]'),('\'\'\'',),('"""',)],
            'script': [('#+',r'[\n]')],
            'lua':    [(r'--\[\[',r'\]\]--')]
        }
    
    def __init__(self, tagname):
        HTMLParser.__init__(self)
        self.tagname = tagname
        self._readin = False
        self._value = ''
    
    def handle_starttag(self, tag, attrs):
        if tag == self.tagname:
            self._readin = True
            self._attrs = dict(attrs)
    
    def handle_endtag(self, tag):
        if tag == self.tagname:
            self._readin = False
            self._tag_list.append((self._value, self._attrs))
            self._value = ''
    
    def handle_data(self, txt):
        if self._readin:
            self._value += txt
    
    @classmethod
    def _parse(cls, tagname, txt):
        reader = cls(tagname)
        reader._readin = False
        reader._value = ''
        reader._tag_list = []
        reader.feed(txt)
        return reader._tag_list
    
    @classmethod
    def _get_commentstyles(cls, ext):
        for comment_style in cls._comment_style_expr[ext]:
            if len(comment_style) == 1:
                start_expr = comment_style[0]
                end_expr = comment_style[0]
            elif len(comment_style) == 2:
                start_expr = comment_style[0]
                end_expr = comment_style[1]
            yield start_expr, end_expr
    
    @classmethod
    def _commented(cls, txt, ext):
        comment_spans = list()
        for start_expr, end_expr in cls._get_commentstyles(ext):
            pos = 0
            while pos < len(txt):
                start_match = re.search(start_expr, txt[pos:])
                if start_match:
                    start_pos = pos + start_match.end()
                    end_match = re.search(end_expr, txt[start_pos:])
                    if end_match:
                        end_pos = start_pos + end_match.start()
                        pos = start_pos + end_match.end()
                    else:
                        end_pos = len(txt)
                        pos = end_pos
                    comment_spans.append((start_pos, end_pos))
                else:
                    break
        for span in sorted(comment_spans, key=lambda s: s[0]):
            yield txt[span[0]:span[1]]
    
    @classmethod
    def extract_tag(cls, tagname, filename, wd=os.getcwd()):
        with open(os.path.join(wd, filename), 'r') as f:
            txt = f.read()
        ext = filename.split('.')[-1]
        return cls._parse(tagname, '\n'.join(cls._commented(txt, ext)))

extract_tag = _TagExtractor.extract_tag

