import difflib
import functools
import itertools
import logging
import os
import re
import sysconfig
import subprocess
import tempfile



logging.basicConfig(filename='testchill.log', level=logging.DEBUG, filemode='w')
#logging.basicConfig(level=logging.INFO)

python_version = sysconfig.get_python_version()
python_version_major = int(sysconfig.get_python_version().split('.')[0])
python_version_minor = int(sysconfig.get_python_version().split('.')[1])

if python_version_major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

_temp_dirs = []
_temp_files = []

### Errors ###
### Shell Util ###

def shell(cmd, args=[], stdout=None, stderr=None, env={}, wd=os.getcwd()):
    """
    Execute a shell command.
    @params cmd The command name
    @params args A list of command line arguments (defaults to [])
    @params stdout A file like object or file number that reads input written to stdout.
            stdout will be returned as a string if this is None or not given.
    @params stderr A file like object or file number that reads input written to stderr.
    @params env A dict of environment variables. Before the command is executed, these will be exported
    @params wd The working directory. Before the command is executed, the working directory will be changed to wd. (wd defaults to the current working directory)
    """
    fullcmd = ' '.join(['export {}={};'.format(k,str(v)) for k,v in env.items()] + ['cd {};'.format(wd)] + [cmd] + args)
    logging.info('shell: '+fullcmd)
    if stdout == None:
        outp = subprocess.check_output(fullcmd, stderr=stderr, shell=True)
        if python_version_major == 2:
            return outp
        elif python_version_major == 3:
            return outp.decode()
    else:
        subprocess.check_call(fullcmd, stdout=stdout, stderr=stderr, shell=True)

def mkdir_p(directory, temp=False, **kwargs):
    """
    Make directory (equivelent to shell('mkdir', ['-p', directory]))
    """
    if not os.path.exists(directory):
        if temp and (directory not in _temp_dirs):
            _temp_dirs.append(directory)
        shell('mkdir', ['-p', directory], **kwargs)

def set_tempfile(filename):
    """
    Add a file to a list of temp files
    @param filename The full path to a temparary file.
    """
    _temp_files.append(filename)

def withtmp(wtfunc, rdfunc):
    """
    Perform some operation using a temporary file.
    @param wtfunc A function that writes to the temparary file
    @param rdfybc A function that reads from the temparary file
    """
    with tempfile.TemporaryFile() as f:
        wtfunc(f)
        f.seek(0)
        return rdfunc(f)

def rmtemp():
    """
    Clean temp files and directories
    """
    for temp_file in list(_temp_files):
        if os.path.exists(temp_file):
            shell('rm', [temp_file])
        _temp_files.remove(temp_file)
        
    for temp_dir in list(_temp_dirs):
        if os.path.exists(temp_dir):
            shell('rm', ['-rf', temp_dir])
        _temp_dirs.remove(temp_dir)

def mktemp(mode=None):
    """
    Create a temparary file. Returns a two-tuple with an open file object and the filename.
    """
    fd, name = tempfile.mkstemp()
    _temp_files.append(name)
    if mode is None:
        os.close(fd)
        return name
    else:
        return os.fdopen(fd, mode), name
    

### Misc Util ###

def copy(obj, exclude=[]):
    """
    Make a shallow copy of a python object with __dict__, excluding any attribute in exclude
    @param obj The object to copy
    @param exclude A list of attributes to ignore
    """
    nobj = type(obj)()
    for k, v in vars(obj).items():
        if k in exclude: continue
        setattr(nobj, k, v)
    return nobj

def applyenv(line):
    """
    Apply bash style environment variables to a string
    @param line The input string
    """
    return re.sub(r'\$([a-zA-Z_][a-zA-Z_0-9]*)\b',lambda m: str(os.getenv(m.group(1), '')), line)

def callonce(func):
    """
    Assert that a function is only ever called once.
    @param func Function to only be run once.
    """
    pred_name = '__' + func.__module__.replace('.','__') + '_' + func.__name__ + '_called'
    globals()[pred_name] = False
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not globals()[pred_name]:
            globals()[pred_name] = True
            return func(*args, **kwargs)
        else:
            raise Exception
    return wrapper

def isdiff(strone, strtwo):
    """
    Diff two strings. Returns a two element tuple. The first is True if the the two files are different, and the
    next is a textual representation of the diff.
    @param strone First string.
    @param strtwo Second string.
    """
    diff = list(difflib.ndiff(strone.splitlines(), strtwo.splitlines()))
    return len(list(line for line in diff if line[0] in ['-','+'])) != 0, '\n'.join(diff)

def filterext(ext_list, filenames):
    """
    Filter file names by extension.
    @param ext_list A list of extensions.
    @param filenames An iterable object of file names.
    """
    return iter(s for s in filenames if any(s.strip().endswith(e) for e in ext_list))

def extract_tag(tagname, filename, wd=os.getcwd()):
    """
    Extract commented out text in each html tag '<tagname>'. Returns a list of tuples for each tag.
    Each tuple has two elements, the first is the text found in the tag, the second contains a dict
    of attributes given in the tag.
    @param tagname The name of the tag to search for.
    @param filename A filename to search for comments in.
    @param wd The working directory.
    """
    from . import _extract
    return _extract.extract_tag(tagname, filename, wd)

def textstream(txt):
    """
    Creates a stream from text. Intended to hide version differences between 2 and 3.
    @param txt A string to use as the default data in a stream.
    """
    if python_version_major == 2:
        import StringIO
        return StringIO.StringIO(txt)
    elif python_version_major == 3:
        import io
        return io.StringIO(txt)

