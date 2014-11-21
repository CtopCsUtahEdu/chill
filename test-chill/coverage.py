import argparse
import pickle


def loadcov(filename = 'coverage.pickle'):
    with open(filename) as f:
        return pickle.load(f)


def lines(covset, filename):
    for line in covset.coverage_by_file[filename].lines:
        yield line.lineno, line.count(), line.code


def nonexecuted(covset, filename):
    return filter(lambda line: line[1] == 0, lines(covset, filename))


def commented(covset, filename):
    return filter(lambda line: line[1] is None, lines(covset, filename))


def linerange(lineiter, minline, maxline):
    return filter(lambda line: line[0] >= minline and line[0] <= maxline, lineiter)


def print_nonexec(argsns, cov):
    if argsns.filename is None:
        covlist = list((k, len(list(nonexecuted(cov, k)))) for k in cov.filenames)
        covlist = sorted(covlist, key=lambda i: i[1])
        for i in reversed(range(len(covlist))):
            print('{}: {}'.format(covlist[i][0].ljust(24), covlist[i][1]))
    else:
        minline, maxline = map(int,argsns.linerange)
        for lineno, count, code in linerange(nonexecuted(cov, argsns.filename), minline, maxline):
            print('{}: {}'.format(str(lineno).rjust(5), code))


def make_argparser():
    arg_parser = argparse.ArgumentParser('coverage.py')
    cmd_parser_set = arg_parser.add_subparsers()
    nonexec_cmd = cmd_parser_set.add_parser('nonexec')
    nonexec_cmd.add_argument('-f', dest='filename', default=None)
    nonexec_cmd.add_argument('-r', dest='linerange', nargs=2, default=(0, 120000), metavar='STARTLINE ENDLINE')
    nonexec_cmd.set_defaults(func=print_nonexec)
    return arg_parser
    

if __name__ == '__main__':
    argsns = make_argparser().parse_args()
    cov = loadcov()
    argsns.func(argsns, cov)

