import pylang.parser
import pickle

if __name__ == '__main__':
    gstream = open('testchill/cpp_validate/grammar.txt', 'r')
    env = dict()
    exec('from testchill._cpp_validate_env import *', None, env)
    parser = pylang.parser.generate(gstream, env)
    pickle.dump(parser, open('testchill/cpp_validate/parser.pickle', 'wb'), 2)
