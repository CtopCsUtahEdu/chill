#!/usr/bin/python2

import xml.etree.ElementTree as ET
import os
import subprocess
import sys
import glob


class UTdriver:
    def __init__(self):
        self.tests = 0
        self.failures = 0
        self.errors = 0
        self.skip = 0
        self.skiplog = ""
        self.errorlog = ""

    def getOutput(self, testsuit):
        os.system(testsuit + " --gtest_output=xml:res.xml")
        e = ET.parse('res.xml').getroot()
        self.tests += int(e.get('tests'))
        self.failures += int(e.get('failures'))
        self.errors += int(e.get('errors'))

    def printResult(self):
        print 'Total of {0:d} tests:'.format(self.tests)
        print '\tpass  {0:d}'.format(self.tests - self.failures - self.errors - self.skip)
        print '\tskip  {0:d}'.format(self.skip)
        print '\terror {0:d}'.format(self.errors)
        print '\tfail  {0:d}'.format(self.failures)


def runTestFolder(driver, folder):
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        if os.path.isfile(fp):
            print fp
            driver.getOutput(fp)


def runScriptFolder(driver, folder):
    for file in sorted(glob.glob(folder + "/*.test")):
        fp = file
        if os.path.isfile(fp):
            ret = subprocess.call([fp], stderr=None, stdout=None)
            driver.tests += 1
            if ret == 77:
                driver.skiplog += fp + " .. SKIP\n"
                driver.skip += 1
            elif ret == 99:
                driver.errorlog += fp + " .. FAIL\n"
                driver.failures += 1
            elif ret != 0:
                driver.errorlog += fp + " .. ERROR\n"
                driver.errors += 1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage <this> <unittests> <script folders>'
    d = UTdriver()
    # TODO better multiplexer
    runTestFolder(d, sys.argv[1])
    runScriptFolder(d, sys.argv[2])
    d.printResult()
    print d.skiplog, d.errorlog
    if d.failures + d.errors > 0:
        exit(-1)
