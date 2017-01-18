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

    def getOutput(self,testsuit):
        os.system(testsuit + " --gtest_output=xml:res.xml")
        e = ET.parse('res.xml').getroot()
        self.tests += int(e.get('tests'))
        self.failures += int(e.get('failures'))
        self.errors += int(e.get('errors'))

    def printResult(self):
        print 'Total of {0:d} tests:\npass  {1:d}\nskip  {2:d}\nerror {3:d}\nfail  {4:d}\n'.format(self.tests,self.tests
                - self.failures - self.errors - self.skip, self.skip, self.errors, self.failures)

def runTestFolder(driver,folder):
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        if os.path.isfile(fp):
            print fp
            driver.getOutput(fp)

def runScriptFolder(driver,folder):
    for file in sorted(glob.glob(folder+"/*.test")):
        fp = file
        if os.path.isfile(fp):
            ret = subprocess.call([fp], stderr=None, stdout=None)
            driver.tests += 1
            if ret == 77:
                print fp+" .. SKIP"
                driver.skip += 1
            elif ret == 99:
                print fp+" .. FAIL"
                driver.failures += 1
            elif ret != 0:
                print fp+" .. ERROR"
                driver.errors += 1

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage <this> <unittests> <script folders>'
    d = UTdriver()
# TODO better multiplexer
    runTestFolder(d, sys.argv[1])
    runScriptFolder(d, sys.argv[2])
    d.printResult()
    if d.failures + d.errors > 0:
        exit(-1)

