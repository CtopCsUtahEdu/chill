#!/usr/bin/python2

import xml.etree.ElementTree as ET
import os
import sys

class UTdriver:
    def __init__(self):
        self.tests=0
        self.failures=0
        self.errors=0

    def getOutput(self,testsuit):
        os.system(testsuit + " --gtest_output=xml:res.xml")
        e = ET.parse('res.xml').getroot()
        self.tests += int(e.get('tests'))
        self.failures += int(e.get('failures'))
        self.errors += int(e.get('errors'))

    def printResult(self):
        print 'Total of {0:d} tests have {1:d} errors'.format(self.tests,self.failures+self.errors)

def runTestFolder(driver,folder):
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        if os.path.isfile(fp):
            print fp
            driver.getOutput(fp)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print 'Usage <this> <testfolder>'
        exit(-1)
    d = UTdriver()
    runTestFolder(d,sys.argv[1])
    d.printResult()
    if d.failures + d.errors > 0:
        exit(-1)
    else:
        exit(0)
