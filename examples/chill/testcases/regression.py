#! /usr/bin/python

import glob
import string
import sys
from subprocess import call
import os
import stat
import time 

# find all the .script files
files = glob.glob("*.script")
#print files 
#files = files[0:1] # pick one
#files = [ "unroll1a.script" ]  # pick one 

files = sorted( files )

RESULTSDIR = "RIGHTANSWERS/"


def cleanup( lines ):  # this does not handle   /* stuff */

  #for line in lines:
  #    print line,

  l1 = []
  # remove things to the right of // on each line
  for line in lines:
     print line,
     if "//" in line:
       print "contains comment " + line,
       where = line.find("//")
       l = line[:where] + "\n" # remove from end
       #print l,
       l = l.strip() # remove blanks from both ends
       llen = len(l)
       print "length %d '%s'" % ( llen, l)
       if llen > 0: 
         l1.append(l)
       else:
         print "removing comment-only line"
     else:
       l1.append( line )


  # remove #define lines, for now, since clang cna't handle that
  l2 = []
  for line in l1:
    if "#define" not in line:
      l2.append(line)
      
  # turn tab to a single space, delete blank lines 
  l3 = []
  for line in l2:
    stripped =  string.strip( line.replace("\t", " ") )
    llen = len(stripped)
    if llen > 0:
       l3.append( stripped )

  onebig = ""
  for line in l3:
     onebig += line
  
  #print onebig + "\n\n"
  l = len(onebig)

  norepeats = ""  # no repeated spaces 
  prev = ""
  for i in range(l):
     if onebig[i] == " ":
        if prev != " ": 
           norepeats += onebig[i]   # first space
     else:
        norepeats += onebig[i]   # not a space 
     prev = onebig[i]
  #print norepeats


  return norepeats 
  
      


def comparelines( lines, klines):  # lines and known good lines. may have comments 

  same = False
  
  output = cleanup( lines)
  known  = cleanup( klines )

  if output != known:
    for l in output:
      print l
    print "\ndoes not match known good\n"

    klen = len(klines)
    llen = len(lines)
    print "klen %d    llen %d" % ( klen, llen)
    minlen = llen
    if klen < llen:
       minlen = klen
    print "minlen %d" % minlen
    
    c = 0
    for c in range( minlen ): 
      print c,
      print klines[c],
      print c,
      print lines[c]
      c += 1
    print "\n\n"
  else:
    #print "output matches known good"
    same = True
  return same


def regressionTest():
  resultssucceeded = []
  resultsfailed = []

  print "\n"
  for file in files:
    print file

    # read all the lines in a single .script file 
    f = open( file, "r" )
    lines = f.readlines()
    f.close()

    src = ""
    dest=""

    #   for files that SHOULD fail. failing is the correct thing
    #   if the test is SUPPOSED to fail, there will be a file named
    #   RESULTSDIR + file + ".SHOULDFAIL"
    failfile = RESULTSDIR + file + ".SHOULDFAIL"
    
    shouldfail = False
    try:
      f = open(failfile)  
      print file + " should NOT work"
      f.close()
      shouldfail = True
    except:
      shouldfail = False   # there was no blahblah.SHOULDFAIL file 
    print file + " shouldfail ",
    print shouldfail

    #  for test that we know currently fail that should work
    knownfile = RESULTSDIR + file + ".KNOWNFAIL"
    knownfail = False
    print "checking for file " + knownfile 
    try:
      f = open(knownfile)  
      print file + " should work but currently is expected to fail"
      f.close()
      knownfail = True
    except:
      knownfail = False  # there was no blahblah.KNOWNFAIL file 
    print file + " knownfail ",
    print knownfail
      


    # find the source and dest lines in the file  (dest may not exist)
    count = 1
    for line in lines:
       #print "%d: '%s'" % (count, line)
       count += 1
       if "#" in line:
         # remove stuff after the pound sign
         line = line.split("#")[0]
           
       if "source:" in line:
         #print line,
         parts = line.split(":")
         #print parts
         src = parts[1].strip()
         print "source '" + src + "'"
       if "dest:" in line:
         #print line,
         parts = line.split(":")
         #print parts
         dest = parts[1].strip()
         print "dest   '" + dest + "'"

    if  src != "":   # OK so far (there is a source file) 
       if dest == "": # src but no dest, make up something
         parts = src.split(".")
         parts.insert( 1, "BUH" )
         dest = string.replace( src, parts[0], parts[0]+"modified", 1)
         print "dest   '" + dest + "'"
           
    print "\n"


    if src == "": # beets
      result =  "file %-20s FAILED    doesn't have a source?\n" % file
      resultsfailed.append( result )
      continue

    sys.stdout.flush()
    
    # check what time it is before running 
    before = time.time()  
    #print "before  ",
    #print before

    # run the chill script 
    command = "../../../chill " + file
    print command
    status = 0
    status = os.system( command )
    
    # 
    after = time.time()
    #print "after  ",
    #print after
    
    # see if the status says it worked 
    if status != 0:
      if shouldfail:
         print "file %s failed and should have" % file
         result = "file %-20s SUCCEEDED (it FAILED but was supposed to! returned statu%d)" % (file, status)
         resultssucceeded.append( result )
      else:
         print "file %s failed with status %d" % ( file, status )
         sys.stdout.flush()

         if knownfail:
            print "file %s failed but it's a bug we know about" % file
            result = "file %-20s FAILED    (it's a bug we know about)" % file
            resultsfailed.append( result )
         else:
            result = "file %-20s FAILED    (returned status %d)" % (file, status)
            resultsfailed.append( result )
         
      sys.stdout.flush()
      continue
    #else:
    #  print file + " succeeded?   status " + str(status)
    sys.stdout.flush()  

    

    # verify 
    # see if there is a recent version of the output file
    print "checking to see if %s exists" % dest
    destexists = False
    destrecent = False
    try:
      statstruct = os.stat( dest )
      destexists = True
      print "%s exists and has %d bytes" % ( dest, statstruct.st_size )
       
      modified = statstruct.st_mtime
      print modified

      elapsed = abs(after - modified)
      print "elapsed %f\n" % elapsed
      
      if elapsed > 3.0:  # output is NOT recent  ( 3 seconds? )
        destrecent = False # too old
      else:
        destrecent = True  # exists and just got created 
    except:
      destexists = False


    #see if there is a known good output 
    # find the known good file
    knowngood = "RIGHTANSWERS/" + dest
    print "knowngood " + knowngood
          
    # make sure the file exists
    print "checking for knowngood %s" % knowngood
    goodexists = False
    try:
      k = open( knowngood, 'r' )
      goodexists = True
    except:
      goodexists = False


    if destexists and destrecent:
       if goodexists:
         # compare with known good output
         # TODO needs to be smart about whitespace
         klines = k.readlines()
         k.close()

         f = open( dest, 'r' )
         lines = f.readlines()
         f.close

         #print "%d lines vs %d lines" % ( len(lines), len(klines) )

         same = comparelines( lines, klines)
         
         if not same:
           if knownfail:  # failed, but not a surprise. 
             result = "file %-20s FAILED    (we know that there's a bug)" % file
             resultsfailed.append( result )
           else: 
             result = "file %-20s FAILED    output does not match knowngood" % file
             resultsfailed.append( result )
         else:  # IS the same 
           print "%s is the same as %s\n\n" % ( dest, knowngood )
           result = "file %-20s SUCCEEDED" % file 
           resultssucceeded.append( result )
       else: # known good does not exist
         print "missing known good file " + knowngood
         if knownfail:  # failed, but not a surprise. 
          result = "file %-20s FAILED    (we know that there's a bug)" % file
         else:
           result = "file %-20s FAILED    there is no known good output %s to compare to" % (file, knowngood ) 
         resultsfailed.append( result )
    else:
      print "there is no dest file " + dest # ??

      # see if this one SHOULD fail (fail before creating an output) 
      if shouldfail:
         result = "file %-20s SUCCEEDED   (it FAILED but was supposed to! returned stas %d)" % (file, status)
         resultssucceeded.append( result )
      else:
        if knownfail:  # failed, but not a surprise. 
          result = "file %-20s FAILED    (we know that there's a bug)" % file
          resultsfailed.append( result )
          
        else: 
          result = "file %-20s FAILED    no output file %s was generated ???" % (file, dest )
          resultsfailed.append( result )

    numfiles     = len(files)
    numsucceeded = len(resultssucceeded)
    numfailed    = len(resultsfailed)
    
    for result in resultssucceeded:
      print result

    print ""
    for result in resultsfailed:
      print result

    print "\n%d / %d passed" % ( numsucceeded, numfiles )
    print "%d / %d failed" % ( numfailed   , numfiles )
    if numfiles != numsucceeded + numfailed:
      print "something is wrong. the totals do not add up"
    
if __name__ == "__main__":
  regressionTest()

