#! /usr/bin/python

import glob
import string
import sys
from subprocess import call
import os
import time 

files = glob.glob("*.script")
#print files 
#files = files[0:1] # pick one

print "\n"
for file in files:
    print file
    f = open( file, "r" )
    lines = f.readlines()
    f.close()

    src = ""
    dest=""

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

    if  src != "":   # OK so far
        if dest == "": # src but no dest, make up something
           parts = src.split(".")
           parts.insert( 1, "BUH" )
           dest = string.replace( src, parts[0], parts[0]+"modified", 1)
           print "dest   '" + dest + "'"
           
    print "\n"


    if src == "": # beets
       print "file " + file + " doesn't have a source?\n" 
       sys.exit(-1)

    # run
    before = time.time()  
    print "before  ",
    print before

    command = "../../../chill " + file
    print command
    status = os.system( command )
    now = time.time()
    print "now  ",
    print now
    
    # see if the status say it worked 
    if status != 0:
       print file + " failed?   status " + status
       sys.exit(-1)
    else:
      print file + " succeeded?   status " + str(status)
      

    

    # verify
    # see if there is a recent version of the output file

    statstruct = os.stat( dest )
    if statstruct:
       print "%s exists and has %d bytes" % ( dest, statstruct.st_size )
       
       modified = statstruct.st_mtime
       print modified

       elapsed = abs(now - modified)
       print "elapsed %f\n" % elapsed

       if elapsed < 3.0:  # output is recent

          # compare with known good output
          # TODO needs to me smart abpout whitespace

          # find the known good file
          knowngood = "RIGHTANSWERS/" + dest
          print "knowngood " + knowngood
          
          # make sure the file exists
          k = open( knowngood, 'r' )

          if k == None: #
             print "file %s does not exist" % knowngood
             sys.exit(-1)

          else: # file exists, compare

                klines = k.readlines()
          k.close()

          f = open( dest, 'r' )
          lines = f.readlines()
          f.close

          print "%d lines vs %d lines" % ( len(lines), len(klines) )

          # TODO deal with whitespace

          if len(lines) == len(klines):
             n = len(lines)

             same = True
             for i in range(n):
                 l = lines[i]
                 k = klines[i]
                 if l != k:   # string compare
                    print "\nline %d" % i
                    print l,
                    print k
                    same = False


          if not same:
             exit(-1)

          print "%s is the same as %s\n\n" % ( dest, knowngood )
          

