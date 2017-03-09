#! /usr/bin/python

# THIS IS CUDAIZE.PY

import chill
import sys
import math 

strided = 0
counted = 1

def print_code():
    chill.print_code()
    print ""
    sys.stdout.flush()

    
def table_contains_key( table, key ):  # use a dict for the 'table'?
    return table.has_key(key) # (key in table)?

def print_array( arr ):  # a useful function to mimic lua output 
    for a in arr[:-1]:
        print "%s," % a,
    print "%s" % arr[-1]
    sys.stdout.flush()

def valid_indices( statement, indices ):
    #print "valid_indices() python calling C cur_indices"
    #print statement
    cur = chill.cur_indices(statement) # calls C
    #print "python valid_indices(), cur = ",
    #print cur
    #print "indices = ",
    #print indices

    for index in indices:
        if not index in cur:
            return False
    return True

def next_clean_level( indices_at_each_level, level):
    #print "next_clean_level( ..., %d )" % level 
    #print "indices_at_each_level ",
    print_array( indices_at_each_level )

    numlevels = len(indices_at_each_level)
    #print "loop to %d" % numlevels
    for i in range(level+1, numlevels+1):
        pythoni = i-1 # LUA index starts at 1
        #print "Checking level %d = '%s'" % (i, indices_at_each_level[pythoni])
        sys.stdout.flush()
        if len(indices_at_each_level[pythoni]) > 0: # LUA INDEX STARTS AT 1
            #print "returning %d" % i
            return i  # MATCH lua return value, LUA index starts at one
    return -1  # no non-dummy indices




def build_order(  final_order, tile_index_names, control_index_names, tile_index_map, current_level):
    order = []   
    #print "\nbuild_order()"
    #print "build_order(): final_order = (",
    count = 0
    for f in final_order:
        #if count+1 == len(final_order):
        #    print "%s )" % f
        #else:
        #    print "%s," % f ,
        count += 1

        keys = control_index_names.keys()
        keys.sort()
        #if (2 == len(keys)):
        #    print "build_order(): ctrl_idx_names = (%s, %s)" % (control_index_names[0], control_index_names[1])
        #else:
        #    print "build_order(): ctrl_idx_names = (%s" % control_index_names[0],
        #    for k in keys[1:]:
        #        print ", %s" % control_index_names[k],
        #    print ")"

    #print control_index_names
    #print "cur_level %d" % current_level
    
    #print "tile index map: ",
    #print tile_index_map


    for i in range(len(final_order)):
        k = final_order[i]  # not used?
        skip = False
        cur = final_order[i]  
        # control loops below our current level should not be in the current order

        # skip = cur in control_index_names[current_level+2:] 
        #print "\n%d control_index_names, " % len(control_index_names)
        #print control_index_names

        for j in range(current_level+1, len(control_index_names)):
            #print "comparing cur %s with cin[%d] %s" % ( cur, j, control_index_names[j])
            if control_index_names[j] == cur:
                skip = True 
                #print "SKIP %s  " % cur

        # possibly substitute tile indices if necessary
        if tile_index_map.has_key(cur):
            approved_sub = False
            sub_string = tile_index_map[cur]
            #print "sub_string = ",
            #print sub_string

            # approved_sub = sub_string in tile_index_names[current_level+2:]
            for j in range(current_level+1, len(tile_index_names)):
                if tile_index_names[j] == sub_string:
                    approved_sub = True
            if approved_sub:
                cur = sub_string

        if not skip:
            order.append( cur)  
    #print "build_order() returning order (",
    #print order
    #for o in order:
    #    print "%s," % o,
    #print ")"
    return order

def find_cur_level( stmt, idx ):
    #print "find_cur_level(stmt %d, idx %s)  Cur indices" % ( stmt, idx ),
    
    cur = chill.cur_indices(stmt)
    #for c in cur[:-1]:
    #    print "%s," % c,
    #print "%s" % cur[ -1 ] 

    index = 1 # lua starts indices at 1 !!  
    for c in cur:
        if c == idx:
            #print "found it at index %d" % index
            #sys.stdout.flush()
            #print "in find_cur_level, returning ",
            #print index
            return index
        index += 1
    #print "find_cur_level(), Unable to find index %s in" % idx,
    #print cur
    #print "in find_cur_level, returning -1"
    return -1  # special meaning "it's not there"

def chk_cur_level( stmt, idx ):
    # search cur_indices for a ind at stmt
    cur = chill.cur_indices(stmt)
    if idx in cur:
       return 1 + cur.index(idx)  # lua index starts at 1 !
    return -1

def find_offset( cur_order, tile, control):
    #print "Looking for tile '%s' and control '%s' in (" % (tile, control),
    #print cur_order
    #for o in cur_order:
    #    print "%s," % o,
    #print ")"

    idx1 = -1
    idx2 = -1
    if tile in cur_order: 
        idx1 = 1 + cur_order.index(tile) # lua indexes from 1!
    else:
        print "find_offset(), unable to find tile %s in current list of indices" % tile
        sys.exit(-1)

    if control in cur_order:
        idx2 = 1 + cur_order.index(control) # lua indexes from 1!
    else:
        print "find_offset(), unable to find control %s in current list of indices" % control
        sys.exit(-1)

    #print "found at level %d and %d" % ( idx2, idx1 )
    # this appears horrible
    if idx2 < idx1:
        return idx2-idx1+1 # bad ordering
    else:
        return idx2-idx1



def tile_by_index( tile_indices, sizes, index_names, final_order, tile_method):
    #print "STARTING TILE BY INDEX"
    #print "tile_by_index() tile_method ",
    #print tile_method
    #print "index_names: ",
    #print index_names

    stmt = 0 # assume statement 0
    if not valid_indices( stmt, tile_indices):
        print "python tile_by_index() one or more of ",
        print tile_indices,
        print " is not valid"
        sys.exit(-1)

    if tile_method == None:
        #print "CREATING tile_method = 1"
        tile_method = 1 # "counted"

    tile_index_names = []
    for ti in tile_indices:
        tile_index_names.append( ti )  # make a copy? 
    #print "tile_index_names:",
    #print tile_index_names

    control_index_names = dict()
    tile_index_map = dict()
    
    #print "index_names: "
    #print index_names

    for control, name in index_names.items():
        valid = False
        
        if control[0] == "l" and control[1].isdigit():
            if control.endswith("_control"):
                index = int(control[1: -8])
                control_index_names[index-1] = name
                valid = True

            elif control.endswith("_tile"):
                index = int(control[1: -5])
                #print "index %d" % index
                tile_index_names[index-1] = name # ?? 
                tile_index_map[name] = tile_indices[index-1]
                valid = True
        if not valid:
            print "%s is not a proper key for specifying tile or control loop indices\n" % control

    #print "control_index_names = ",
    #print control_index_names

    #print "tile_index_names = ",
    #print tile_index_names

    #print "before call to build_order(), tile_index_map = ",
    #print tile_index_map


    # filter out control indices (and do name substitution of unprocessed tile indices) for a given level
    cur_order = build_order(final_order, tile_indices, control_index_names, tile_index_map, -1)

    #print "returned from build_order python\n\n"

    # print("permute("..stmt..", {"..list_to_string(cur_order).."})")
    #print "permute(%d, {" % stmt,
    #print "cur_order = ",
    #print cur_order,
    #print "})"

    print cur_order
    chill.permute(stmt, list(cur_order)) 
    #print "in cudaize.py, returned from C code chill.permute()\n"

    for i in range(len(tile_indices)):
        cur_idx = tile_indices[i]
        #print "i %d  cur_idx %s calling build order ********" % (i, cur_idx)
        cur_order = build_order( final_order, tile_indices, control_index_names, tile_index_map, i)
        #print "cur_idx %s return from build order" % cur_idx
        
        # Find an offset between tile loop and control loop
        #  0   = control loop one level above tile loop
        #  -1  = control loop two levels above tile loop
        #  > 0 = tile loop above control loop
        #  In the last case, we do two extra tile commands to get the control
        #  above the tile and then rely on the final permute to handle the
        #  rest
        level = find_cur_level(stmt,cur_idx)
        #print "level %d\n" % level     

        offset = find_offset(cur_order, tile_index_names[i], control_index_names[i])
        #print "offset %d" % offset

        if offset <= 0:
            #print "[offset<=0]1tile(%d, %d, %d, %d, %s, %s, %d)" % (stmt, level, sizes[i], level+offset, tile_index_names[i], control_index_names[i], tile_method  )
            chill.tile7( stmt, level, sizes[i], level+offset, tile_index_names[i], control_index_names[i], tile_method  )
            #print "in cudaize.py, returned from C code chill.tile7\n"

        else:
            #print "2tile(%d, %d, %d, %d, %s, %s, %d)" % (stmt, level, sizes[i], level+offset-1, tile_index_names[i], control_index_names[i], tile_method  )
            chill.tile7( stmt, level, sizes[i], level+offset-1, tile_index_names[i], control_index_names[i], tile_method  ) # regular level

            # flip and tile control loop
            #print "3tile(%d, %d, %d)" % ( stmt, level+1, level+1)
            chill.tile3( stmt, level+1, level+1)

            #print "4tile(%d, %d, %d)" % ( stmt, level+1, level)
            chill.tile3( stmt, level+1, level)

            #print_code()

        # Do permutation based on cur_order
        #print("permute based on build order calling build_order()")
        cur_order = build_order(final_order, tile_indices, control_index_names, tile_index_map, i)

        #print("permute based on build order return from build_order()")

        #  print("permute("..stmt..", {"..list_to_string(cur_order).."})")
        topermute = cur_order
        chill.permute(stmt, list(topermute)) 
        #print "\nafter permute(), code is:"
        #print_code()

def normalize_index( index ):
    #print "in cudaize.py, normalize_index( %s )" % index
    stmt = 0  # assume stmt 0
    l = find_cur_level( stmt, index )
    chill.tile3( stmt, l, l )

def is_in_indices( stmt, idx):
    cur = chill.cur_indices(stmt)
    return idx in cur

def copy_to_registers( start_loop, array_name ):
    #print "\n\n****** starting copy to registers"
    #sys.stdout.flush()

    stmt = 0    # assume stmt 0
    cur = chill.cur_indices(stmt) # calls C    
    table_Size = len(cur)

    #print "Cur indices",
    #print_array(cur)
    #print "\nThe table size is %d" % table_Size
    #count=1
    #for c in cur:
    #    print "%d\t%s" % (count,c)
    #    count += 1

    #print_code()

    # would be much cleaner if not translating this code from lua!
    level_tx = -1
    level_ty = -1   
    if is_in_indices(stmt,"tx"):
        level_tx = find_cur_level(stmt,"tx")
    if is_in_indices(stmt,"ty"):
        level_ty = find_cur_level(stmt,"ty")
    #print "level_tx %d  level_ty %d" % ( level_tx, level_ty )
    #sys.stdout.flush()

    ty_lookup_idx = "" 
    org_level_ty = level_ty

    # UGLY logic. Lua index starts at 1, so all tests etc here are off by 1 from the lua code
    # level_ty initializes to -1 , which is not a valid index, and so there is added code to 
    # make it not try to acccess offset -1.   -1 IS a valid python array index
    # to top it off, the else below can assign a NIL to ty_lookup_idx! 
    if level_ty != -1 and cur[level_ty] != "":
        #print "IF  cur[%d] = %s" % ( level_ty, cur[level_ty] )
        ty_lookup_idx = cur[level_ty] 
    else:
        #print "ELSE ty_lookup_idx = cur[%d] = %s" % ( level_ty, cur[level_ty-1]) 
        ty_lookup_idx = cur[level_ty-1] 
    #print "ty_lookup_idx '%s'" % ty_lookup_idx

    if level_ty > -1:
        #print "\ntile3(%d,%d,%d)" % (stmt,level_ty,level_tx+1)
        chill.tile3(stmt,level_ty,level_tx+1) 
    #print_code()   

    cur = chill.cur_indices(stmt) # calls C 
    table_Size = len(cur)
    #print "Cur indices ",
    #for c in cur:
    #    print "%s," % c,
    #print "\nThe table size is %d" % len(cur)
    #count=1
    #for c in cur:
    #    print "%d\t%s" % (count,c)
    #    count += 1
    #sys.stdout.flush()

    if is_in_indices(stmt,"tx"):
        level_tx = find_cur_level(stmt,"tx")
    if ty_lookup_idx != "":                      # perhaps incorrect test 
        if is_in_indices(stmt,ty_lookup_idx):
           level_ty = find_cur_level(stmt,ty_lookup_idx)
           
    ty_lookup = 1
    idx_flag = -1
    # find the level of the next valid index after ty+1
    #print "\nlevel_ty %d" % level_ty
    if level_ty > -1:
       #print "table_Size %d" % table_Size
       for num in range(-1 + level_ty+ty_lookup,table_Size):   # ??  off by one?
           #print "num=%d   cur[num] = '%s'" % (num+1, cur[num]) # num+1 is lua index ????
           sys.stdout.flush()
           if cur[num] != "":
               idx_flag = find_cur_level(stmt,cur[num])
               #print "idx_flag = %d" % idx_flag
               break
               
    #print "\n(first) I am checking all indexes after ty+1 %s" % idx_flag
    #print_code()   
    #print "" 

    how_many_levels = 1
    
    #print "idx_flag = %d   I will check levels starting with %d" % (idx_flag, idx_flag+1)
    # lua arrays start at index 1. the next loop in lua starts at offset 0, since idx_flag can be -1
    # thus the check for "not equal nil" in lua (bad idea)
    # python arrays start at 0, so will check for things that lua doesn't (?)
    startat = idx_flag + 1
    if idx_flag == -1:
        startat = 1  # pretend we're lua for now.   TODO: fix the logic

    for ch_lev in range(startat,table_Size+1):       # logic may be wrong (off by one)
        #print "ch_lev %d" % ch_lev
        if ch_lev <= table_Size and cur[ch_lev-1] != "":
           #print "cur[%d] = '%s'" % ( ch_lev, cur[ch_lev-1] )
           how_many_levels += 1

    #print "\nHow Many Levels %d" % how_many_levels
    sys.stdout.flush()
    sys.stdout.flush()

    if how_many_levels< 2:
        while( idx_flag >= 0):
            for num in range(level_ty+ty_lookup,table_Size+1):
                #print "at top of loop, num is %d" % num
                #print "cur[num] = '%s'" % cur[num-1]
                if cur[num-1] != "":
                    idx = cur[num-1]
                    #print "idx '%s'" % idx
                    sys.stdout.flush()
                    curlev = find_cur_level(stmt,idx)
                    #print "curlev %d" % curlev

                    #print "\n[COPYTOREG]tile(%d,%d,%d)"%(stmt,curlev,level_tx)

                    chill.tile3(stmt, curlev, curlev)
                    curlev = find_cur_level(stmt,idx)
                    #print "curlev %d" % curlev
                    chill.tile3(stmt,curlev,level_tx)
                    #print "hehe '%s'" % cur[num-1]
                    
                    cur = chill.cur_indices(stmt)
                    #print "Cur indices INSIDE",
                    #for c in cur:
                    #    print "%s," % c,
                    table_Size = len(cur)
                    #print "\nTable Size is: %d" % len(cur)

                    level_tx = find_cur_level(stmt,"tx")
                    #print "\n level TX is: %d" % level_tx
                    level_ty = find_cur_level(stmt,ty_lookup_idx)
                    #print "\n level TY is: %d" %level_ty
                    idx_flag = -1
                    #print "idx_flag = -1"


                    #- find the level of the next valid index after ty+1
                    #- the following was num, which conflicts with loop we're already in, and otherwise wasn't used (?)
                    for num2 in range( -1 + level_ty+ty_lookup ,table_Size): # lua starts index at one
                        #print "num mucking num = %d" % num2
                        if(cur[num2] != ""):
                            #print "cur[%d] = '%s'" % ( num2, cur[num2] )
                            idx_flag = find_cur_level(stmt,cur[num2])
                            #print("\n(second) I am checking all indexes after ty+1 %s",cur[num2])
                            break

                    #print "num mucked to %d     idx_flag = %d" % (num, idx_flag)

                #print "at bottom of loop, num is %d" % num
          
    #print "done with levels"

    # this was a block comment ???

#    for num in range(level_ty+1, table_Size+1):
#        print "num %d" % num
#        if cur[num-1] != "":
#            idx_flag = find_cur_level(stmt,cur[num-1])  ## ugly 
#    print "idx_flag = %d" % idx_flag

    # change this all to reflect the real logic which is to normalize all loops inside the thread loops. 
#    print "change this all ...\n"
#    print "level_ty+1 %d  table_Size-1 %d     idx_flag %d" %( level_ty+1, table_Size-1, idx_flag)
#    sys.stdout.flush()
#    sys.stdout.flush()

#    while level_ty+1 < (table_Size-1) and idx_flag >= 0:
#        print "*** level_ty %d" %  level_ty
#        for num in range(level_ty+2,table_Size+1):  # lua for includes second value
#            print "num %d   cur[num] %s" % (num, cur[num])
#            if cur[num] != "":
#                idx = cur[num]
#                print "idx='%s'" % idx
#                #print_code()
                
                
            

    #print "ARE WE SYNCED HERE?"
    #print_code()

    #  [Malik] end logic
    start_level = find_cur_level(stmt, start_loop) # start_loop was passed parameter!

    # We should hold constant any block or tile loop
    block_idxs  = chill.block_indices()
    thread_idxs = chill.thread_indices()
    #print"\nblock indices are"
    #for index, val in enumerate(block_idxs):
    #    print "%d\t%s" % ( int(index)+1 , val )
    #print"\nthread indices are"
    #for index, val in enumerate(thread_idxs):
    #    print "%d\t%s" % ( int(index)+1 , val )
    #print "\nStart Level: %d" % start_level

    hold_constant = []
    #print("\n Now in Blocks")
    for idx in block_idxs:
        blocklevel = find_cur_level(stmt,idx)
        if blocklevel >= start_level:
           hold_constant.append(idx)
           #print "\nJust inserted block %s in hold_constant" %idx

    #print("\n Now in Threads")
    for idx in thread_idxs:
        blocklevel = find_cur_level(stmt,idx)
        if blocklevel >= start_level:
            hold_constant.append(idx)
            #print "\nJust inserted thread %s in hold_constant" %idx
    #print "\nhold constant table is: "
    #for index, val in enumerate(hold_constant):
    #    print "%d\t%s" % ( int(index)+1 , val )
    
    #print("\nbefore datacopy pvt")
    old_num_stmts = chill.num_statements()
    #sys.stdout.flush()

    #print "\n[DataCopy]datacopy_privatized(%d, %s, %s, " % (stmt, start_loop, array_name),
    #print hold_constant,
    #print ")"
    passtoC = [stmt, start_loop, array_name ] # a list
    passtoC.append( len(hold_constant ) )
    for h in hold_constant:
        passtoC.append( h )
    chill.datacopy_privatized( tuple( passtoC ))
    sys.stdout.flush()
    sys.stdout.flush()
    
    new_num_statements = chill.num_statements()
    #print "new num statements %d" % new_num_statements    

    # Unroll to the last thread level
#    for stmt in range(old_num_statements, new_num_statements):
#        print "unrolling statement %d" % stmt
#        level = find_cur_level(stmt,thread_idxs[-1]) #get last thread level
#        print "level is %d" % level
#        idxs = chill.cur_indices(stmt)
#        if level < len(idxs):
#            chill.unroll(stmt,level+1,0)



def copy_to_shared( start_loop, array_name, alignment ):
    #print "\nstarting copy to shared( %s, %s, %d)" % (start_loop, array_name, alignment ) 
    #print "copy_to_shared( %s, %s, %d) in cudaize.py" % ( start_loop, array_name, alignment )
    stmt = 0 # assume statement 0

    cur = chill.cur_indices(stmt)
    #print "Cur indices ",
    #print_array( cur )

    start_level = find_cur_level( stmt, start_loop )
    #print "start_level %d" % start_level

    old_num_statements = chill.num_statements()
    #print "old_num_statements %d" % old_num_statements
    

    # Now, we give it indices for up to two dimensions for copy loop
    copy_loop_idxs = ["tmp1","tmp2"]
    #chill.datacopy_9arg(stmt, start_level, array_name, copy_loop_idxs, False, 0, 1, alignment,True)
    passtoC = [stmt, start_level, array_name]   # a list
    passtoC.append( len(copy_loop_idxs))
    for i in copy_loop_idxs:
        passtoC.append(i)
    passtoC.append( 0 ) # False
    passtoC.append( 0 )
    passtoC.append( 1 )
    passtoC.append( alignment )
    passtoC.append( 1 )   # True
    #print "\n[DataCopy]datacopy( ",
    #print passtoC,
    #print ")"

    #if array_name == "b":
    #    chill.cheat(1)
    #if array_name == "c":
    #    chill.cheat(2)
    
    chill.datacopy_9arg( tuple( passtoC ))

    #print "back from datacopy_9arg\n\n\n"
    #sys.stdout.flush()


    #print "calling add_sync( %d, %s )" % ( stmt, start_loop )
    chill.add_sync( stmt, start_loop )
    #print "back from add_sync()\n\n"

    new_num_statements = chill.num_statements()
    
    #  This is fairly CUBLAS2 specific, not sure how well it generalizes,
    #  but for a 2D copy, what we want to do is "normalize" the first loop
    #  "tmp1" then get its hard upper bound. We then want to tile it to
    #  make the control loop of that tile "ty". We then tile "tmp2" with a
    #  size of 1 and make it "tx".

    #print "fairly CUBLAS2 specific, OLD %d  NEW %d" % ( old_num_statements, new_num_statements)
    sys.stdout.flush()
    sys.stdout.flush()

    for stmt in range(old_num_statements, new_num_statements):
        #print "for stmt = %d" % stmt
        level = find_cur_level( stmt, "tmp2")
        #print "FOUND CUR LEVEL?  level '",
        #print level,
        #print "'"

        #print "in loop, stmt %d   level %d" % ( stmt, level )
        if level != -1:
            #print "\nCopy to shared: [If was no error]\n"
            find_cur_level(stmt,"tmp2")
            chill.tile3( stmt, level, level )
            
            #print "hard_loop_bounds( %d, %d )" % (stmt, level)
            bounds = chill.hard_loop_bounds(stmt, level)
            lower = bounds[0]
            upper = 1+ bounds[1]
            #print "lower %d  upper %d" % ( lower, upper )

            dims = chill.thread_dims()
            #print "in cudaize.py copy_to_shared, dims =",
            #print dims
            tx = dims[0]
            ty = dims[1]
            #print "2-loop cleanup: lower, upper: %d, %d,  tx: %d" % ( lower, upper, tx)

            level = find_cur_level(stmt,"tmp1")
            #print "level %d" % level
            if tx == upper and ty == 1:
                #print "tx = %d    upper = %d     ty = %d"% (tx, upper, ty)
                #print "Don't need"

                # Don't need an extra tile level, just move this loop up
                second_level = find_cur_level(stmt,"tmp2")
                chill.tile7(stmt, second_level, 1, level, "tx", "tx", counted)

            else:
                #print "DO need?"
                if ty == 1:
                    new_ctrl = "tmp3" 
                else:
                    new_ctrl = "ty"

                # LOTS of commented out code here in cudaize.lua 

                #print_code()
                #print "\nStarting tmp2\n"
                first_level  = find_cur_level(stmt,"tmp1")
                second_level = find_cur_level(stmt,"tmp2")
                bounds = chill.hard_loop_bounds(stmt, second_level)
                lower = bounds[0]
                upper = 1 + bounds[1]   # BROKEN?
                        
                #print "[Malik]-loop cleanup@tmp2: lower, upper: %d, %d, tx: %d,first level:%d,second_level:%d" % ( lower, upper-1, tx, first_level, second_level) 

                # Move the fastest changing dimension loop to the outermost,identified by "tmp2" and to be identified as tx.
                #print "\n[fastest]tile(%d, %d, %d,%d,%s,%s,counted)"%(stmt, second_level,1,first_level, "tx", "tx")
                chill.tile7(stmt, second_level,1,first_level,"tx","tx",counted)
                #print_code()

                first_level = find_cur_level(stmt,"tmp1")
                bounds = chill.hard_loop_bounds(stmt, first_level)
                lower_1 =     bounds[0]
                upper_1 = 1 + bounds[1]
                tx_level = find_cur_level(stmt,"tx")
                bounds = chill.hard_loop_bounds(stmt,tx_level)
                lower_tx =   bounds[0]
                upper_tx = 1+bounds[1]
                #print "UL_1 %d %d     UL_tx %d %d" % ( lower_1, upper_1-1, lower_tx, upper_tx-1)

                if int(math.ceil( float(upper_tx)/float(tx))) > 1:
                     #print "ceil I say"
                     #print "\n[Tile1]tile(%d, %d, %d,%d,%s,%s,counted)" % (stmt, tx_level,tx,tx_level, "tx", "tmp1")
                     chill.tile7(stmt,tx_level,tx,tx_level,"tx","tmp_tx",counted)
                     #print_code()

                     repeat = find_cur_level(stmt,"tx")
                     #print "\n[Tile1]tile(%d, %d, %d)" % (stmt, repeat, repeat)
                     chill.tile3(stmt, repeat, repeat)  #find_cur_level(stmt,"tx"),find_cur_level(stmt,"tx"))
                     #print_code()

                     if find_cur_level(stmt,"tx")>find_cur_level(stmt,"tmp_tx"):
                        #print "\nagain [Tile1]tile(%d, %d, %d)" % (stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp_tx"))
                        chill.tile3(stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp_tx"))
                        #print_code()

                #print_code()

                #print "\nStarting tmp1\n"
                # Handle the other slower changing dimension, the original outermost loop, now identified by "tmp1", to be identified as "ty".
                chill.tile3(stmt,find_cur_level(stmt,"tmp1"),find_cur_level(stmt,"tmp1"))      
                #print_code()

                ty_level = find_cur_level(stmt,"tmp1")
                bounds = chill.hard_loop_bounds(stmt,ty_level)
                lower_ty = bounds[0]
                upper_ty = 1 + bounds[1]

                tx_level = find_cur_level(stmt,"tx")
                bounds = chill.hard_loop_bounds(stmt,tx_level)
                lower_tx = bounds[0]
                upper_tx = 1 + bounds[1]

                #print "[Malik]-loop cleanup@tmp1: lowerty, upperty: %d, %d, ty: %d,ty level:%d,tx_level:%d, stmt: %d" % ( lower_ty, upper_ty-1, ty, ty_level, tx_level, stmt)
                
                #print "before ceil"
                #sys.stdout.flush()

                if(math.ceil(float(upper_ty)/float(ty)) > 1):
                    #print "CEIL IF"
                    #print "\n Inside upper_ty/ty > 1\n"

                    #print "\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)"%(stmt, ty_level,ty,ty_level, "ty", "tmp_ty")
                    chill.tile7(stmt,ty_level,ty,ty_level,"ty","tmp_ty",counted)
                    #print_code()

                    #print "\n[Tile2-1]tile(%d, %d, %d)"%(stmt,find_cur_level(stmt  ,"ty"),find_cur_level(stmt,"ty"))
                    chill.tile3(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"ty"))
                    #print_code()

                    cur_idxs = chill.cur_indices(stmt)
                    #print "\n cur indexes are ",
                    #print_array( cur_idxs)
                    #sys.stdout.flush()

                    # Putting ty before any tmp_tx
                    idx_flag = -1
                    if "tmp_tx" in cur_idxs:
                        idx_flag = 1 + cur_idxs.index("tmp_tx")   # lua index starts at 1
                    #print "\n (1) so i have found out the value of idx flag as %d" % idx_flag
                    #sys.stdout.flush()      
                    
                    if idx_flag >= 0:
                         if find_cur_level(stmt,"ty") > find_cur_level(stmt,"tmp_ty"):
                             #print "\n[Tile2-2]tile(%d, %d, %d)"%(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                             chill.tile3(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                             #print_code()
                    
                    
                    #  Now Putting ty before any tmp_ty
                    sys.stdout.flush()      
                    idx_flag = -1
                    if "tmp_ty" in cur_idxs:
                        idx_flag = 1 + cur_idxs.index("tmp_ty") # lua index starts at 1
                    #print "\n IF  so i have found out the value of idx flag as %d" % idx_flag
                    #sys.stdout.flush()      
                                            
                    if idx_flag >= 0:
                        #print "one more test"
                        sys.stdout.flush()
                        if find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty"):
                            #print "\n[Tile2-2]tile(%d, %d, %d)"%(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                            #sys.stdout.flush()
                            chill.tile3(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                            #print_code()



                else:
                    #print "CEIL ELSE"
                    #print "\n[Tile3]tile(%d, %d, %d,%d,%s,%s,counted)" % (stmt, ty_level,1,ty_level, "ty", "ty")
                    #sys.stdout.flush()
                    chill.tile7( stmt, ty_level, 1, ty_level, "ty", "ty", counted )
                    #print_code()

                    #print "\n[Tile3-1]tile(%d, %d, %d)"%(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1)
                    sys.stdout.flush()

                    chill.tile3(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1)
                    #print_code()


                    idx_flag = -1
                    # LUA code checks to see if cur_idxs exists?  it is unused except in the other clause of this is
                    #if(cur_idxs) then
                        #print "CAN NEVER GET HERE?  cur_idxs"
                        #for num= 0,table.getn(cur_idxs) do
                            #if(cur[num] == "tmp_ty") then
                            #idx_flag = find_cur_level(stmt,cur[num])
                            #break
                        #end
                    #end
                    print "\n ELSE so i have found out the value of idx flag as %d" % idx_flag
                    if idx_flag >= 0:  # can't happen
                        print "tile( stmt %d, level ty %d, level ty %d" % ( stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                        #chill.tile3(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                    
                        
                    

                    
            #print "\n\n *** at bottom of if in copy to shared, "
            #print_code()
            #print "end of if"

        else:
            #  copy to shared only created one level, not two, so we use a different approach (MV & TMV)
            #print "\nCopy to shared: [If was error]\n"
            level = find_cur_level(stmt,"tmp1")
            chill.tile3(stmt, level, level)

            dims = chill.thread_dims()
            #print dims
            tx = dims[0]
            ty = dims[1]

            bounds = chill.hard_loop_bounds(stmt, level)
            lower = bounds[0]   
            upper = bounds[1]

            #print "bounds  lower %d    upper %d" % (lower, upper)
            upper = upper+1 # upper bound given as <=, compare to dimensions tx which is <
            if upper == tx:
                #print "upper == tx"
                chill.rename_index( stmt, "tmp1", "tx")
            else:
                #print "upper is not tx"
                #print "upper %d tx %d stmt: %d level: %d" % ( upper, tx, stmt, level)
                chill.tile7( stmt, level, tx, level, "tx", "tmp_tx", counted)
                #print_code()

                #print "stmt:%d level+1: %d" % ( stmt, level+1) 
                #print("TILE 7")
                chill.tile7( stmt, level+1,1,level+1,"tx", "tx",counted)
                #print("TILE 3")
                chill.tile3( stmt, level+1, level)
                #print_code()           


                if ty > 1:
                   #print "GOING IN"
                   bounds = chill.hard_loop_bounds(stmt, level+1)
                   lower = bounds[0]   
                   upper = bounds[1]   
                   #print "ty %d  lower %d  upper %d" % ( ty, lower, upper )
                   floatdiv = float(upper)/float(ty)
                   bound =  int(math.ceil(float(upper)/float(ty)))
                   #print "NOW FOR Y: upper %d ty %d stmt: %d level: %d bound: %d" % ( upper, ty, stmt, level+1,   bound)
                   chill.tile7(stmt, level+1, bound, level+1, "tmp_ty", "ty", counted)

        # Always add sync
        chill.add_sync( stmt, start_loop )
    #print "ending copy to shared\n"
    #sys.stdout.flush()
    #print_code()     



















def unroll_to_depth( max_depth ):
    print "\n\nunroll_to_depth(%d)" % max_depth
    print "SYNC UP"
    sys.stdout.flush()

    cur = chill.cur_indices(0)
    thread_idxs = chill.thread_indices()
    guard_idx = thread_idxs[-1]  # last one

    print "cur    indices",
    print_array(cur)
    print "thread indices", 
    print_array(thread_idxs)
    print "guard_idx = %s" % guard_idx

    #print "thread_idxs = ",
    #print thread_idxs
    guard_idx = thread_idxs[-1]
    #print "guard_idx = %s" % guard_idx

    #  HERE FIND OUT THE LOOPS WHICH ARE COMMON BETWEEN STATEMENTS
    common_loops = []
    comm_loops_cnt = 0
    num_stmts = chill.num_statements()
    print "num statements %d" % num_stmts

    for stmt in range(num_stmts):
        sys.stdout.flush()
        print "\nSTMT %d" % stmt,
        cur_idxs = chill.cur_indices(stmt)
        print "Current Indices:",
        for c in cur_idxs[:-1]:
            print "%s," % c,
        print "%s" % cur_idxs[-1]   # last one
        sys.stdout.flush()
        #print_code()
        
        if chk_cur_level(stmt, "tx") > 0:
            
            for ii in range(find_cur_level(stmt,"tx")-1):
                print "ii = %d\ncur_idxs[%d] = '%s'" % (ii+1, ii+1, cur_idxs[ii]) # print to match lua
                id = cur_idxs[ii]
                if id not in ["bx", "by", "", "tx", "ty"]:

                    print "id %s is not in the list" % id

                    for stmt1 in range(stmt+1, num_stmts):
                        print "\nii %d stmt1 is %d" % (ii+1, stmt1)  # print to match lua 
                        cur_idxs1 = chill.cur_indices(stmt1)
                        print "\nstmt1 cur_idxs1 is ",
                        for ind in cur_idxs1[:-1]:
                            print "%s," % ind,
                        print "%s" % cur_idxs1[-1]

                        print "cur level(%d, %s) = %d" % (stmt, "tx", find_cur_level(stmt,"tx") )
                        sys.stdout.flush()

                        endrange = find_cur_level(stmt,"tx")-1
                        print "for iii=1, %d do" % endrange
                        sys.stdout.flush()
                        for iii in range(endrange):   # off by one?  TODO 
                            print "stmt %d   ii %d   iii %d\n" % (stmt, ii+1, iii+1),
                            sys.stdout.flush()
                            
                            if iii >= len(cur_idxs1):
                                print "stmt %d   ii %d   iii %d  cur_idxs1[%d] = NIL" % (stmt, ii+1, iii+1, iii+1, )  # print to match lua 
                            else:
                                print "stmt %d   ii %d   iii %d  cur_idxs1[%d] = '%s'" % (stmt, ii+1, iii+1, iii+1, cur_idxs1[iii])  # print to match lua 
                            sys.stdout.flush()

                            # this will still probably die 
                            if iii < len(cur_idxs1) and [iii] not in ["bx", "by", "tx", "ty", ""]:
                                if cur_idxs[ii] == cur_idxs1[iii]:
                                    print "\nfound idx:%s" % cur_idxs[ii]
                                    common_loops.append(cur_idxs[ii])
                                    print "cl[%d] = '%s'" % ( comm_loops_cnt, cur_idxs[ii] )
                                    comm_loops_cnt = len(common_loops)

    if len(common_loops) > 0:
        print "\n COMM LOOPS :TOTAL %d, and are " % comm_loops_cnt,
        print common_loops, 
        print " this loop : %s" % common_loops[0]
    else:
        print "UNROLL can't unroll any loops?"


    while True:  # break at bottom of loop   (repeat in lua)
        old_num_statements = chill.num_statements()
        print "old_num_statements %d" % old_num_statements

        for stmt in range(old_num_statements):
            cur_idxs = chill.cur_indices(stmt)
            print "stmt %d    cur_idxs =" % stmt,
            index = 0
            for i in cur_idxs:
                index +=1
                if index == len(cur_idxs):
                    print "%s" %i
                else:
                    print "%s," % i,

            if len(cur_idxs) > 0:
                guard_level = -1
                if chk_cur_level(stmt, guard_idx) > 0:
                    guard_level = find_cur_level(stmt,guard_idx)
                print "guard_level(sp) = %d" % guard_level
                if guard_level > -1:
                    level = next_clean_level(cur_idxs,guard_level)
                    print "next clean level %d" % level

                    
                    #print "looking at %d" % stmt
                    #print "comparing %d and %d in" % (guard_level, level),
                    #index = 0
                    #for i in cur_idxs:
                    #index +=1
                    #if index == len(cur_idxs):
                    #    print "%s" %i
                    #else:
                    #    print "%s," % i,

                    # need to handle max_depth
                    num_unrolled = 0
                    level_unroll_comm = level
                    level_arr = []

                    #print "before while, level = %d" % level 
                    while level >= 0:
                        print "while: level = %d" % level 
                        if num_unrolled == max_depth:
                            break

                        print "Unrolling %d at level %d index %s" % ( stmt, level, cur_idxs[guard_level])  # ??? 
                        level_arr.append(level)

                        guard_level = find_cur_level(stmt,guard_idx)
                        level = next_clean_level(cur_idxs,level+1)

                    print "OK, NOW WE UNROLL"
                    if level_unroll_comm >= 0:
                        level_arr.reverse()  
                        for i,lev in enumerate(level_arr):
                            print "\ni=%d" % i
                            print "[Unroll]unroll(%d, %d, 0)" % (stmt, lev)
                            chill.unroll(stmt, lev, 0)


        new_num_statements = chill.num_statements()
        if old_num_statements == new_num_statements:
            break  # exit infinite loop


#  all other calls to C have a routine in this file   (?)
def unroll( statement, level, unroll_amount ):
    chill.unroll( statement, level, unroll_amount )

