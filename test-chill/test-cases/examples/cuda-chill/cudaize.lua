
-- THIS IS CUDAIZE.LUA

function table.contains_key(table, key)
   for k in pairs(table) do
      if k == key then
         return true
      end
   end
   return false
end

function valid_indices(stmt, indices)
   --print( "valid_indices() lua calling C cur_indices")
   --io.flush()
   cur = cur_indices(stmt) 
   --print("Cur indices "..list_to_string(cur))
   for idx in pairs(indices) do
      if not table.contains_key(cur,idx) then
         return false
      end
   end
   return true
end

function next_clean_level(cur_idxs,level)
   --print("next_clean_level( ..., "..level.." )")
   --print(string.format("indices_at_each_level %s ",list_to_string(cur_idxs) ))
   
   --print("loop to "..#cur_idxs)
   for i=level+1,#cur_idxs do
      --print("Checking level "..i.." = '"..cur_idxs[i].."'")
      if (# cur_idxs[i] > 0) then
         --print("Good enough"..(# cur_idxs[i]))
         --print("returning "..i)
         return i
      end
   end
   return -1 --sentinal that there were no non-dummy indices left
end

function build_order(final_order, tile_idx_names, ctrl_idx_names, tile_idx_map, cur_level)
   order = {}
   --print("\nbuild_order()")
   --print("build_order(): final_order = ( "..list_to_string(final_order).." )")
   --print("build_order(): ctrl_idx_names = ("..list_to_string(ctrl_idx_names).." )")
   --print("cur_level "..cur_level.."")
   --io.flush()
   
   for i,k in ipairs(final_order) do
      skip = false
      cur = final_order[i]
      --print("\ncur "..cur.." = final_order["..i.."] = "..final_order[i].."  ")
      --control loops below our current level should not be in the current order
      for j=cur_level+2,# ctrl_idx_names do
         --print("j "..j.." final_order["..i.."] = "..final_order[i].."  ")
         if ctrl_idx_names[j] == final_order[i] then
            skip = true
            --print("SKIP "..final_order[i].."  ")
            --io.flush()
         end
      end
      --possibly substitute tile indices ifn necessar
      if table.contains_key(tile_idx_map,final_order[i]) then
         approved_sub = false
         sub_string = tile_idx_map[final_order[i]]
         for j=cur_level+2,# tile_idx_names do
            if tile_idx_names[j] == sub_string then
               approved_sub = true
            end
         end
         if approved_sub then
            cur = sub_string
         end
      end
      if not skip then
         table.insert(order,cur)
      end
   end
   return order
end

function list_to_string(str_list)
   --Helpful debug output
   l = ""
   for i,str in ipairs(str_list) do
      if i > 1 then
         l = l .. ", " .. str
      else
         l = str
      end
   end
   return l
end


function find_cur_level(stmt,idx)
   --Search cur_indices for a idx at stmt
   cur = cur_indices(stmt)
   --print(string.format("find_cur_level(stmt %d, idx %s)  Cur indices %s", stmt, idx, list_to_string(cur)))
   for i,cidx in ipairs(cur) do
      if cidx == idx then
         --print(string.format("found it at index %d", i))
         return i
      end
   end
   error("Unable to find "..idx.." in current list of indices")
end


function chk_cur_level(stmt,idx)
   --Search cur_indices for a idx at stmt
   cur = cur_indices(stmt)
   for i,cidx in ipairs(cur) do
      if cidx == idx then
         return i
      end
   end
   return -1
end


function find_offset(cur_order, tile, control)
   --print("Looking for tile '"..tile.."' and control '"..control.."' in ( "..list_to_string(cur_order)..", )")
   idx1 = -1
   idx2 = -1
   for i,cur in ipairs(cur_order) do
      if(cur == tile) then
         idx1 = i
      end
      if(cur == control) then
         idx2 = i
      end
   end
   if(idx1 < 0) then
      error("Unable to find tile " .. tile .. " in current list of indices")
   end
   if(idx2 < 0) then
      error("Unable to find control " .. control .. " in current list of indices")
   end
   --print("found at level " .. idx2 .. " and " .. idx1)
   if(idx2 < idx1) then
      return idx2-idx1+1
   else
      return idx2-idx1
   end
end

function tile_by_index(tile_indices, sizes, index_names, final_order, tile_method)
   --print "STARTING TILE BY INDEX"
   --io.flush()
   stmt = 0 --assume stmt 0
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   if not valid_indices(stmt,tile_indices) then
      error('One of the indices in the first parameter were not '..
            'found in the current set of indices.')
   end
   if not tile_method then tile_method = counted end
   tile_idx_names = {}
   for i,s in ipairs(tile_indices) do tile_idx_names[i]=s end --shallow copy
   --print("tile_index_names: ['"..list_to_string(tile_indices).."']")
   
   --print("index_names:  ") 
   --for k,v in pairs(index_names) do print(k,v) end
   
   --io.flush()
   
   ctrl_idx_names = {}
   tile_idx_map = {}
   for k,v in pairs(index_names) do
      valid = false
      if(string.sub(k,1,1) == "l") then
         if string.sub(k,-8) == "_control" then
            i = tonumber(string.sub(k,2,-9))
            if i and i >= 1 and i <= (# tile_indices) then
               ctrl_idx_names[i] = v
               --print(string.format("Handling control %s for loop level %d",v,i))
               --print("control "..k.."   name  "..v.." ")
               valid = true
            end
         elseif string.sub(k,-5) == "_tile" then
            i = tonumber(string.sub(k,2,-6))
            if i and i >= 1 and i <= (# tile_indices) then
               --print(string.format("tile %s -> %s",tile_indices[i], v))
               tile_idx_names[i] = v
               tile_idx_map[v] = tile_indices[i]
               --print(string.format("tile %s -> %s",tile_indices[i], v))
               valid = true
            end
         end
      end
      if not valid then error(string.format("%s is not a proper key for specifying "..
                                            "tile or control loop indices\n", k)) end
   end
   
   --filter out control indices (and do name substitution of unprocessed tile indices) for a given level
   cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, -1)
   permute(stmt, cur_order)
   
   for i,cur_idx in ipairs(tile_indices) do
      --print(string.format("i %d  cur_idx %s calling build order ********", i-1, cur_idx))
      cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, i-1)
      --Find a offset between tile loop and control loop
      -- 0   = control loop one level above tile loop
      -- -1  = control loop two levels above tile loop
      -- > 0 = tile loop above control loop
      -- In the last case, we do two extra tile commands to get the control
      -- above the tile and then rely on the final permute to handle the
      -- rest
      level = find_cur_level(stmt,cur_idx)
      offset = find_offset(cur_order, tile_idx_names[i], ctrl_idx_names[i])
      --print(string.format("offset %d", offset))
      
      if (offset <= 0) then
         --print(string.format("[offset<=0]1tile(%d, %d, %d, %d, %s, %s, %s)",stmt, level, sizes[i], level+offset, tile_idx_names[i], ctrl_idx_names[i], tile_method)) 
         tile(stmt, level, sizes[i], level+offset, tile_idx_names[i], ctrl_idx_names[i], tile_method)
      else
         --print(string.format("2tile(%d, %d, %d, %d, %s, %s, %s)", stmt, level, sizes[i], level, tile_idx_names[i], ctrl_idx_names[i], tile_method))
         tile(stmt, level, sizes[i], level, tile_idx_names[i], ctrl_idx_names[i], tile_method);--regular level
         --flip tile and control loop
         --print(string.format("3tile(%d, %d, %d)",stmt, level+1, level+1))
         tile(stmt, level+1, level+1);
         --print(string.format("4tile(%d, %d, %d)",stmt, level+1, level))
         tile(stmt, level+1, level);
         --print(string.format("\n[offset>0]tile(%d, %d, %d, %d,%s,%s,%s)",stmt, level, sizes[i], level, tile_idx_names[i], ctrl_idx_names[i], tile_method)) 
	 --print_code()
         
      end
      
      --Do permutation based on cur_order
      --print "permute based on build order calling build_order()"
      --print "cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, i-1)"
      cur_order = build_order(final_order, tile_indices, ctrl_idx_names, tile_idx_map, i-1)
      --print "permute(stmt, cur_order);"
      permute(stmt, cur_order);
      --print "\nafter permute(), code is:"
      --print_code()
   end
   --print "ENDING TILE BY INDEX"
   --print_code()
end

function normalize_index(index)
   stmt = 0 --assume stmt 0cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   l = find_cur_level(stmt, index)
   tile(stmt, l, l)
   --print(string.format("\n[Normalize]tile(%d, %d, %d)",stmt, l,l)) 
end

function is_in_indices(stmt, idx)
   cur = cur_indices(stmt)
   for i=0,#cur,1 do
      if(cur[i]==idx) then
         return true
      end
   end
   return false
   
end


function copy_to_registers(start_loop, array_name)
   
   --print("\n\n****** starting copy to registers")
   io.flush()

   stmt = 0 --assume stmt 0
   
   -- [Malik] first we make sure that tx and ty are consecutive loops in the 2D thread setup, otherwise all levels for subsequent operations are messed up. Start logic.
   cur = cur_indices(stmt)
   table_Size = table.getn(cur)
   
   --print(string.format("Cur indices %s,",list_to_string(cur)))
   --print(string.format("The table size is %d", table_Size))
   --table.foreach(cur, print)
   --print_code()
   
   level_tx = -1
   level_ty = -1
   if is_in_indices(stmt,"tx") then level_tx = find_cur_level(stmt,"tx") end
   if is_in_indices(stmt,"ty") then level_ty = find_cur_level(stmt,"ty") end
   --print(string.format("level_tx %d  level_ty %d", level_tx, level_ty))
   
   ty_lookup_idx = "" 
   org_level_ty = level_ty
   
   --if(cur[level_tx+1]~=nil and cur[level_tx+1]~="") then ty_lookup = ty_lookup+1 end
   if(cur[level_ty+1]~=nil and cur[level_ty+1]~="") then 
      --print(string.format("IF  cur[%d] = %s", level_ty+1, cur[level_ty+1]))
      ty_lookup_idx = cur[level_ty+1] 
   else
      --if cur[level_ty]  ~= nil then print(string.format("ELSE ty_lookup_idx = cur[%d] = %s", level_ty, cur[level_ty])) --   TODO 
      --else print "ELSE (dangerous)" end
      ty_lookup_idx = cur[level_ty]  -- may assign nil !?
   end
   --if ty_lookup_idx ~= nil then print(string.format("ty_lookup_idx '%s'", ty_lookup_idx))  --  TODO 
   --else print "ty_lookup_idx is NIL"
   --end
   
   if level_ty > 0 then
      --print(string.format("\ntile3(%d,%d,%d)",stmt,level_ty,level_tx+1))
      tile(stmt,level_ty,level_tx+1) 
   end
   --print_code()
   
   --print("\ntylookup is %d",ty_lookup)
   --exit(0)
   --
   cur = cur_indices(stmt)
   table_Size = table.getn(cur)
   --print(string.format("Cur indices %s,",list_to_string(cur)))
   --print("The table size is "..table.getn(cur))
   --table.foreach(cur, print)
   
   if is_in_indices(stmt,"tx") then   level_tx = find_cur_level(stmt,"tx") end
   if ty_lookup_idx then
      if is_in_indices(stmt,ty_lookup_idx) then level_ty = find_cur_level(stmt,ty_lookup_idx) end
   end
   
   ty_lookup = 1
   idx_flag = -1
   -- find the level of the next valid index after ty+1
   --print(string.format("\nlevel_ty %d", level_ty))
   if level_ty > 0 then
      --print(string.format("table_Size %d", table_Size))
      for num= level_ty+ty_lookup,table_Size do
         --print(string.format("num=%d   cur[num] = '%s'",num, cur[num]))
         if(cur[num] ~= "") then
            idx_flag = find_cur_level(stmt,cur[num])
            --print (string.format("idx_flag = %d", idx_flag))
            break
         end
      end
   end
   
   --print(string.format("\n(first) I am checking all indexes after ty+1 %s",idx_flag))
   --print_code()
   --print ""
   
   how_many_levels = 1
   startat = idx_flag + 1
   if startat == 0 then startat = 1 end  -- avoid attempt to examine an illegal array offset
   --print(string.format("idx_flag = %d   I will check levels starting with %d", idx_flag, idx_flag+1))
   
   for ch_lev = startat,table_Size,1 do    -- was for ch_lev = idx_flag+1,table_Size,1 do
      --print(string.format("ch_lev %d", ch_lev))
      if(cur[ch_lev] ~= nil and cur[ch_lev] ~= "") then
         --print(string.format("cur[%d] = '%s'", ch_lev, cur[ch_lev])) 
         how_many_levels = how_many_levels+1
      end
   end
   --print("\nHow Many Levels",how_many_levels)
   
   -- change this all to reflect the real logic which is to normalize all loops inside the thread loops. 
   if(how_many_levels <2) then
      while( idx_flag >= 0) do
         for num = level_ty+ty_lookup,(table_Size) do
            --print(string.format("at top of loop, num is %d", num))
            --print(string.format("num %d", num))
            --print(string.format("cur[num] = '%s'", cur[num]))
            if(cur[num] ~= "") then
               idx=cur[num]
               --print(string.format("idx '%s'", idx))
               
               curlev = find_cur_level(stmt,idx)
               --print(string.format("curlev %d", curlev))
               
               --print_code()
               --print(string.format("\n[COPYTOREG]tile(%d,%d,%d)",stmt,find_cur_level(stmt,idx),level_tx))
               tile(stmt,find_cur_level(stmt,idx),find_cur_level(stmt,idx))
               curlev = find_cur_level(stmt,idx)
               --print(string.format("curlev %d", curlev))
               tile(stmt,find_cur_level(stmt,idx),level_tx)
               --print(string.format("hehe '%s'",cur[num]))
               
               cur = cur_indices(stmt)
               --print("Cur indices INSIDE"..list_to_string(cur))
               table_Size = table.getn(cur)
               --print(string.format("Table Size is: %d",table_Size))
               level_tx = find_cur_level(stmt,"tx")
               --print(string.format("\n level TX is: %d",level_tx))
               level_ty = find_cur_level(stmt,ty_lookup_idx)
               --print(string.format("\n level TY is: %d",level_ty))
               idx_flag = -1
               --print "idx_flag = -1"
               
               -- find the level of the next valid index after ty+1
               
               -- the following was num, which conflicts with loop we're already in, and otherwise wasn't used (?)
               for num= level_ty+ty_lookup,table_Size do
                  --print(string.format("num mucking num = %d", num))
                  if(cur[num] ~= nil and cur[num] ~= "") then
                     idx_flag = find_cur_level(stmt,cur[num])
                     --print("\n(second) I am checking all indexes after ty+1 %s",cur[num])
                     break
                  end
               end
               --print(string.format("num mucked to %d     idx_flag = %d", num, idx_flag))
               
            end
            --print(string.format("at bottom of loop, num is %d", num))
         end
      end
   end
   --print "done with levels"
   
   
   
   
   --print "ARE WE SYNCED HERE?"
   --print_code()
   --print("\ntile(%d,%d,%d)",stmt,level_k,level_k)
   --tile(stmt,level_k,level_k)
   
   -- [Malik] end logic
   --print_code()
   start_level = find_cur_level(stmt, start_loop)
   --We should hold contant any block or tile loop
   block_idxs = block_indices()
   thread_idxs = thread_indices()
   --print("\nblock indices are")
   --table.foreach(block_idxs, print)
   --print("\nthread indices are")
   --table.foreach(thread_idxs, print)
   --print(string.format("\nStart Level: %d",start_level))
   
   hold_constant = {}
   --print("\n Now in Blocks")
   for i,idx in ipairs(block_idxs) do
      --print(string.format("\n Idx:%s : Level: %d",idx,find_cur_level(stmt,idx)))
      if find_cur_level(stmt,idx) >= start_level then
         table.insert(hold_constant, idx)
         --print(string.format("\nJust inserted block %s in hold_constant",idx))
      end
   end
   
   
   --print("\n Now in Threads")
   for i,idx in ipairs(thread_idxs) do
      --print(string.format("\n Idx:%s : Level: %d",idx,find_cur_level(stmt,idx)))
      if find_cur_level(stmt,idx) >= start_level then
         table.insert(hold_constant, idx)
         --print(string.format("\nJust inserted thread %s in hold_constant",idx))
      end
   end
   
   --print "\nhold constant table is: "
   --table.foreach(hold_constant, print)
   
   --print("\nbefore datacopy pvt")
   old_num_stmts = num_statements()
   --print_code()
   --print(string.format("\n[DataCopy]datacopy_privatized(%d, %s, %s, vector having privatized levels)",stmt, start_loop, array_name)) 
   --table.foreach(hold_constant, print)
   datacopy_privatized(stmt, start_loop, array_name, hold_constant)
   
   --print(hold_constant)
   new_num_stmts = num_statements()
   --print("\nthe num of statements:%d\n",new_num_stmt)
   --print_code()
   --exit(0)
   -- [Malik] normalize the copy loops created.
   cur = cur_indices(old_num_stmts)
   --print("Cur indices "..list_to_string(cur))
   for cidx,i in ipairs(cur) do
      if i ~= "tx" and i~="ty" and i~="bx" and i~="by" then
         --tile(old_num_stmts,find_cur_level(old_num_stmts,i),find_cur_level(old_num_stmts,i))
         --print("\nTILE OF REG: tile(%d,%d,%d)",old_num_stmts,find_cur_level(old_num_stmts,i),find_cur_level(old_num_stmts,i))
      end
   end
   --print_code()
   --print("\nthe num of statements OLD+1 :",(old_num_stmts+1))  


--[[ 
   is this commented out? why yes, yes it is   block comment 
   if( (old_num_stmts+1) <= new_num_stmts) then
      cur = cur_indices(old_num_stmts+1)
      --print("Cur indices+1 "..list_to_string(cur))
      for cidx,i in ipairs(cur) do
         if i ~= "tx" and i~="ty" and i~="bx" and i~="by" then
            tile(old_num_stmts+1,find_cur_level(old_num_stmts+1,i),find_cur_level(old_num_stmts+1,i))
	    --print("\nTILE OF REG: tile(%d,%d,%d)",old_num_stmts+1,find_cur_level(old_num_stmts+1,i),find_cur_level(old_num_stmts+1,i))
         end
      end
   end
--]]


   --Unroll to the last thread level
   --for stmt=old_num_stmts,new_num_stmts-1 do
   -- level = find_cur_level(stmt,thread_idxs[#thread_idxs])--get last thread level
   --if level < #cur_indices(stmt) then
   -- unroll(stmt,level+1,0)
   --print(string.format("\n[Unroll]unroll(%d, %d, 0)",stmt, level+1)) 
   ----print_code()
   --end
   --end
   io.flush()
   --print("****** ending copy to registers\n\n")
   --io.flush()
end

function copy_to_shared(start_loop, array_name, alignment)
   --print(string.format("\nstarting copy to shared(%s, %s, %d )",start_loop,array_name,alignment))
   stmt = 0 --assume stmt 0
   cur = cur_indices(stmt)
   --print("Cur indices "..list_to_string(cur))
   
   start_level = find_cur_level(stmt, start_loop)
   --print(string.format("start_level %d", start_level))
   
   old_num_stmts = num_statements()
   --print(string.format("old_num_statements %d", old_num_stmts))
   
   --Now, we give it indices for up to two dimentions for copy loop
   copy_loop_idxs = {"tmp1","tmp2"}
   --print(string.format("\n[DataCopy]datacopy(%d, %d, %s, {\"tmp1\",\"tmp2\"},false,0,1,%d,true)",stmt, start_level, array_name, alignment)) 
   datacopy(stmt, start_level, array_name, copy_loop_idxs, false, 0, 1, alignment,true)
   
   add_sync(stmt,start_loop)
   new_num_stmts = num_statements()
   
   --This is fairly CUBLAS2 specific, not sure how well it generalizes,
   --but for a 2D copy, what we want to do is "normalize" the first loop
   --"tmp1" then get its hard upper bound. We then want to tile it to
   --make the control loop of that tile "ty". We then tile "tmp2" with a
   --size of 1 and make it "tx".
   --print(string.format("fairly CUBLAS2 specific, OLD %d  NEW %d",  old_num_stmts, new_num_stmts ))
   
   for stmt=old_num_stmts,new_num_stmts-1 do
      --print(string.format("for stmt = %d", stmt))
      was_no_error, level = pcall(find_cur_level, stmt, "tmp2")
      
      if was_no_error then 
         --print_code() 
         --print("\nCopy to shared: [If was no error]\n")
         find_cur_level(stmt,"tmp2")
         tile(stmt, level, level)
         
         lower,upper = hard_loop_bounds(stmt, level)
         upper = upper + 1
         --print(string.format("lower %d  upper %d", lower, upper))
         
         tx,ty = thread_dims()
         --print("2-loop cleanup: lower, upper: "..lower..", "..upper..", tx: "..tx)
         
         level = find_cur_level(stmt,"tmp1")
         --print(string.format("level %d", level))
         
         if tx == upper and ty == 1 then
            --print(string.format("tx = %d    upper = %d     ty = %d", tx, upper, ty))
            --print "Don't need"
            
            --Don't need an extra tile level, just move this loop up
            second_level = find_cur_level(stmt,"tmp2")
            --print(string.format("\n[Tile0]tile(%d, %d, 1, %d,%s,%s,counted)",stmt, second_level, level, "tx", "tx")) 
            tile(stmt, second_level, 1, level, "tx", "tx", counted)
         else
            --print "DO need?"
            --print_code()
            if(ty == 1) then new_ctrl = "tmp3" else new_ctrl = "ty" end


--[[ Commenting out a block of Gabe's code in this control flow
               -- level = find_cur_level(stmt,"tmp1")
               tile(stmt, level, level)

               lower,upper = hard_loop_bounds(stmt, level)
               upper = upper + 1
               --print_code()
               --print("2-loop cleanup: lower, upper: "..lower..", "..upper..", tx: "..tx..", level: "..level)
               if(math.ceil(upper/ty) > 1)then
                  tile(stmt, level, math.ceil(upper/ty), level, "tmp", new_ctrl, counted)
                  --print(string.format("\n[Tile1]tile(%d, %d, %f[%d,%d], %d,%s,%s,counted)",stmt, level,  math.ceil(upper/ty),upper,ty, level, "tmp", new_ctrl)) 
               else
                  tile(stmt, level, math.ceil(upper/ty), level, "ty", new_ctrl, counted)
		  --print(string.format("\n[Tile1]tile(%d, %d, %f[%d,%d], %d,%s,%s,counted)",stmt, level,  math.ceil(upper/ty),upper,ty, level, "tx", new_ctrl))
               end
               
               --print_code()    
               -- [Malik] If here we have the loop upper bound > tx, then we should tile once more after the next tile, to carve out the correct tx. 
               lower1,upper1 = hard_loop_bounds(stmt,level)
               level1 = level
               stmt1 = stmt
               -- [Malik] Do the tile after the second level tile with if condition. Just to keep the original order, the tile is being pushed to the end. 
               
               --print("[Malik]-loop cleanup: lower1, upper1: "..lower1..", "..upper1..", tx: "..tx..", level:"..level1)

               --print_code()
               --level = find_cur_level(stmt,"tmp")
               --tile(stmt,level,level)
               --print_code() 
               
               --[Malik] if you are moving the loop above the level1, you need to update level1 with new position which would be level1+2 or second_level
               if(level <= level1) then level1 = level1+2 end
 	       --print(string.format("\n[Tile2]tile(%d, %d, 1, %d,%s,%s,counted)",stmt, second_level, level, "tx", "tx")) 
               --print("\n----------------------------------")
               --print_code()
               --print("\n**********************************")
               --print("[Malik]-loop cleanup: lower1, upper1: "..lower1..", "..upper1..", tx: "..tx..", level:"..level1)
               -- [Malik] If the upper bound > tx, we do another tile to carve out the correct tx from a bigger loop. Else just normalize the bounds. 
               if( upper1 > ty) then
                  third_level = find_cur_level(stmt1,"tmp")
                  --print("\n\n\n\t\t\t\tthirdlevel:"..third_level)
                  tile(stmt1, third_level, ty, third_level, "ty", "tmp", counted)
                  --print(string.format("\n[Tile3]tile(%d, %d, %d,%d,%s,%s,counted)",stmt1, third_level, ty,third_level, "ty", "tmp"))
                  tile(stmt1,third_level+1,third_level+1)
                  --print(string.format("\n[Tile3]tile(%d, %d, %d)",stmt1, third_level+1, third_level+1))
                  tile(stmt1,third_level+1,third_level)
                  --print(string.format("\n[Tile3]tile(%d, %d, %d)",stmt1, third_level+1, third_level))
               else
                  tile(stmt1,level1,level1)
                  --print(string.format("\n[Tile3ELSE]tile(%d, %d, %d)",stmt1,level1,level1))
               end
               
               --print("\nStarting tmp2\n");--print_code();
               second_level = find_cur_level(stmt,"tmp2")
               lower,upper = hard_loop_bounds(stmt,second_level)
               level = second_level
               --print("[Malik]-loop cleanup@tmp2: lower, upper: "..lower..", "..upper..", tx: "..tx..", level:"..level)
               
               if(math.ceil(upper/tx) > 1)then
                  tile(stmt, second_level,math.ceil(upper/tx), level, "tmp", "tx", counted)
                  --print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,math.ceil(upper/tx),second_level, "tmp", "tx"))
               else
                  tile(stmt, second_level,math.ceil(upper/tx), level, "tx", "tx", counted)
                  --print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,math.ceil(upper/tx),second_level, "tx", "tx"))
               end
               --print_code()
               lower2,upper2 = hard_loop_bounds(stmt,level)
               level2 = level
               stmt2 = stmt
               --print("[Malik]-loop cleanup@tmp2: lower2, upper2: "..lower2..", "..upper2..", tx: "..tx..", level:"..level2)
               -- now for the second level.
               if( upper2 > tx) then
                  forth_level = find_cur_level(stmt2,"tmp")
                  --print("\n\n\n\t\t\t\tforthlevel:"..forth_level)
                  --print_code()
                  tile(stmt2, forth_level, 1, forth_level, "tx", "tmp", counted)
                  --print(string.format("\n[Tile3B]tile(%d, %d, %d,%d,%s,%s,counted)",stmt2, forth_level, tx,forth_level, "ty", "tmp"))
                  --print_code()
                  --tile(stmt2,forth_level+1,forth_level+1)
                  --print(string.format("\n[Tile3B]tile(%d, %d, %d)",stmt2, forth_level+1, forth_level+1))
                  --tile(stmt2,forth_level+1,forth_level)
                  --print(string.format("\n[Tile3B]tile(%d, %d, %d)",stmt2, forth_level+1, forth_level))
               else
                  new_level = find_cur_level(stmt2,"ty")
                  tile(stmt2,level2,1,new_level,"tx","tx",counted)
                  --print(string.format("\n[Tile3BELSE]tile(%d, %d, %d)",stmt2,level2,level2))
                  tmp_level = find_cur_level(stmt2,"tmp")
                  tile(stmt2,tmp_level,tmp_level)
               end
               
               --print_code()
               --print("\n----------------------------------")
--]]
               
               --print_code() 
               --print("\nStarting tmp2\n");--print_code();
               first_level = find_cur_level(stmt,"tmp1")
               second_level = find_cur_level(stmt,"tmp2")
               lower,upper = hard_loop_bounds(stmt,second_level)
               
               --print("[Malik]-loop cleanup@tmp2: lower, upper: "..lower..", "..upper..", tx: "..tx..",first level:"..first_level..",second_level:"..second_level)
               
               -- Move the fastest changing dimension loop to the outermost,identified by "tmp2" and to be identified as tx.
               --print(string.format("\n[fastest]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, second_level,1,first_level, "tx", "tx"))
               tile(stmt,second_level,1,first_level,"tx","tx",counted)
               --print_code()
               
               first_level = find_cur_level(stmt,"tmp1")
               lower_1,upper_1 = hard_loop_bounds(stmt,first_level)
               tx_level = find_cur_level(stmt,"tx")
               lower_tx,upper_tx = hard_loop_bounds(stmt,tx_level)
               --print(string.format("UL_1 %d %d     UL_tx %d %d", lower_1, upper_1, lower_tx, upper_tx))
               
               if(math.ceil(upper_tx/tx) > 1)then
                  --print "ceil I say"
                  --print(string.format("\n[Tile1]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, tx_level,tx,tx_level, "tx", "tmp1"))
                  tile(stmt,tx_level,tx,tx_level,"tx","tmp_tx",counted)
                  --print_code()
                  
                  peat = find_cur_level(stmt,"tx")
                  --print(string.format("\n[Tile1]tile(%d, %d, %d)",stmt, peat, peat))
                  tile(stmt, peat, peat )  --find_cur_level(stmt,"tx"),find_cur_level(stmt,"tx"))
                  --print_code()
                  
                  if (find_cur_level(stmt,"tx")>find_cur_level(stmt,"tmp_tx")) then
                     --print(string.format("\nagain [Tile1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp_tx")))
                     tile(stmt,find_cur_level(stmt,"tx"),find_cur_level(stmt,"tmp_tx"))
                     --print_code()
                  end
                  --else
                  --tile(stmt, tx_level,1, tx_level, "tx", "tx", counted)
                  --print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, tx_level,1,tx_level, "tx", "tx"))
               end
               --print_code()
               --]]  -- this apparently is NOT the end of a block comment
               
               --print("\nStarting tmp1\n")
               -- Handle the other slower changing dimension, the original outermost loop, now identified by "tmp1", to be identified as "ty".
               tile(stmt,find_cur_level(stmt,"tmp1"),find_cur_level(stmt,"tmp1"))     
               --print_code()  
               
               ty_level = find_cur_level(stmt,"tmp1")
               lower_ty,upper_ty = hard_loop_bounds(stmt,ty_level)
               
               tx_level = find_cur_level(stmt,"tx")
               lower_tx,upper_tx = hard_loop_bounds(stmt,tx_level)
               --print("[Malik]-loop cleanup@tmp1: lowerty, upperty: "..lower_ty..", "..upper_ty..", ty: "..ty..",ty level:"..ty_level..",tx_level:"..tx_level..", stmt: "..stmt)
               
               --print "before ceil"
               if(math.ceil(upper_ty/ty) > 1)then
                  --print "CEIL IF"
                  --print("\n Inside upper_ty/ty > 1\n");
                  
                  --print(string.format("\n[Tile2]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, ty_level,ty,ty_level, "ty", "tmp_ty"))
                  tile(stmt,ty_level,ty,ty_level,"ty","tmp_ty",counted)
                  --print_code()
                  
                  --print(string.format("\n[Tile2-1]tile(%d, %d, %d)",stmt,find_cur_level(stmt  ,"ty"),find_cur_level(stmt,"ty")))
                  tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"ty"))
                  --print_code()
                  
                  -----------------------------------------------------------------------
                  ----------------------------------------------------------------------
                  cur_idxs = cur_indices(stmt)
                  --print("\n cur indexes are "..list_to_string(cur_idxs))
                  
                  -- Putting ty before any tmp_tx   
                  idx_flag = -1
                  for num= 0,table.getn(cur_idxs) do
                     if(cur[num] == "tmp_tx") then
                        idx_flag = find_cur_level(stmt,cur[num])
                        break
                     end
                  end
                  --print(string.format("\n (1) so i have found out the value of idx flag as %d",idx_flag) )
                  
                  if(idx_flag >=0 ) then  
                     if (find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty")) then
                        --print(string.format("\n[Tile2-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
                        tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                        --print_code()
                     end
                  end
                  
                  -- Now Putting ty before any tmp_ty
                  idx_flag = -1
                  for num= 0,table.getn(cur_idxs) do
                     if(cur[num] == "tmp_ty") then
                        idx_flag = find_cur_level(stmt,cur[num])
                        break
                     end
                  end
		  --print(string.format("\n IF  so i have found out the value of idx flag as %d",idx_flag) )
                  if(idx_flag >=0 ) then  
                     --print "one more test"
                     if ((find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty"))) then
                        --print(string.format("\n[Tile2-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
                        tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                        --print_code()
                     end
                  end
               else
                  --print "CEIL ELSE"
                  --cur_idxs = cur_indices(stmt)
                  --print("\n Inside upper_ty/ty <= 1\n");
                  
                  --print(string.format("\n[Tile3]tile(%d, %d, %d,%d,%s,%s,counted)",stmt, ty_level,1,ty_level, "ty", "ty"))
                  tile(stmt, ty_level,1, ty_level, "ty", "ty", counted)
                  --print_code()
                  
                  --print(string.format("\n[Tile3-1]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1))
                  tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tx")+1)
                  --print_code()
                  
                  idx_flag = -1
                  if(cur_idxs) then
                     --print "CAN NEVER GET HERE?  cur_idxs"
                     for num= 0,table.getn(cur_idxs) do
                        if(cur[num] == "tmp_ty") then
                           idx_flag = find_cur_level(stmt,cur[num])
                           break
                        end
                     end
                  end
                  --print(string.format("\n ELSE so i have found out the value of idx flag as %d",idx_flag) )
                  if(idx_flag >=0 ) then  
                     if (find_cur_level(stmt,"ty")>find_cur_level(stmt,"tmp_ty")) then
                        --print(string.format("tile( stmt %d, level ty %d, level ty %d",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))) 
                        tile(stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty"))
                        --print(string.format("\n[Tile3-2]tile(%d, %d, %d)",stmt,find_cur_level(stmt,"ty"),find_cur_level(stmt,"tmp_ty")))
                     end
                  end
               end
               
               --print_code()
         end
         
         
         --print "\n\n *** at bottom of if in copy to shared, "
         --print_code()
         --print "end of if"
         
      else
         --copy to shared only created one level, not two, so we use a different approach (MV & TMV)
         --print("\nCopy to shared: [If was error]\n")
         level = find_cur_level(stmt,"tmp1")
         tile(stmt, level, level)
         
         --print(string.format("\n[Tile]tile(%d, %d, %d)",stmt, level, level)) 
         tx,ty = thread_dims()
         lower,upper = hard_loop_bounds(stmt, level)
         upper = upper+1 --upper bound given as <=, compare to dimensions tx which is <
         --print("upper "..upper.." tx "..tx)
         if upper == tx then
            rename_index(stmt, "tmp1", "tx")
         else
            --print("upper is not tx")
            --TODO: Don't know, maybe do some tileing etc
            --print_code()
            --print("upper "..upper.." tx "..tx.." stmt: "..stmt.." level: "..level)
            tile(stmt, level,tx,level, "tx", "tmp_tx", counted)
            --print_code()
            
            --print("stmt:"..stmt.." level+1: "..level+1)
            --print("TILE 7")
            tile(stmt, level+1,1,level+1,"tx", "tx",counted)
            --print("TILE 3")
            tile(stmt,level+1,level)
            --print_code()
            
            if(ty > 1) then
               --print_code()
               --print("GOING IN")
               lower,upper = hard_loop_bounds(stmt, level+1)
               --print(string.format("ty %d  lower %d  upper %d", ty, lower, upper))
               --upper=125
               --print("NOW FOR Y: upper "..upper.." ty "..ty.." stmt: "..stmt.." level: "..(level+1).." bound:"..math.ceil(upper/ty))
               tile(stmt, level+1,math.ceil(upper/ty),level+1, "tmp_ty", "ty", counted)
               --tile(stmt, level+2,math.ceil(upper/ty),level+2, "tmp_ty", "ty", counted)
            end
            --print_code()
            --rename_index(stmt, "tmp1", "tx")
            --print("Warning: Need to implement some logic here to tile the single level shared copy loop to match thread dimensions")
         end
      end
      --Always add sync
      add_sync(stmt,start_loop)
      
   end
   --print("ending copy to shared\n")
   --print_code()
end

function unroll_to_depth(max_depth)
   --print(string.format("\n\nunroll_to_depth(%d)", max_depth ))
   --print "SYNC UP"
   
   cur = cur_indices(0)
   thread_idxs = thread_indices()
   guard_idx = thread_idxs[#thread_idxs]
   
   --print(string.format("cur    indices %s",list_to_string(cur)))
   --print(string.format("thread indices %s",list_to_string(thread_idxs)))
   --print(string.format("#thread_idxs = %d", #thread_idxs))
   --print(string.format("guard_idx = %s", guard_idx))
   
   ---- HERE FIND OUT THE LOOPS WHICH ARE COMMON BETWEEN STATEMENTS   
   common_loops = {}
   comm_loops_cnt = 0
   num_stmts = num_statements()
   --print(string.format("num statements %d", num_stmts))
   
   for stmt=0,num_stmts-1 do
      cur_idxs = cur_indices(stmt)
      
      --print(string.format("\nSTMT %d Current Indices: %s",stmt,list_to_string(cur_idxs)))
      
      if(chk_cur_level(stmt,"tx")>0) then
         for ii=1,find_cur_level(stmt,"tx")-1 do    -- started at 0
            --print(string.format("ii = %d", ii)) -- index starts at 1, what does index 0 do?
            --if cur_idxs[ii] == nil then print "cur_idxs[i]] is NIL" 
            --else print(string.format("cur_idxs[%d] = '%s'", ii, cur_idxs[ii])) -- index starts at 1, what does index 0 do?
            --end
            
            if(cur_idxs[ii] ~= "bx" and cur_idxs[ii] ~= "by" and cur_idxs[ii] ~= nil and cur_idxs[ii] ~= "tx" and cur_idxs[ii] ~= "ty" and cur_idxs[ii] ~= "") then 
               
               --print(string.format("id %s is not in the list", cur_idxs[ii] ))
               
               for stmt1=stmt+1,num_stmts-1 do
                  --print(string.format("\nii %d stmt1 is %d", ii, stmt1))          
                  cur_idxs1 = cur_indices(stmt1)
                  --print("\nstmt1 cur_idxs1 is "..list_to_string(cur_idxs1))   
                  
                  --print(string.format("cur level(%d, %s) = %d", stmt, "tx",  find_cur_level(stmt,"tx")))    
                  
                  endrange = find_cur_level(stmt,"tx")-1
                  --print(string.format("for iii=1, %d do", endrange))
                  
                  for iii=1,find_cur_level(stmt,"tx")-1 do  -- started at 0
                     --print(string.format("stmt %d   ii %d   iii %d ", stmt, ii, iii))
                     --if(cur_idxs1[iii] ~= nil) then 
                     --   print(string.format("stmt %d   ii %d   iii %d  cur_idxs1[%d] = '%s'", stmt, ii, iii, iii, cur_idxs1[iii]))  
                     --else 
                     --   print(string.format("stmt %d   ii %d   iii %d  cur_idxs1[%d] = NIL", stmt, ii, iii, iii))  
                     --end
                     
                     if(cur_idxs1[iii] ~= "bx" and cur_idxs1[iii] ~= "by" and cur_idxs1[iii] ~= nil and cur_idxs1[iii] ~= "tx" and cur_idxs1[iii] ~= "ty" and cur_idxs1[iii] ~= "") then  
                        if(cur_idxs[ii] == cur_idxs1[iii]) then
                           --print("\nfound idx:"..cur_idxs[ii])
			   --if(comm_loops_cnt == 0) then print "\n\n*** WARNING *** assigning to array index ZERO in Lua" end
                           common_loops[comm_loops_cnt] = cur_idxs[ii]
                           --print(string.format("cl[%d] = '%s'", comm_loops_cnt,   common_loops[comm_loops_cnt]))
                           comm_loops_cnt = comm_loops_cnt + 1
                        end
                     end  
                  end
               end  
            end
         end
      end
   end
   ----
   --if(comm_loops_cnt>0) then 
   --   print("\n COMM LOOPS :TOTAL "..comm_loops_cnt..", and are "..list_to_string(common_loops).." this loop :"..common_loops[0])
   --else
   --   print "UNROLL can't unroll any loops?"
   --end
   
   
   
   
   repeat
      old_num_stmts = num_statements()
      --print(string.format("old_num_statements %d", old_num_stmts))
      
      for stmt=0,old_num_stmts-1 do
         cur_idxs = cur_indices(stmt)
         --print(string.format("stmt %d    cur_idxs = %s", stmt, list_to_string(cur_idxs)))
         if(#cur_idxs > 0) then 
            gaurd_level = -1
            if(chk_cur_level(stmt,guard_idx)>0) then
               gaurd_level = find_cur_level(stmt,guard_idx)
            end
            --print(string.format("guard_level(sp) = %d", gaurd_level))
            
            if(gaurd_level>-1) then
               level = next_clean_level(cur_idxs,gaurd_level)
               --print(string.format("next clean level %d", level))
               
               --need to handle max_depth
               num_unrolled = 0
               level_unroll_comm = level
               level_arr = {}
               while level >= 0 do
                  --print(string.format("while: level = %d", level))
                  
                  if num_unrolled == max_depth then break end
                  --print("Unrolling "..stmt.." at level "..(level).." index ".. cur_idxs[gaurd_level+1])
                  
                  level_arr[num_unrolled] = level
                  num_unrolled = num_unrolled + 1
                  
                  guard_level = find_cur_level(stmt,guard_idx)
                  level = next_clean_level(cur_idxs,level+1)
               end
               --dies print("How many levels for unroll commands"..table.getn(level_arr).." which is "..level_arr[0].." and "..level_arr[#level_arr])
               --if(table.getn(level_arr) ~= nil) then
               
               --print "OK, NOW WE UNROLL"
               
               if(level_unroll_comm >= 0)then
                  for i = table.getn(level_arr),0,-1 do
                     --print(string.format("\ni=%d", i))
                     --print(string.format("[Unroll]unroll(%d, %d, 0)",stmt, level_arr[i]))     
                     
                     unroll(stmt,level_arr[i],0)
                     --print("finished unroll]]\n")
                     --print_code()
                  end
               end
------
            end    
--[[

THERE WAS A BIG BLOCK OF COMMENTED OUT CODE HERE 


--]]
------
         end
      end
      new_num_stmts = num_statements()

   until old_num_stmts == new_num_stmts

end


