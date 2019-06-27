#include "loop.hh"
#include "chill_error.hh"
#include <omega.h>
#include "omegatools.hh"
#include <code_gen/CG_utils.h>
#include "iegenlib.h"

//Order taking out dummy variables
static std::vector<std::string> cleanOrder(std::vector<std::string> idxNames) {
  std::vector<std::string> results;
  for (int j = 0; j < idxNames.size(); j++) {
    if (idxNames[j].length() != 0)
      results.push_back(idxNames[j]);
  }
  return results;
}

void Loop::permute_by_index(int stmt_num, const std::vector<std::string>& curOrder) {
  std::vector<std::string> cIdxNames = cleanOrder(idxNames[stmt_num]);
    bool same = true;
    std::vector<int> pi;
    for (int i = 0; i < curOrder.size(); i++) {
      bool found = false;
      for (int j = 0; j < cIdxNames.size(); j++) {
        if (strcmp(cIdxNames[j].c_str(), curOrder[i].c_str()) == 0) {
          debug_fprintf(stderr, "pushing pi for j+1=%d\n", j+1);

          pi.push_back(j + 1);
          found = true;
          if (j != i)
            same = false;
        }
      }
      if (!found) {
        throw std::runtime_error(
                                 "One of the indexes in the permute order were not "
                                 "found in the current set of indexes.");
      }
    }
    for (int i = curOrder.size(); i < cIdxNames.size(); i++) {
      debug_fprintf(stderr, "pushing pi for i=%d\n", i);
      pi.push_back(i);
    }
    if (same)
      return;
    permute_by_index(stmt_num, pi);
    //Set old indexe names as new
    for (int i = 0; i < curOrder.size(); i++) {
      idxNames[stmt_num][i] = curOrder[i].c_str(); //what about sibling stmts?
    }
}

bool Loop::permute_by_index(int stmt_num, const std::vector<int> &pi) {
  // check for sanity of parameters
  if (stmt_num >= stmt.size() || stmt_num < 0)
    throw std::invalid_argument("invalid statement " + std::to_string(stmt_num));
  const int n = stmt[stmt_num].xform.n_out();
  if (pi.size() > (n - 1) / 2) {
 debug_fprintf(stderr, "\n\nloop_cuda_CHILL.cc L 761, pi.size() %d  > ((n=%d)-1)/2 =  %d\n", pi.size(), n, (n-1)/2);
    for (int i=0; i<pi.size(); i++) debug_fprintf(stderr, "pi[%d] = %d\n", i, pi[i]);

    throw std::invalid_argument(
                                "iteration space dimensionality does not match permute dimensionality");
  }
  int first_level = 0;
  int last_level = 0;
  for (int i = 0; i < pi.size(); i++) {
    if (pi[i] > (n - 1) / 2 || pi[i] <= 0)
      throw std::invalid_argument(
                                  "invalid loop level " + std::to_string(pi[i])
                                  + " in permuation");

    if (pi[i] != i + 1) {
      if (first_level == 0)
        first_level = i + 1;
      last_level = i + 1;
    }
  }
  if (first_level == 0)
    return true;

  std::vector<int> lex = getLexicalOrder(stmt_num);
  std::set<int> active = getStatements(lex, 2 * first_level - 2);
  Loop::permute(active, pi);
}

void Loop::flatten_by_index(
    int                         stmt_num,
    std::string                 idxs,
    std::vector<std::string>&   loop_level_names,
    std::string                 inspector_name) {

  std::vector<int> loop_levels;
  for(auto i = 0; i < loop_level_names.size(); i++) {
    loop_levels.push_back(findCurLevel(stmt_num, loop_level_names[i]));
  }
  flatten(stmt_num, idxs, loop_levels, inspector_name);

  bool initial_val = false;

  idxNames.push_back(idxNames[stmt_num]);
  idxNames[stmt_num + 1].push_back(idxs);
}

void Loop::distribute_by_index(std::vector<int> &stmt_nums, std::string loop_level) {

  std::set<int> stmts;

  for (int i = 0; i < stmt_nums.size(); i++)
    stmts.insert(stmt_nums[i]);

  distribute(stmts, findCurLevel(stmt_nums[0], loop_level));

  //syncs.push_back()

}

void Loop::fuse_by_index(std::vector<int> &stmt_nums, std::string loop_level) {

  std::set<int> stmts;

  for (int i = 0; i < stmt_nums.size(); i++)
    stmts.insert(stmt_nums[i]);

  fuse(stmts, findCurLevel(stmt_nums[0], loop_level));
}

void Loop::peel_by_index(int stmt_num, std::string level, int amount) {
  int old_stmt_num = stmt.size();
  peel(stmt_num, findCurLevel(stmt_num, level), amount);
  int new_stmt_num = stmt.size();
  //For all statements that were in this unroll together, drop index name for unrolled level
  for (int i = old_stmt_num; i < new_stmt_num; i++) {
    idxNames.push_back(idxNames[stmt_num]);
  }
}

void Loop::shift_to_by_index(int stmt_num, std::string level, int absolute_position) {

  shift_to(stmt_num, findCurLevel(stmt_num, level), absolute_position);

}



void Loop::scalar_by_index(int stmt_num, std::vector<std::string>& level_names,
                                  std::string arrName, int memory_type, int padding,
                                  int assign_then_accumulate) {

  std::vector<int> loop_levels;
  for(auto i = 0; i < level_names.size(); i++) {
    loop_levels.push_back(findCurLevel(stmt_num, level_names[i]));
  }

  int old_num_stmts = num_statement();
  scalar_expand(stmt_num, loop_levels, arrName, memory_type, padding, assign_then_accumulate);
  int new_num_stmts = num_statement();

  std::vector<std::string> namez = idxNames[stmt_num];

  for (int i = 0; i < new_num_stmts - old_num_stmts; i++) {  // ???
    idxNames.push_back(idxNames[stmt_num]);
    //stmt_nonSplitLevels.push_back(std::vector<int>());
  }
}



void Loop::split_with_alignment_by_index(int stmt_num, std::string level, int alignment, int direction) {

  split_with_alignment(stmt_num, findCurLevel(stmt_num, level), alignment, direction);
  idxNames.push_back(idxNames[stmt_num]);
}



void Loop::compact_by_index(int stmt_num, std::string level, std::string new_array,
                            int zero, std::string data_array) {
  int old_num_stmts = num_statement();
  compact(stmt_num, findCurLevel(stmt_num, level), new_array, zero, data_array);
  int new_num_stmts = num_statement();
  int i;
  for (i = 0; i < new_num_stmts - old_num_stmts - 1; i++) {
    idxNames.push_back(idxNames[stmt_num]);
    //stmt_nonSplitLevels.push_back(std::vector<int>());
  }
  std::vector<std::string> last_index;
  for (int j = 0; j < idxNames[stmt_num].size() - 1; j++)
    last_index.push_back(idxNames[stmt_num][j]);

  idxNames.push_back(last_index);
  //stmt_nonSplitLevels.push_back(std::vector<int>());

}



void Loop::make_dense_by_index(int stmt_num, std::string loop_level_name, std::string new_loop_index) {
  int loop_level = findCurLevel(stmt_num, loop_level_name);
  make_dense(stmt_num, loop_level, new_loop_index);
  std::vector<std::string> new_idx;
  for (int i = 0; i < loop_level - 1; i++)
    new_idx.push_back(idxNames[stmt_num][i]);

  new_idx.push_back(new_loop_index);

  for (int i = loop_level - 1; i < idxNames[stmt_num].size(); i++)
    new_idx.push_back(idxNames[stmt_num][i]);

  idxNames[stmt_num] = new_idx;

}



void Loop::skew_by_index(std::vector<int> stmt_num, std::string level_name, std::vector<int> coefs) {

  std::set<int> stmts;
  for (int i = 0; i < stmt_num.size(); i++)
    stmts.insert(stmt_num[i]);

  skew(stmts, findCurLevel(stmt_num[0], level_name), coefs);

}



void Loop::reduce_by_index(int stmt_num, std::vector<std::string>& level_names, int param,
                           std::string func_name, std::vector<int> seq_level, int bound_level) {

  std::vector<int> loop_levels;
  for(auto i = 0; i < level_names.size(); i++) {
    loop_levels.push_back(findCurLevel(stmt_num, level_names[i]));
  }
  Loop::reduce(stmt_num, loop_levels, param, func_name, seq_level);

}


