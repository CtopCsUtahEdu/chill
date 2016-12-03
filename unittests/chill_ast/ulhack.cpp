//
// Created by joe on 10/9/16.
//

#include "gtest/gtest.h"
#include "chill_ast.hh"

TEST (chill_ast, ulhack_1024UL) {
    char *orig = strdup("1024UL");
    EXPECT_STREQ(ulhack(orig),"1024");
}

