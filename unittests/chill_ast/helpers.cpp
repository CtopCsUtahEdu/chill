//
// Created by joe on 10/9/16.
//

#include "gtest/gtest.h"
#include "chill_ast.hh"

TEST (helpers, ulhack) {
    char *orig1 = strdup("1024UL");
    EXPECT_STREQ(ulhack(orig1),"1024");
    char *orig2 = strdup("0UL");
    EXPECT_STREQ(ulhack(orig2),"0");
    char *orig3 = strdup("-1024UL");
    EXPECT_STREQ(ulhack(orig3),"-1024");
}

TEST (helpers, parseUnderlyingType) {
    char *orig1 = strdup("float *");
    EXPECT_STREQ(parseUnderlyingType(orig1),"float");
    char *orig2 = strdup("struct abc *");
    EXPECT_STREQ(parseUnderlyingType(orig2),"struct abc");
}

TEST (helpers, restricthack) {
    char *orig = strdup("int * __restrict__");
    EXPECT_STREQ(restricthack(orig),"int * ");
}

TEST (helpers, splitTypeInfo) {
    char *orig = strdup("float **[100][100]");
    EXPECT_STREQ(splitTypeInfo(orig),"[100][100]");
    EXPECT_STREQ(orig,"float **");
}

