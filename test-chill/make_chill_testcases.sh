#!/bin/bash

srcdir=$1
shift 1

is_test_file() {
    if [ -n `sed -n "s/^destination(.*)/\0/p" $1` ]; then
        echo 1;
    else
        echo 0;
    fi
}

if [ ! -e ./test-chill ]; then
    mkdir ./test-chill
fi

make_test_file=./test-chill/chill_testcases.mk
if [ -e $make_test_file ]; then
    rm $make_test_file
fi

for test_dir in $@;
do
    for test_file_path in `ls $srcdir/$test_dir/*.py`
    do
        if [ `is_test_file $test_file_path` -ne 0 ]; then
            test_file=`basename $test_file_path`
            
            ## Add run test
            run_test_file="test-chill/test-$test_file"
            echo ""                                                                                   >  $run_test_file # make new file
            echo "err=\`$srcdir/test-chill/runchilltest.sh $srcdir/$test_dir/$test_file check-run\`"  >> $run_test_file
            echo "exit \$?"                                                                           >> $run_test_file
            chmod +x $run_test_file
            
            echo "TESTS += $run_test_file"                                                            >> $make_test_file
        fi
    done
done
