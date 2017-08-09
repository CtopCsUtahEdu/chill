#!/bin/sh

for test_file in *.py
do
    ../../../chill $test_file \
            1> RIGHTANSWERS/$(basename $test_file).stdout \
            2> RIGHTANSWERS/$(basename $test_file).stderr
done

