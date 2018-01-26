#!/bin/bash 


## Exit with a hard error, or continue
maybe_exit_with_error_code() {
    if [ $1 != 0 ]; then
        echo $@
        exit 99
    fi
}

## Exit with a skip, or continue
maybe_exit_with_skip_code() {
    if [ $1 != 0 ]; then
        echo $@
        exit 77
    fi
}


## Exit as either pass or fail, depending on both ther error code
##      and whether or not the expectfail flag is set
exit_with_passfail_code() {
    local err=$1
    if [ $expect_fail == 0 ]; then
        if [ $err != 0 ]; then
            echo $@
            exit $err
        else
            exit 0
        fi
    else
        if [ $err == 0 ]; then
            echo $@
            exit 1
        else
            echo $@
            exit 0
        fi
    fi
}

exit_with_skip_code() {
    echo $@
    exit 77 
}


## Get the destination filename from the script
get_destination() {
    cmd="sed -n \"s/destination([\\\'\\\"]\\\(.*\\\)[\\\'\\\"])/\\\1/p\" $1"
    echo `eval $cmd`
}


## Read input
call_path=$(dirname `realpath $0`)
chill_exec=`realpath $1`
chill_script_path=$(dirname `realpath $2`)
chill_script=$(basename $2)
chill_dest=`get_destination $2`
chill_answers_path=`realpath $3`
shift 3

chill_generated_source=$chill_script_path/$chill_dest
chill_generated_stdout=$chill_script_path/$(basename $chill_script).stdout
chill_generated_stderr=$chill_script_path/$(basename $chill_script).stderr
chill_generated_object=$chill_script_path/$(basename $chill_script).o
chill_correct_source=$chill_answers_path/$chill_dest
chill_correct_stdout=$chill_answers_path/$(basename $chill_generated_stdout)
chill_correct_stderr=$chill_answers_path/$(basename $chill_generated_stderr)

echo "CHiLL exec:            $chill_exec"
echo "CHiLL script path:     $chill_script_path"
echo "CHiLL script name:     $chill_script"
echo "Generated source file: $chill_generated_source"
echo "Correct source file:   $chill_correct_source"
echo "Generated object file: $chill_generated_object"

## remove generated files if they exist
if [ -e $chill_generated_source ]; then
    pushd $chill_script_path >/dev/null
    rm $chill_generated_source
    popd >/dev/null
fi

if [ -e $chill_generated_stdout ]; then
    pushd $chill_script_path >/dev/null
    rm $chill_generated_stdout
    popd
fi

if [ -e $chill_generated_stderr ]; then
    pushd $chill_script_path >/dev/null
    rm $chill_generated_stderr
    popd
fi


## Defaults
expect_fail=0
skip_test=0
compiler=gcc

## Read arguments
arg_index=1
while [ $arg_index -lt $(( $# + 1 )) ]; do
    case ${!arg_index} in
        exfail)
                expect_fail=1
            ;;
        skip)
                skip_test=1
            ;;
        check-run)
                test_type=${!arg_index}
            ;;
        check-diff)
                test_type=${!arg_index}
            ;;
        check-stdout)
                test_type=${!arg_index}
            ;;
        check-stderr)
                test_type=${!arg_index}
            ;;
        check-compile)
                test_type=${!arg_index}
            ;;
        cuda)
                compiler=nvcc
            ;;
    esac
    arg_index=$[$arg_index + 1]
done



## A basic run chill command
##      $1 - file to send stdout to
##      $2 - file to send stderr to
##      $3 - check output file exists
##          0 or nothing    - don't check
##          > 0             - the error code if file does not exist

run_chill() {
    pushd $chill_script_path >/dev/null
    $chill_exec $chill_script 1>$1 2>$2
    local err=$?
    if [ $err == 0 ]; then
        if [ "x$3" != "x" -a "x$3" != "x0" ]; then
            if [ ! -e $chill_generated_source ]; then
                err=$3
                msg="CHiLL did not generate output"
            fi
        fi
    else
        if [ $err != 0 -a "x$3" != "x" -a "x$3" != "x0" ]; then
            err=$3
        fi
        msg="error while running CHiLL"
    fi
    popd >/dev/null
    echo $err $msg
}


## Check diff between generated file and expected output (ignorring errors)
##      $1 - first file
##      $2 - second file

check_diff() {
    local generated_file="$1.temp"
    local correct_file="$2.temp"

    sed 's/\(.*\)\/\/.*/\1/' $1 > $generated_file
    sed 's/\(.*\)\/\/.*/\1/' $2 > $correct_file
    
    local diffout=`diff -qwB $generated_file $correct_file`
    if [ -n "$diffout" ]; then
        echo "1 output file is not correct"
    else
        echo "0"
    fi

    rm $generated_file
    rm $correct_file
}


## Compile generated source
##      $1 - compiler
##      $2 - error code on failure

compile_chill() {
    $1 -c $chill_generated_source -o $chill_generated_object
    local err=$?
    if [ $err == 0 ]; then
        echo 0
    else
        echo $2
    fi
}


## Skip Test? ##
if [ $skip_test != 0 ]; then
    exit_with_skip_code
fi

## Run Test $$
case $test_type in
    check-run)
            err=`run_chill /dev/null /dev/null 2`
            exit_with_passfail_code $err
        ;;
    check-diff)
            err=`run_chill /dev/null /dev/null 77`
            maybe_exit_with_skip_code $err
            err=`check_diff $chill_generated_source $chill_correct_source`
            exit_with_passfail_code $err
        ;;
    check-stdout)
            tmp=`pwd`/tmp
            err=`run_chill $tmp /dev/null 77`
            if [ $err == 77 ]; then
                rm -f $tmp
            fi
            maybe_exit_with_skip_code $err
            err=`check_diff $tmp $chill_correct_stdout`
            rm -f $tmp
            exit_with_passfail_code $err
        ;;
    check-stderr)
            tmp=`pwd`/tmp
            err=`run_chill $tmp /dev/null 77`
            if [ $err == 77 ]; then
                rm -f $tmp
            fi
            maybe_exit_with_skip_code $err
            err=`check_diff $tmp $chill_correct_stderr`
            rm -f $tmp
            exit_with_passfail_code $err
        ;;
    check-compile)
            err=`run_chill /dev/null /dev/null 77`
            maybe_exit_with_skip_code $err
            err=`compile_chill $compiler 1`
            exit_with_passfail_code $err
        ;;
esac

