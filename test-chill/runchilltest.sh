#!/bin/bash 


## Exit with a hard error, or continue
maybe_exit_with_error_code() {
    if [ $1 != 0 ]; then
        echo $@
        exit 99
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
    cmd="sed -n \"s/destination('\(.*\)')/\1/p\" $1"
    echo `eval $cmd`
}


## Read input
call_path=$(dirname `realpath $0`)
chill_exec=`realpath $1`
chill_script_path=$(dirname `realpath $2`)
chill_script=$(basename $2)
chill_answers_path=`realpath $3`
shift 3

chill_generated_source=$(get_destination $chill_script)
chill_correct_source=$chill_answers_path/$chill_generated_source

## remove generated file if it exists
if [ -e $chill_generated_source ]; then
    pushd $chill_script_path >/dev/null
    rm $chill_generated_source
    popd >/dev/null
fi


## Defaults
expect_fail=0
skip_test=0

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
    err=$?
    if [ $err == 0 -a -n "$3" -a $3 -gt 0 ]; then
        if [ ! -e $chill_generated_source ]; then
            echo "$3 output file was not generated"
            exit $3
        fi
    else
        msg="error while running CHiLL"
    fi
    popd >/dev/null
    maybe_exit_with_error_code $err $msg
}

check_diff() {
    local generated_file=$1
    local correct_file=$2
    
    diff_flags="-qwB"
    if [ -n "`diff $generated_file $correct_file`" ]; then
        echo "1 output file is not correct"
    else
        echo "0"
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
            
            pushd $chill_script_path >/dev/null
            if [ -e $chill_generated_source ]; then
                rm $chill_generated_source
            fi
            popd >/dev/null
            exit_with_passfail_code $err
        ;;
    check-diff)
            run_chill /dev/null /dev/null 99
            err=`check_diff $chill_generated_source $chill_correct_source`
            exit_with_passfail_code $err
        ;;
esac

