#!/bin/bash 


## Exit with a hard error, or continue
maybe_exit_with_error_code() {
    if [ $1 != 0 ]; then
        exit 77
    fi
}


## Exit as either pass or fail, depending on both ther error code
##      and whether or not the expectfail flag is set
exit_with_passfail_code() {
    err=$1
    shift 1
    msg=$@
    
    if [ $expect_fail == 0 ]; then
        if [ $err != 0 ]; then
            echo $err $msg
        fi
        exit $err
    else
        if [ $err == 0 ]; then
            exit 1
        else
            exit $err
        fi
    fi
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
shift 2

pushd $chill_script_path >/dev/null
chill_generated_source=$(get_destination $chill_script)

## remove generated file if it exists
if [ -e $chill_generated_source ]; then
    rm $chill_generated_source
fi


## Defaults
expect_fail=0

## Read arguments
arg_index=1
while [ $arg_index -lt $(( $# + 1 )) ]; do
    case ${!arg_index} in
        expectfail)
                expect_fail=1
            ;;
        check-run)
                test_type=${!arg_index}
            ;;
        check-generated-diff)
                test_type=${!arg_index}
                arg_index=$[$arg_index + 1]
                pushd $call_path >/dev/null
                check_output_diff_answer_file=`realpath ${!arg_index}`
                popd >/dev/null
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
    $chill_exec $chill_script 1>$1 2>$2
    err=$?
    if [ $err == 0 -a -n "$3" -a $3 -gt 0 ]; then
        if [ ! -e $chill_generated_source ]; then
            err=$3
            msg="output file was not generated"
        fi
    else
        msg="error while running CHiLL"
    fi
    echo $err $msg
}


## Run Test $$
case $test_type in
    check-run)
            err=`run_chill /dev/null /dev/null 1`
            if [ -e $chill_generated_source ]; then
                rm $chill_generated_source
            fi
            exit_with_passfail_code $err
        ;;
esac

popd >/dev/null

