#!/bin/bash
set -e
# Initialize the Variables
BUILD_MODE="Normal"
BUILD_OUT_DIR=${BANGPY_BUILD_PATH}
if [ -n "${BUILD_OUT_DIR}" ]; then
    if [ "${BUILD_OUT_DIR:0-1}" != "/" ]; then
        BUILD_OUT_DIR=${BANGPY_BUILD_PATH}"/"
    fi
else
    echo "Please set BANGPY_BUILD_PATH environment variable first."
    echo "eg. export BANGPY_BUILD_PATH=/mlu-ops/bangpy-ops/outs/"
    exit -1
fi

if [ -z "${BANGPY_HOME}" ]; then
    echo "Please set BANGPY_HOME environment variable first."
    echo "eg. export BANGPY_HOME=/mlu-ops/bangpy-ops/"
    exit -1
fi

if [ "${BANGPY_HOME:0-1}" != "/" ]; then
    BANGPY_HOME=${BANGPY_HOME}"/"
fi
BANGPY_UTILS_PATH=${BANGPY_HOME}"utils/"

usage () {
    echo "USAGE: release.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help                     Print usage"
    echo "      --filter=*                     Build specified OP only"
    echo "      --target=*                     Test mlu target:[mlu270, mlu370-s4, mlu220-m2, mlu290]"
    echo "      --opsfile=*                    Operators list file"
    echo "      -r, --release                  Build by release mode"
    echo "      -t, --test                     Test operators after build"

    echo
}
ARGS="$*"
TEST_ENABLE="False"
cmdline_args=$(getopt -o h,r,t --long filter:,release,test,target:,opsfile: -n 'release.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $? != 0 ]; then echo "Unknown options, use -h or --help" >&2 ; exit -1; fi
if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --filter)
          shift
          shift
          ;;
      --target)
          shift
          shift
          ;;
      --opsfile)
          shift
          shift
          ;;
      -r | --release)
          BUILD_MODE="Release"
          shift
          ;;
      -t | --test)
          TEST_ENABLE="True"
          shift
          ;;
      -h | --help)
          usage
          exit 0
          ;;
      --)
          shift
          break
          ;;
      *)
          echo "-- Unknown options ${1}, use -h or --help"
          usage
          exit -1
          ;;
    esac
  done
fi

${BANGPY_UTILS_PATH}"build_operators.sh" ${ARGS}

if [ "${BUILD_MODE}" == "Release" ]; then
    # Link the files of .so in the compiled directory to bangpylib.
    BANGPY_HEAD_FILES_LIST=()
    BANGPY_LINK_FILE_LIST=()
    BANGPY_LINK_OP_DIR_LIST=()
    if [ -z "${BANGPY_LINK_OP_DIR_LIST}" ]; then
        for file in $(find ${BUILD_OUT_DIR} -maxdepth 1 -type d)
        do  
            file_name=${file##*/}
            BANGPY_LINK_OP_DIR_LIST+=(${file_name})
        done
    fi

    BANGPY_ALL_SO=()

    for i in ${BANGPY_LINK_OP_DIR_LIST[@]}
    do 
        for link_file in $(find ${BUILD_OUT_DIR}$i/ -maxdepth 4 -name libmluops.so)
        do  
            BANGPY_ALL_SO+=(${link_file})
        done

        for link_file in $(find ${BUILD_OUT_DIR}$i/ -maxdepth 1 -name "*\.o")
        do  
            BANGPY_LINK_FILE_LIST+=(${link_file})
        done

        for link_file in $(find ${BUILD_OUT_DIR}$i/ -maxdepth 1 -name host.h)
        do  
            BANGPY_HEAD_FILES_LIST+=(${link_file})
        done
    done

    if [ -z "${BANGPY_LINK_FILE_LIST}" ]; then
        echo "Could not find any linkable file, please check input operators."
        exit -1
    fi

    gcc -shared -o ${BUILD_OUT_DIR}"libmluops.so"  ${BANGPY_LINK_FILE_LIST[@]}

    if [ -n "${BANGPY_HEAD_FILES_LIST}" ]; then
        for i in ${BANGPY_HEAD_FILES_LIST[@]}
        do 
            BANGPY_HEARDER_LIST_STR=${BANGPY_HEARDER_LIST_STR}"$i"","
        done
        BANGPY_HEARDER_LIST_STR=${BANGPY_HEARDER_LIST_STR%?}
    fi

    python3 ${BANGPY_UTILS_PATH}"generate_all_ops_header.py" ${BANGPY_HEARDER_LIST_STR}

    # Delete the files of libmluops.so in the compiled directory.
    for i in ${BANGPY_ALL_SO[@]}
    do
        rm $i
    done
fi

if [ "${TEST_ENABLE}" == "True" ]; then
    ${BANGPY_UTILS_PATH}"test_operators.sh" ${ARGS} "--only_test"
fi

# rm all but libbangpy_ops.so and mluops.h
if [ "${BUILD_MODE}" == "Release" ]; then
    for i in ${BANGPY_LINK_OP_DIR_LIST[@]}
    do
        rm ${BUILD_OUT_DIR}$i -rf
    done
fi

