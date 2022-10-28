#!/bin/bash
set -e
# Initialize the Variables
BUILD_OUT_DIR=${BANGPY_BUILD_PATH}

if [ -z "${BANGPY_HOME}" ]; then
    echo "Please set BANGPY_HOME environment variable first."
    echo "eg. export BANGPY_HOME=/mlu-ops/bangpy-ops/"
    exit -1
fi

if [ "${BANGPY_HOME:0-1}" != "/" ]; then
    BANGPY_HOME=${BANGPY_HOME}"/"
fi
BANGPY_UTILS_PATH=${BANGPY_HOME}"utils/"
OPS_DIR=${BANGPY_HOME}"ops/"

if [ -n "${BUILD_OUT_DIR}" ]; then
    if [ "${BUILD_OUT_DIR:0-1}" != "/" ]; then
        BUILD_OUT_DIR=${BANGPY_BUILD_PATH}"/"
    fi
else
    echo "Please set BANGPY_BUILD_PATH environment variable first."
    echo "eg. export BANGPY_BUILD_PATH=/mlu-ops/bangpy-ops/outs/"
    exit -1
fi

usage () {
    echo "USAGE: test_operators.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help                     Print usage"
    echo "      --filter=*                     Test specified OP only"
    echo "      --target=*                     Test mlu target:[mlu270, mlu370-s4, mlu220-m2, mlu290]"
    echo "      --opsfile=*                    Operators list file"
    echo "      --only_test                    Test without build"
    echo "      --cases_dir=*                  Test with prototxt cases"

    echo
}

TEST_ARGS="$*"
BUILD_ENABLE="True"
cmdline_args=$(getopt -o h,r,t --long filter:,release,test,target:,cases_dir:,only_test,opsfile: -n 'test_operators.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $? != 0 ]; then echo "Unknown options, use -h or --help" >&2 ; exit -1; fi
if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --filter)
          shift
          BANGPY_BUILD_INCLUDE_OP=$1
          echo "test operators: ${BANGPY_BUILD_INCLUDE_OP}."
          shift
          ;;
      --target)
          shift
          MLU_TARGET=$1
          shift
          ;;
      --opsfile)
          shift
          BANGPY_OP_FILE=$1
          if [ ! -f "${BANGPY_OP_FILE}" ]; then
              echo "File \"${BANGPY_OP_FILE}\" dose not exist, please check your \"--opsfile\" value."
              exit -1
          fi
          echo "Test operators in file \"${BANGPY_OP_FILE}\"."
          shift
          ;;
      --only_test)
          BUILD_ENABLE="False"
          shift
          ;;
      --cases_dir)
          shift
          BANGPY_TEST_CASES=$1
          shift
          ;;
      -r | --release)
          shift
          ;;
      -t | --test)
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

if [ -n "${BANGPY_OP_FILE}" ]; then
    if [ -z "${BANGPY_BUILD_INCLUDE_OP}" ]; then
        for op in `cat ${BANGPY_OP_FILE}`
        do
            BANGPY_BUILD_INCLUDE_OP=${BANGPY_BUILD_INCLUDE_OP}"$op"","
        done
        if [ -z "${BANGPY_BUILD_INCLUDE_OP}" ]; then
            echo "File \"${BANGPY_OP_FILE}\" is a empty file!"
            exit -1
        fi
        BANGPY_BUILD_INCLUDE_OP=${BANGPY_BUILD_INCLUDE_OP%?}
    fi
fi

if [ "${BUILD_ENABLE}" == "True" ]; then
    ${BANGPY_UTILS_PATH}"build_operators.sh" ${TEST_ARGS}
fi

# Get all operator names in directory OPS_DIR to generate BANGPY_BUILD_OP_DIR_STRIG.
BANGPY_BUILD_OP_DIR_LIST=()
if [ -z "${BANGPY_BUILD_OP_DIR_LIST}" ]; then
    if [ -n "${BANGPY_BUILD_INCLUDE_OP}" ]; then
        ifs_old=$IFS
        IFS=$','
        for include_op in $(echo "${BANGPY_BUILD_INCLUDE_OP}")
        do
            BANGPY_BUILD_OP_DIR_LIST+=(${include_op})
        done
        IFS=$ifs_old
    else
        for file in $(find ${OPS_DIR} -maxdepth 1 -type d)
        do
            file_name=${file##*/}
            BANGPY_BUILD_OP_DIR_LIST+=(${file_name})
        done
    fi
fi


if [ -n "${BANGPY_BUILD_OP_DIR_LIST}" ]; then
    for i in ${BANGPY_BUILD_OP_DIR_LIST[@]}
    do 
        BANGPY_BUILD_OP_DIR_STRIG=${BANGPY_BUILD_OP_DIR_STRIG}"$i"","
    done
    BANGPY_BUILD_OP_DIR_STRIG=${BANGPY_BUILD_OP_DIR_STRIG%?}
fi

LD_LIBRARY_PATH_BAK=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${BUILD_OUT_DIR}:${LD_LIBRARY_PATH}

# Test
if [ -z ${MLU_TARGET} ]; then
    python3 ${BANGPY_UTILS_PATH}"build_and_test_all_operators.py" -t ${BANGPY_BUILD_OP_DIR_STRIG} "--cases_dir"=${BANGPY_TEST_CASES}
else
    python3 ${BANGPY_UTILS_PATH}"build_and_test_all_operators.py" -t ${BANGPY_BUILD_OP_DIR_STRIG} "--target="${MLU_TARGET} "--cases_dir"=${BANGPY_TEST_CASES}
fi
if [ -n "${BANGPY_BUILD_OP_DIR_LIST}" ]; then
    for i in $(find ${OPS_DIR} -maxdepth 1 -name "\.pytest_cache")
    do
        rm ${i} -rf
    done
    for i in $(find ${OPS_DIR} -maxdepth 2 -name __pycache__)
    do
        rm ${i} -rf
    done
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_BAK}
echo "Test Done!"

