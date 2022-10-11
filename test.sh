#!/bin/bash
# Test BANGC and BANGPy all operators cases, used for CI test.
# If you want to run specify operators, refer to bangc-ops and bangpy-ops README.md.
# You need to run build.sh, before running this script.
set -e

source env.sh
usage () {
    echo "USAGE: test.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help         Print usage"
    echo "      --sub_module=*     Mlu-ops sub_module:[bangc, bangpy]"
    echo "      --cases_dir=*      [Optional]Test cases for bangc-ops test"
    echo
}
if [ $# == 0 ]; then echo "Have no options, use -h or --help"; exit -1; fi
cmdline_args=$(getopt -o h --long sub_module:,cases_dir: -n 'test.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --sub_module)
          shift
          MLU_SUB_MODULE=$1
          shift
          ;;
      --cases_dir)
          shift
          CASES_DIR=$1
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

if [[ ${MLU_SUB_MODULE} == "bangc" ]]; then
  # Test BANGC all operators cases.
  cd bangc-ops/build/test/
  ./mluop_gtest
  
  if [[ -n "${CASES_DIR}" && -a "${CASES_DIR}" ]]; then
    ./mluop_gtest --cases_dir="${CASES_DIR}"
  fi

  cd ../../..
fi

if [[ ${MLU_SUB_MODULE} == "bangpy" ]]; then
  # Check all python file format.
  python3 -m pylint ./bangpy-ops  --rcfile=./bangpy-ops/utils/pylintrc

  # Test BANGPy all operators cases.
  pip3 install prototxt_parser==1.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
  if [[ -n "${CASES_DIR}" && -a "${CASES_DIR}" ]]; then
    ./bangpy-ops/utils/test_operators.sh --only_test --cases_dir="${CASES_DIR}"
  fi
fi
