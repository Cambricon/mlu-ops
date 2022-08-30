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
    echo "      --target=*         Test mlu target:[mlu270, mlu370-s4, mlu220-m2, mlu290]"
    echo
}
if [ $# == 0 ]; then echo "Have no options, use -h or --help"; exit -1; fi
cmdline_args=$(getopt -o h --long sub_module:,target: -n 'test.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --sub_module)
          shift
          MLU_SUB_MODULE=$1
          shift
          ;;
      --target)
          shift
          MLU_TARGET=$1
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

echo "MLU_SUB_MODULE: ${MLU_SUB_MODULE}"
echo "MLU_TARGET: ${MLU_TARGET}"

if [[ ${MLU_SUB_MODULE} == "bangc" ]]; then
  # Test BANGC all operators cases.
  cd bangc-ops/build/test/
  ./mluop_gtest
  cd ../../..
fi

if [[ ${MLU_SUB_MODULE} == "bangpy" ]]; then
  # Check all python file format.
  python3 -m pylint ./bangpy-ops  --rcfile=./bangpy-ops/utils/pylintrc

  # Test BANGPy all operators cases.
  ./bangpy-ops/utils/test_operators.sh --only_test --target=${MLU_TARGET}
fi
