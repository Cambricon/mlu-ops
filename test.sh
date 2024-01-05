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
    echo "      --cases_dir=*      [Optional]Test cases for bangc-ops test"
    echo
}
if [ $# == 0 ]; then echo "Have no options, use -h or --help"; exit -1; fi
cmdline_args=$(getopt -o h --long cases_dir: -n 'test.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $# != 0 ]; then
  while true; do
    case "$1" in
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

# Test all operators cases.
cd build/test/
./mluop_gtest
  
if [[ -n "${CASES_DIR}" && -a "${CASES_DIR}" ]]; then
    ./mluop_gtest --cases_dir="${CASES_DIR}"
fi
