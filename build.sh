#!/bin/bash
# Build BANGC and BANGPy all operators, used for CI test.
set -e
usage () {
    echo "USAGE: test.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help         Print usage"
    echo "      --sub_module=*     Mlu-ops submodule:[bangc, bangpy]"
    echo
}
if [ $# == 0 ]; then echo "Have no options, use -h or --help"; exit -1; fi
cmdline_args=$(getopt -o h --long sub_module: -n 'test.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --sub_module)
          shift
          MLU_SUB_MODULE=$1
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

source env.sh

# 1.build BANGC ops
if [[ ${MLU_SUB_MODULE} == "bangc" ]]; then
  cd bangc-ops
  ./build.sh
  cd ..
fi

# 2.build BANGPy ops
if [[ ${MLU_SUB_MODULE} == "bangpy" ]]; then
  pip3 install prototxt_parser==1.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
  cd bangpy-ops/utils
  ./build_operators.sh
  cd ../..
fi
