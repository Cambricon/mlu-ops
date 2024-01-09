#!/bin/bash

usage () {
    echo "USAGE: test.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      on        Set GenCase environment variables"
    echo "      off       Unset GenCase environment variables"
    echo
    echo "If those ENVs are enabled, the case will be generated."
    echo "Following files will be generated in build/test dir."
    echo "  gen_case/<op_name>/<op_name>_xxx.prototxt."
    echo "The generated case can be test with following command like:"
    echo "  ./mluop_gtest --gtest_filter=*<op_name>* --case_path=./gen_case/<op_name>/<op_name>_xxx.prototxt "
    echo
    echo "For details, please refer to:"
    echo "    https://github.com/Cambricon/mlu-ops/blob/master/docs/Gencase-User-Guide-zh.md"
    echo
}

if [[ $# == 1 && $1 == "off" ]]; then
  export MLUOP_GEN_CASE=0
  export CNNL_GEN_CASE=0
elif [[ $# == 1 && $1 == "on" ]]; then
    export MLUOP_GEN_CASE=2
    export MLUOP_GEN_CASE_DUMP_DATA=1
    export CNNL_GEN_CASE=2
    export CNNL_GEN_CASE_DUMP_DATA=1
else
  echo "Bad params!!!."
  usage
fi
