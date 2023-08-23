#!/bin/bash

usage () {
    echo "USAGE: test.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      on        Set dump data environment variables"
    echo "      off       Unset dump data environment variables"
    echo
    echo "If MLUOP_GTEST_DUMP_DATA is enabled, the test data will be dumped."
    echo "Following files will be generated in build/test dir."
    echo "    baseline_output_xx, mlu_output_xx, hex_inputx, hex_outputx."
    echo
    echo "For details, please refer to:"
    echo "    https://github.com/Cambricon/mlu-ops/blob/master/README.md"
    echo
}
if [[ $# == 1 && $1 == "on" ]]; then
    export MLUOP_GTEST_DUMP_DATA=ON
    export MLUOP_MIN_VLOG_LEVEL=5
elif [[ $# == 1 && $1 == "off" ]]; then
    export MLUOP_GTEST_DUMP_DATA=OFF
    export MLUOP_MIN_VLOG_LEVEL=0
else
  echo "Bad params!!!."
  echo
  usage
fi