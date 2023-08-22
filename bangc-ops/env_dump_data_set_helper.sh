#!/bin/bash
usage() {
  echo "Dump data set: source env_dump_data_set_helper.sh on."
  echo "Turn off dump data: source env_dump_data_set_helper.sh off."
}
if [[ $# == 1 ]]; then
  if [[ $1 == "on"  ]]
    export MLUOP_GTEST_DUMP_DATA=ON
    export MLUOP_MIN_LOG_LEVEL=0
    export MLUOP_MIN_VLOG_LEVEL=5
  else
    export MLUOP_GTEST_DUMP_DATA=OFF
    export MLUOP_MIN_LOG_LEVEL=0
    export MLUOP_MIN_VLOG_LEVEL=0
  fi
else
  echo "Bad params!!!."
  usage
fi