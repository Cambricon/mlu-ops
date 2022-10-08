#!/bin/bash

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

cd ./build/
cmake .. -DNEUWARE_HOME="${NEUWARE_HOME}"
cmake --build .


