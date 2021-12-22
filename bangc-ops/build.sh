#!/bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
CMAKE=cmake

if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

if [ ! -z "${NEUWARE_HOME}" ]; then
  echo "-- using NEUWARE_HOME = ${NEUWARE_HOME}"
else
  echo "-- NEUWARE_HOME is null, refer README.md to prepare NEUWARE_HOME environment."
  exit -1
fi

pushd ${BUILD_PATH} > /dev/null
  rm -rf *
  echo "-- Build cambricon release test cases."
  ${CMAKE}  ../ -DNEUWARE_HOME="${NEUWARE_HOME}"
popd > /dev/null
${CMAKE} --build build --  -j
