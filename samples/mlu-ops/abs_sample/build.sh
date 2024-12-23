#!/bin/bash

#check compiler version and consider activate devtoolset for CentOS 7
if [ "$OS_RELEASE_ID" = "centos" -a "$OS_RELEASE_VERSION_ID" = "7" ]; then
  if [ ! -f "/opt/rh/devtoolset-7/enable" ]; then
    echo "You are using CentOS 7 but without 'devtoolset-7' installed."
    echo "You should use docker image, or prepare devtoolset-7 by yourself."
    sleep 1 # I hope user will see it
  fi
fi

if [[ "$(g++ --version | head -n1 | awk '{ print $3 }' | cut -d '.' -f1)" -lt "5" ]]; then
  echo "we do not support g++<5, try to activate devtoolset-7 env"
  source /opt/rh/devtoolset-7/enable && echo "devtoolset-7 activated" \
    || ( echo "source devtoolset-7 failed, ignore this info if you have set env TOOLCHAIN_ROOT, TARGET_C_COMPILER, TARGET_CXX_COMPILER properly (see more details in README.md)" && sleep 4 ) # I hope user will see it
fi

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
if [[ ! -d "$BUILD_PATH" ]]; then
  mkdir "$BUILD_PATH"
fi

if [[ -z ${MLUOPS_STATIC} ]]; then
  MLUOPS_STATIC=OFF
fi

cd ./build/
cmake .. -DNEUWARE_HOME="${NEUWARE_HOME}" -DMLUOPS_STATIC="${MLUOPS_STATIC}"
cmake --build .

