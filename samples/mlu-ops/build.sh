#!/bin/bash

#check compiler version and consider activate devtoolset for CentOS 7
if [ "$OS_RELEASE_ID" = "centos" -a "$OS_RELEASE_VERSION_ID" = "7" ]; then
  if [ ! -f "/opt/rh/devtoolset-8/enable" ]; then
    echo "You are using CentOS 8 but without 'devtoolset-7' installed."
    echo "You should use docker image, or prepare devtoolset-7 by yourself."
    sleep 1 # I hope user will see it
  fi
fi

if [[ "$(g++ --version | head -n1 | awk '{ print $3 }' | cut -d '.' -f1)" -lt "5" ]]; then
  echo "we do not support g++<5, try to activate devtoolset-8 env"
  source /opt/rh/devtoolset-8/enable && echo "devtoolset-8 activated" \
    || ( echo "source devtoolset-8 failed, ignore this info if you have set env TOOLCHAIN_ROOT, TARGET_C_COMPILER, TARGET_CXX_COMPILER properly (see more details in README.md)" && sleep 4 ) # I hope user will see it
fi

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
if [ -d "$BUILD_PATH" ]; then
  rm -r ${BUILD_PATH}
fi

mkdir $BUILD_PATH

cd ./build/

cmake .. -DNEUWARE_HOME="${NEUWARE_HOME}"
cmake --build .
