#!/bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
CMAKE=cmake
MLUOPS_TARGET_CPU_ARCH=`uname -m`

usage () {
    echo "USAGE: ./build.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help         Print usage."
    echo "      -c, --coverage     Build mluops with coverage test."
    echo
}

cmdline_args=$(getopt -o ch -n 'build.sh' -- "$@")
eval set -- "$cmdline_args"

script_path=`dirname $0`

pushd $script_path/../
BUILD_VERSION=$(cat build.property|grep "version"|cut -d ':' -f2|cut -d '-' -f1|cut -d '"' -f2|cut -d '.' -f1-3)
popd
echo "build_version: $BUILD_VERSION"

MAJOR_VERSION=$(echo ${BUILD_VERSION}|cut -d '-' -f1|cut -d '.' -f1)
echo "major_version=${MAJOR_VERSION}"


if [ $# != 0 ]; then
  while true; do
    case "$1" in
      -c | --coverage)
          shift
          export MLUOP_BUILD_COVERAGE_TEST="ON"
          ;;
      --asan)
          shift
          export MLUOP_BUILD_ASAN_CHECK="ON"
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

#check compiler version and consider activate devtoolset for CentOS 7
if [ "$OS_RELEASE_ID" = "centos" -a "$OS_RELEASE_VERSION_ID" = "7" ]; then
  if [ ! -f "/opt/rh/devtoolset-7/enable" ]; then
    echo "You are using CentOS 7 but without 'devtoolset-7' installed."
    echo "You should use docker image, or prepare devtoolset-7 by yourself."
    sleep 1 # I hope user will see it
  fi
fi

if [[ "$(g++ --version | head -n1 | awk '{ print $3 }' | cut -d '.' -f1)" < "5" ]]; then
  echo "we do not support g++<5, try to activate devtoolset-7 env"
  source /opt/rh/devtoolset-7/enable && echo "devtoolset-7 activated" \
    || ( echo "source devtoolset-7 failed, ignore this info if you have set env TOOLCHAIN_ROOT, TARGET_C_COMPILER, TARGET_CXX_COMPILER properly (see more details in README.md)" && sleep 4 ) # I hope user will see it
fi

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
  if [[ ${MLUOP_BUILD_COVERAGE_TEST} == "ON" ]]; then
    echo "-- Build cambricon coverage test cases."
    ${CMAKE}  ../ -DNEUWARE_HOME="${NEUWARE_HOME}" \
                  -DMLUOP_BUILD_COVERAGE_TEST="${MLUOP_BUILD_COVERAGE_TEST}" \
                  -DMLUOPS_TARGET_CPU_ARCH="${MLUOPS_TARGET_CPU_ARCH}"
  else
    echo "-- Build cambricon release test cases."
    ${CMAKE}  ../ -DNEUWARE_HOME="${NEUWARE_HOME}" \
                  -DBUILD_VERSION="${BUILD_VERSION}" \
                  -DMAJOR_VERSION="${MAJOR_VERSION}" \
                  -DMLUOPS_TARGET_CPU_ARCH="${MLUOPS_TARGET_CPU_ARCH}"
  fi

  if [[ ${MLUOP_BUILD_ASAN_CHECK} == "ON" ]]; then
    echo "-- Build cambricon ASAN leak check."
    ${CMAKE}  ../ -DNEUWARE_HOME="${NEUWARE_HOME}" \
                  -DMLUOP_BUILD_ASAN_CHECK="${MLUOP_BUILD_ASAN_CHECK}" \
                  -DMLUOPS_TARGET_CPU_ARCH="${MLUOPS_TARGET_CPU_ARCH}"
  fi
popd > /dev/null
${CMAKE} --build build --  -j
