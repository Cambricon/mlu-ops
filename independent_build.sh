#! /bin/bash
set -e

usage() {
  echo "Usage: $0 -t RELEASE_TYPE"
  echo "-t  release or daily. By default release. If release, need to give module version"
}

export MLUOP_PACKAGE_INFO_SET="ON"

./build.sh --sub_module=bangc

if [ $? != 0 ]; then
  echo "[independent_build.sh] bangc-ops-build failed"
  exit 1
fi
echo "[independent_build.sh] bangc-ops-build success"

BUILD_DIR="bangc-ops/build"
BANGCOPS_DIR="bangc-ops"
PACKAGE_DIR="package/usr/local/neuware"
mkdir -p ${PACKAGE_DIR}
mkdir -p ${PACKAGE_DIR}/include
mkdir -p ${PACKAGE_DIR}/lib64

cp -rf ${BUILD_DIR}/lib/libmluops.so* ${PACKAGE_DIR}/lib64
cp bangc-ops/mlu_op.h ${PACKAGE_DIR}/include

TEST_DIR="test_workspace/mluops"
mkdir -p ${TEST_DIR}/build
mkdir -p ${TEST_DIR}/lib
mkdir -p ${TEST_DIR}/test

cp -rf ${BUILD_DIR}/test ${TEST_DIR}/build
cp -rf ${BUILD_DIR}/lib/libgtest_shared.a ${TEST_DIR}/lib
cp -rf ${BUILD_DIR}/lib/libmluop_test_proto.a ${TEST_DIR}/lib
cp -rf ${BANGCOPS_DIR}/test/* ${TEST_DIR}/test

DEPS_DIR=`echo ${PACKAGE_SERVER} | awk -F '//' '{print $2}'`
rm -rf $DEPS_DIR

rm dependency.txt
