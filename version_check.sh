#!/bin/bash
# Test BANGC source version, used for CI test.
# If you want to run specify operators, refer to README.md.
# You need to run build.sh, before running this script.
set -e

source env.sh

if [ $# == 0 ]; then echo "Have no options, use bash version_check.sh x.y.z"; exit -1; fi

MLU_OPS_VERSION=$1

pushd ${MLUOPS_HOME}/test/version_check
    g++ -std=c++11 version_check.cpp -o version_check -lmluops -lcnrt -lcndrv -L${MLUOPS_HOME}/build/lib -L${NEUWARE_HOME}/lib64 -I${MLUOPS_HOME} -I${NEUWARE_HOME}/include
    ./version_check ${MLU_OPS_VERSION}
popd
