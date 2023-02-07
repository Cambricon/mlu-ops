#!/bin/bash
# Test BANGC and BANGPy source version, used for CI test.
# If you want to run specify operators, refer to bangc-ops and bangpy-ops README.md.
# You need to run build.sh, before running this script.
set -e

source env.sh

if [ $# == 0 ]; then echo "Have no options, use bash version_check.sh x.y.z"; exit -1; fi

MLU_OPS_VERSION=$1

pushd ${BANGC_HOME}/test/version_check
    g++ -std=c++11 version_check.cpp -o version_check -lmluops -lcnrt -lcndrv -L${BANGC_HOME}/build/lib -L${NEUWARE_HOME}/lib64 -I${BANGC_HOME} -I${NEUWARE_HOME}/include
    ./version_check ${MLU_OPS_VERSION}
popd
