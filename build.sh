#!/bin/bash
# Build BANGC and BANGPy all operators, used for CI test.
set -e

source env.sh

# 1.build BANGC ops
cd bangc-ops
./build.sh
cd ..

# 2.build BANGPy ops
#cd bangpy-ops/utils
#./build_operators.sh
#cd ../..
