#!/bin/bash
set e

BANGC_DIR="bangc-docs"
BANGPY_DIR="bangpy-docs"

echo "Generating BANGC-OPS Docs."
pushd ${BANGC_DIR}
bash ./build.sh
mv ./Cambricon*.pdf ../
popd

# echo "Generating BANGPY-OPS Docs."
# pushd ${BANGPY_DIR}
# bash ./build.sh
# mv ./Cambricon*.pdf ../
# popd
