#!/bin/bash
set e

API_GUIDE_DIR="api_guide"
RELEASE_NOTE_DIR="release_notes"
USER_GUIDE_DIR="user_guide"

echo "Generating BANGC-OPS Docs: API Guide..."
pushd ${API_GUIDE_DIR}
bash ./makelatexpdf.sh
cp ./_build/latex/Cambricon*.pdf ../
popd

echo "Generating BANGC-OPS Docs: Release Notes..."
pushd ${RELEASE_NOTE_DIR}
bash ./makelatexpdf.sh
cp ./_build/latex/Cambricon*.pdf ../
popd

echo "Generating BANGC-OPS Docs: User Guide..."
pushd ${USER_GUIDE_DIR}
bash ./makelatexpdf.sh
cp ./_build/latex/Cambricon*.pdf ../
popd
