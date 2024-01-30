#!/bin/bash
set -e
rm -rf *.pdf
rm -rf *.zip

API_GUIDE_DIR="api_guide"
RELEASE_NOTE_DIR="release_notes"
USER_GUIDE_DIR="user_guide"

echo "Generating MLU-OPS Docs: API Guide..."
pushd ${API_GUIDE_DIR}
bash ./makelatexpdf.sh
if [[ $? -ne 0 ]]; then
  echo "Generating MLU-OPS Docs: API Guide FAIL!!!"
  exit 1
fi
cp ./_build/latex/Cambricon*.pdf ../
cp ./_build/cambricon*.zip ../
popd

echo "Generating MLU-OPS Docs: Release Notes..."
pushd ${RELEASE_NOTE_DIR}
bash ./makelatexpdf.sh
if [[ $? -ne 0 ]]; then
  echo "Generating MLU-OPS Docs: Release Notes FAIL!!!"
  exit 1
fi
cp ./_build/latex/Cambricon*.pdf ../
cp ./_build/cambricon*.zip ../
popd

echo "Generating MLU-OPS Docs: User Guide..."
pushd ${USER_GUIDE_DIR}
bash ./makelatexpdf.sh
if [[ $? -ne 0 ]]; then
  echo "Generating MLU-OPS Docs: User Guide FAIL!!!"
  exit 1
fi
cp ./_build/latex/Cambricon*.pdf ../
cp ./_build/cambricon*.zip ../
popd
