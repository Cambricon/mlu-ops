#! /bin/bash

set -e

rm -rf doxygen

cp -r ../../../bangc-ops/mlu_op.h .

python3 -m pip install pip -U
python3 -m pip install -r requirements.txt

sed -i "s/MLUOP_WIN_API//g" mlu_op.h

doxygen doxyfile

# h2rst tool
python3 -m pip install -r h2rst/requirements.txt
python3 h2rst/main.py mlu_op.h

rm -rdf mlu_op.h

make clean
make html
make latexpdf




