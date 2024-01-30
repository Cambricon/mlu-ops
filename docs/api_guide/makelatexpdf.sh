#! /bin/bash

set -e

rm -rf doxygen

cp -r ../../mlu_op.h .

python3 -m pip install pip -U
python3 -m pip install -r requirements.txt

sed -i "s/MLUOP_WIN_API//g" mlu_op.h

doxygen doxyfile

# h2rst tool
python3 -m pip install -r h2rst/requirements.txt
python3 h2rst/main.py mlu_op.h

sed -i 's/deprecated_apis//g' ./api.rst

rm -rdf mlu_op.h
if [ ! -d source/h2rst ]; then 
 mkdir -p source/h2rst
fi
mv *.rst source/h2rst

make clean
make html
zip -qr -P"Cambricon@doc123456" _build/cambricon_mlu_ops_developer_guide.zip _build/html
make latexpdf

rm -rdf source/h2rst



