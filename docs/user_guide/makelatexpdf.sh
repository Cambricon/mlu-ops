#! /bin/bash

make clean

make latexpdf
make html
zip -qr -P"Cambricon@doc123456" _build/cambricon_mlu_ops_user_guide.zip _build/html

