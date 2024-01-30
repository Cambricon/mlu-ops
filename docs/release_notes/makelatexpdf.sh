#! /bin/bash

make clean

make latexpdf
make html
zip -qr _build/cambricon_mlu_ops_release_note.zip _build/html

