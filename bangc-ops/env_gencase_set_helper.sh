#!/bin/bash
usage() {
  echo "Common gencase set: source env_gencase_set_helper.sh on comm opname."
  echo "Binary op gencase set: source env_gencase_set_helper.sh on bin opname."
  echo "Turn off gencase: source env_gencase_set_helper.sh off."
}
if [[ $# == 1 && $1 == "off" ]]; then
  export MLUOP_GEN_CASE=0
  export CNNL_GEN_CASE=0
elif [[ $# == 3 && $1 == "on" ]]; then
  if [[ $2 == "comm" ]]; then
    export MLUOP_GEN_CASE=2
    export MLUOP_GEN_CASE_OP_NAME=$3 #op name
    export MLUOP_GEN_CASE_DUMP_DATA=1
  fi
  
  if [[ $2 == "bin" ]]; then
    export CNNL_GEN_CASE=2
    export CNNL_GEN_CASE_OP_NAME=$3 #op name
    export CNNL_GEN_CASE_DUMP_DATA=1
  fi
else
  echo "Bad params!!!."
  usage
fi
