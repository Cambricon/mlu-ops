SCRIPT_PATH="$(cd "../../$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z ${NEUWARE_HOME} ]]; then
  export NEUWARE_HOME=/usr/local/neuware
  echo "set env 'NEUWRAE_HOME' to default: ${NEUWARE_HOME}."
else
  echo "env 'NEUWRAE_HOME': ${NEUWARE_HOME}."
fi

MLU_OP_H=${NEUWARE_HOME}/include/mlu_op.h

if [[ ! -f ${MLU_OP_H} || -h ${MLU_OP_H} ]]; then
  ln -sf ${SCRIPT_PATH}/mlu_op.h ${MLU_OP_H}
  for MLU_OPS_LIB in $(ls ${SCRIPT_PATH}/build/lib/)
  do 
    if [[ ${MLU_OPS_LIB} == libmluops.* ]]; then
      ln -sf ${SCRIPT_PATH}/build/lib/${MLU_OPS_LIB} ${NEUWARE_HOME}/lib64/${MLU_OPS_LIB}
    fi
  done
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:$LD_LIBRARY_PATH

echo "Ready for mlu-ops-building."
