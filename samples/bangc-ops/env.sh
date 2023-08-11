SCRIPT_PATH="$(cd "../../$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BANGC_HOME=${SCRIPT_PATH}/bangc-ops

if [[ -z ${NEUWARE_HOME} ]]; then
  export NEUWARE_HOME=/usr/local/neuware
  echo "set env 'NEUWRAE_HOME' to default: ${NEUWARE_HOME}."
else
  echo "env 'NEUWRAE_HOME': ${NEUWARE_HOME}."
fi

MLU_OP_H=${NEUWARE_HOME}/include/mlu_op.h

if [[ ! -f ${MLU_OP_H} || -h ${MLU_OP_H} ]]; then
  ln -sf ${BANGC_HOME}/mlu_op.h ${MLU_OP_H}
  for MLU_OPS_SO in $(ls ${BANGC_HOME}/build/lib/)
  do 
    if [[ ${MLU_OPS_SO} == libmluops.so* ]]; then
      ln -sf ${BANGC_HOME}/build/lib/${MLU_OPS_SO} ${NEUWARE_HOME}/lib64/${MLU_OPS_SO}
    fi
  done
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:$LD_LIBRARY_PATH

echo "Ready for mlu-ops-building."
