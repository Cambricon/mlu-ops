SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BANGC_HOME=${SCRIPT_PATH}/bangc-ops

if [[ -z ${NEUWARE_HOME} ]]; then
  export NEUWARE_HOME=/usr/local/neuware
  echo "set env 'NEUWRAE_HOME' to default: ${NEUWARE_HOME}."
else
  echo "env 'NEUWRAE_HOME': ${NEUWARE_HOME}."
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${BANGC_HOME}/build/lib:${LD_LIBRARY_PATH}

echo "Ready for mlu-ops-building."
