SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BANGC_HOME=${SCRIPT_PATH}/bangc-ops
export BANGPY_HOME=${SCRIPT_PATH}/bangpy-ops
export BANGPY_BUILD_PATH=${BANGPY_HOME}/outs
export CPLUS_INCLUDE_PATH=${BANGPY_HOME}/include/:${CPLUS_INCLUDE_PATH}
export PYTHONPATH=${BANGPY_HOME}:${PYTHONPATH}
if [[ -z ${NEUWARE_HOME} ]]; then
  export NEUWARE_HOME=/usr/local/neuware
  echo "set env 'NEUWARE_HOME' to default: ${NEUWARE_HOME}."
else
  echo "env 'NEUWARE_HOME': ${NEUWARE_HOME}."
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${BANGC_HOME}/build/lib:${LD_LIBRARY_PATH}

if [[ ! -d "${PWD}/.git/" ]]; then
  # pass
  echo "Ready for mlu-ops-building."
else
  if [[ -f "${PWD}/.git/hooks/pre-commit" ]]; then
    rm ${PWD}/.git/hooks/pre-commit
  fi
  echo "-- pre-commit hook inserted to ${PWD}/.git/hooks."
  echo "-- Use git commit -n to bypass pre-commit hook."
  ln -sf ${PWD}/tools/pre-commit ${PWD}/.git/hooks

  if [[ -f "${PWD}/.git/hooks/commit-msg" ]]; then
    rm ${PWD}/.git/hooks/commit-msg
  fi
  echo "-- commit-msg hook inserted to ${PWD}/.git/hooks."
  ln -sf ${PWD}/tools/commit-msg ${PWD}/.git/hooks
  echo "Ready for mlu-ops-building."
fi
