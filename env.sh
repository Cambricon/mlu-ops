SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BANGPY_HOME=${SCRIPT_PATH}/bangpy-ops
export BANGPY_BUILD_PATH=${BANGPY_HOME}/outs
export CPLUS_INCLUDE_PATH=${BANGPY_HOME}/include/:${CPLUS_INCLUDE_PATH}
export NEUWARE_HOME=/usr/local/neuware
export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}
