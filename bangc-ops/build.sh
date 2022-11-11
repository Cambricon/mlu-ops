#!/bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
CMAKE=cmake
MLUOP_TARGET_CPU_ARCH=`uname -m`

PROG_NAME=$(basename $0)  # current script filename, DO NOT EDIT

#### variable names with default value, and could be overrided from user env ####
export MLUOP_MLU_ARCH_LIST="${MLUOP_MLU_ARCH_LIST}"
export BUILD_MODE=${BUILD_MODE:-release} # release/debug
export MLUOP_BUILD_COVERAGE_TEST=${MLUOP_BUILD_COVERAGE_TEST:-OFF} # ON/OFF coverage mode
export MLUOP_BUILD_ASAN_CHECK=${MLUOP_BUILD_ASAN_CHECK:-OFF} # ON/OFF Address Sanitizer (ASan)

# import common method like `download_pkg`, `get_json_val`, `common_extract`, etc
. ./scripts/utils.bash

#####################################
## usage and command line parsing  ##
#####################################

# setup short options, follow alphabetical order
short_args=(
  c   # coverage
  h   # help
  d   # debug
)
# setup long options, follow alphabetical order
long_args=(
  asan
  coverage
  debug
  filter:
  help
  mlu270 # mlu arch
  mlu290 # mlu arch
  mlu370 # mlu arch
  mlu590
)

add_mlu_arch_support () {
  local target_mlu=$1
  local bang_arch=
  case "$target_mlu" in
    --mlu270)
      bang_arch="mtp_270;"
      ;;
    --mlu290)
      bang_arch="mtp_290;"
      ;;
    --mlu370)
      bang_arch="mtp_372;"
      ;;
    --mlu590)
      bang_arch="mtp_592;"
      ;;
    *)
      ;;
  esac
  if [ -n "${bang_arch}" ]; then
    _arg_shift=1
  fi

  MLUOP_MLU_ARCH_LIST+=${bang_arch}
}

usage () {
    echo "USAGE: ./build.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -h, --help         Print usage."
    echo "      -c, --coverage     Build bangc-ops with coverage test."
    echo "      --asan             Build with asan check enabled"
    echo "      -d, --debug        Build bangc-ops with debug mode"
    echo "      --mlu270           Build for target product MLU270: __BANG_ARCH__ = 270"
    echo "                                                          __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                          __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                          __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                          cncc --bang-mlu-arch=mtp_270, cnas --mlu-arch mtp_270"
    echo "      --mlu290           Build for target product MLU290: __BANG_ARCH__ = 290"
    echo "                                                          __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                          __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                          __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                          cncc --bang-mlu-arch=mtp_290, cnas --mlu-arch mtp_290"
    echo "      --mlu370           Build for target product MLU370: __BANG_ARCH__ = 372"
    echo "                                                          __MLU_NRAM_SIZE__ = 768KB"
    echo "                                                          __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                          __MLU_SRAM_SIZE__ = 4096KB"
    echo "                                                          cncc --bang-mlu-arch=mtp_372, cnas --mlu-arch mtp_372"
    echo "      --mlu590           Build for target product MLU590: __BANG_ARCH__ = 592"
    echo "                                                          __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                          __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                          __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                          cncc --bang-mlu-arch=mtp_592, cnas --mlu-arch mtp_592"
    echo "      --filter=*         Build specified OP only (string with ';' separated)"
    echo
}

_short_args_joined=$(join_by , ${short_args[@]})
_long_args_joined=$(join_by , ${long_args[@]})

# parse arguments and setup internal env
_cmdline_args=$(getopt -o $_short_args_joined --long $_long_args_joined -n $PROG_NAME -- "$@" || usage -1)
eval set -- "$_cmdline_args"

if [ $# != 0 ]; then
  while true; do
    case "$1" in
      -c | --coverage)
          shift
          export MLUOP_BUILD_COVERAGE_TEST="ON"
          ;;
      --asan)
          shift
          export MLUOP_BUILD_ASAN_CHECK="ON"
          ;;
      -h | --help)
          usage
          exit 0
          ;;
      -d | --debug)
          shift
          export BUILD_MODE="debug"
          prog_log_note "Using debug build mode"
          ;;
      --filter)
        shift
        export MLUOP_BUILD_SPECIFIC_OP=$1
        prog_log_note "Build libmluop.so with OP: ${MLUOP_BUILD_SPECIFIC_OP} only."
        shift
        ;;
      --)
        shift
        if [ $# -gt 0 ]; then
          prog_log_warn "ignore '--', command line args meaning has been changed, check README.md for more details"
          sleep 1 # I hope user will see it
        fi
        ;;
      *)
        if [ $# -eq 0 ]; then
          break
        fi
        _arg_shift=0
        add_mlu_arch_support $1

        if [ "$_arg_shift" = "1" ]; then
          shift
        else
          _cmdline_args=$(getopt -o $_short_args_joined --long $_long_args_joined -n $PROG_NAME -- "$@" || usage -1)
          eval set -- "$_cmdline_args"
        fi
        ;;
    esac
  done
fi

script_path=`dirname $0`
pushd $script_path/../
BUILD_VERSION=$(cat build.property|grep "version"|cut -d ':' -f2|cut -d '-' -f1|cut -d '"' -f2|cut -d '.' -f1-3)
popd
MAJOR_VERSION=$(echo ${BUILD_VERSION}|cut -d '-' -f1|cut -d '.' -f1)
prog_log_info "build_version = $BUILD_VERSION"
prog_log_info "major_version = ${MAJOR_VERSION}"
prog_log_info "BUILD_MODE = ${BUILD_MODE}"
prog_log_info "MLUOP_BUILD_COVERAGE_TEST = ${MLUOP_BUILD_COVERAGE_TEST}"
prog_log_info "MLUOP_BUILD_ASAN_CHECK = ${MLUOP_BUILD_ASAN_CHECK}"
prog_log_info "MLUOP_MLU_ARCH_LIST = ${MLUOP_MLU_ARCH_LIST}"
prog_log_info "MLUOP_TARGET_CPU_ARCH = ${MLUOP_TARGET_CPU_ARCH}"
#check compiler version and consider activate devtoolset for CentOS 7
if [ "$OS_RELEASE_ID" = "centos" -a "$OS_RELEASE_VERSION_ID" = "7" ]; then
  if [ ! -f "/opt/rh/devtoolset-7/enable" ]; then
    prog_log_warn "You are using CentOS 7 but without 'devtoolset-7' installed."
    prog_log_warn "You should use docker image, or prepare devtoolset-7 by yourself."
    sleep 1 # I hope user will see it
  fi
fi

if [[ "$(g++ --version | head -n1 | awk '{ print $3 }' | cut -d '.' -f1)" < "5" ]]; then
  prog_log_note "we do not support g++<5, try to activate devtoolset-7 env"
  source /opt/rh/devtoolset-7/enable && echo "devtoolset-7 activated" \
    || ( echo "source devtoolset-7 failed, ignore this info if you have set env TOOLCHAIN_ROOT, TARGET_C_COMPILER, TARGET_CXX_COMPILER properly (see more details in README.md)" && sleep 4 ) # I hope user will see it
fi

if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

if [ ! -z "${NEUWARE_HOME}" ]; then
  prog_log_info "using NEUWARE_HOME = ${NEUWARE_HOME}"
else
  prog_log_error "NEUWARE_HOME is null, refer README.md to prepare NEUWARE_HOME environment."
  exit -1
fi

pushd ${BUILD_PATH} > /dev/null
  rm -rf *
  ${CMAKE}  ../ -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
                -DNEUWARE_HOME="${NEUWARE_HOME}" \
                -DMLUOP_BUILD_COVERAGE_TEST="${MLUOP_BUILD_COVERAGE_TEST}" \
                -DBUILD_VERSION="${BUILD_VERSION}" \
                -DMAJOR_VERSION="${MAJOR_VERSION}" \
                -DMLUOP_BUILD_ASAN_CHECK="${MLUOP_BUILD_ASAN_CHECK}" \
                -DMLUOP_MLU_ARCH_LIST="${MLUOP_MLU_ARCH_LIST}" \
                -DMLUOP_TARGET_CPU_ARCH="${MLUOP_TARGET_CPU_ARCH}" \
                -DMLUOP_BUILD_SPECIFIC_OP="${MLUOP_BUILD_SPECIFIC_OP}"

popd > /dev/null
${CMAKE} --build build --  -j
