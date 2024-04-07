#! /bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/build
CMAKE=cmake
MLUOP_TARGET_CPU_ARCH=`uname -m`
GEN_SYMBOL_VIS_FILE_PY="./scripts/gen_symbol_visibility_map.py"
MLUOP_SYMBOL_VIS_FILE="symbol_visibility.map"
TARGET_SYMBOL_FILE="mlu_op.h"
PACKAGE_EXTRACT_DIR="dep_libs_extract"

PROG_NAME=$(basename $0)  # current script filename, DO NOT EDIT

#### variable names with default value, and could be overrided from user env ####
export MLUOP_MLU_ARCH_LIST="${MLUOP_MLU_ARCH_LIST}"
export BUILD_MODE=${BUILD_MODE:-release} # release/debug
export MLUOP_BUILD_COVERAGE_TEST=${MLUOP_BUILD_COVERAGE_TEST:-OFF} # ON/OFF coverage mode
export MLUOP_BUILD_ASAN_CHECK=${MLUOP_BUILD_ASAN_CHECK:-OFF} # ON/OFF Address Sanitizer (ASan)
export MLUOP_BUILD_BANG_MEMCHECK=${MLUOP_BUILD_BANG_MEMCHECK:-OFF} # ON/OFF bang memcheck
export MLUOP_BUILD_PREPARE=${MLUOP_BUILD_PREPARE:-ON}
export MLUOP_BUILD_GTEST=${MLUOP_BUILD_GTEST:-ON}
export MLUOP_BUILD_STATIC=${MLUOP_BUILD_STATIC:-OFF}
export BUILD_JOBS="${BUILD_JOBS:-16}" # concurrent build jobs

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
  t:   # release
  j:  # jobs
)
# setup long options, follow alphabetical order
long_args=(
  asan
  coverage
  debug
  enable-bang-memcheck
  filter:
  jobs:
  help
  mlu370 # mlu arch
  mlu590
  no_prepare
  prepare
  disable-gtest
  enable-static
)

add_mlu_arch_support () {
  local target_mlu=$1
  local bang_arch=
  case "$target_mlu" in
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

# check python
python3_version_check() {
  cur_python_ver=(`python3 --version`)
  stat=$?
  if [ ${stat} != 0 ]; then
    echo "Warning: Not found python3"
    echo "If compilation failed, please check python version"
    return
  fi
  required_python_version=$(cat build.property|grep "python"|cut -d ':' -f2|cut -d '"' -f2)
  if [[ "$(printf '%s\n' "${cur_python_ver[1]}" "${required_python_version}" \
        | sort -V | head -n1)" == "${cur_python_ver[1]}" ]]; then
    echo "Warning: python version should no less than ${required_python_version}"
    echo "If compilation failed, please check python version"
    return
  fi
}
python3_version_check

build_requires_version_check() {
  # check build_requires
  python3 version_pre_check.py check_build_requires
}

usage () {
    echo "USAGE: ./independent_build.sh <options>"
    echo
    echo "OPTIONS:"
    echo "    -h, --help                  Print usage."
    echo "    -c, --coverage              Build mlu-ops with coverage test."
    echo "    --asan                      Build with asan check enabled"
    echo "    -d, --debug                 Build mlu-ops with debug mode"
    echo "    --disable-gtest             Build mlu-ops without gtest"
    echo "    --enable-bang-memcheck      Build with cncc '-mllvm -enable-mlisa-sanitizer -Xbang-cnas -O0 -g' arg to enable memcheck"
    echo "    --enable-static             Build mlu-ops static library"
    echo "    --mlu370                    Build for target product MLU370: __BANG_ARCH__ = 372"
    echo "                                                                 __MLU_NRAM_SIZE__ = 768KB"
    echo "                                                                 __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                                 __MLU_SRAM_SIZE__ = 4096KB"
    echo "                                                                 cncc --bang-mlu-arch=mtp_372, cnas --mlu-arch mtp_372"
    echo "    --mlu590                    Build for target product MLU590: __BANG_ARCH__ = 592"
    echo "                                                                 __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                 __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                                 __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                 cncc --bang-mlu-arch=mtp_592, cnas --mlu-arch mtp_592"
    echo "    --filter=*                  Build specified OP only (string with ';' separated)"
    echo "    -j N, --jobs=N              Build for N parallel jobs."
    echo "    --no_prepare                Skip dependency download."
    echo "    --prepare                   Dependency download only."
    echo "    -t                          Build to release."
    echo
}

prepare_cntoolkit () {
  PACKAGE_ARCH="$(uname -m)"
  PACKAGE_SERVER="http://daily.software.cambricon.com"
  PACKAGE_OS="Linux"

  # read build.property, print cntoolkit and cnnl dep-package-version
  build_requires=(`python3 version_pre_check.py get_build_requires`)
  # build_requires is an array(cntoolkit release cntoolkit-version cnnl release cnnl-version)
  arr_modules=(${build_requires[0]} ${build_requires[3]})
  arr_branch=(${build_requires[1]} ${build_requires[4]})
  arr_vers=(${build_requires[2]} ${build_requires[5]})

  n=${#arr_vers[@]}

  sub_pkg_to_extract=(cncc cnas cnperf cngdb cndrv cnrt cnbin cnpapi cndev cntoolkit-cloud)

  if [ -d ${PACKAGE_EXTRACT_DIR} ]; then
    rm -rf ${PACKAGE_EXTRACT_DIR}
  fi
  mkdir -p ${PACKAGE_EXTRACT_DIR}

  if [ -f "/etc/os-release" ]; then
      source /etc/os-release
      if [ ${ID} == "ubuntu" ]; then
          for (( i =0; i < ${n}; i++))
          do
              PACKAGE_DIST="Ubuntu"
              PACKAGE_DIST_VER=${VERSION_ID}
              PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
              REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
              prog_log_info "${arr_modules[$i]} url: ${REAL_PATH}"
              wget -A deb -m -p -E -k -K -np -q --reject-regex 'static'  ${PACKAGE_PATH}

              pushd ${PACKAGE_EXTRACT_DIR} > /dev/null
              for filename in ../${REAL_PATH}*.deb; do
                dpkg -x --force-overwrite ${filename} .
                prog_log_info "extract ${filename}"
                if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                  pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                  for pkg in ${sub_pkg_to_extract[@]}
                  do
                    local fname=$(ls -1 ./var/cntoolkit-${pure_ver}/${pkg}* | grep -E "${pkg}[^[:alnum:]][0-9].*")
                    prog_log_info "extract ${fname}"
                    dpkg -x --force-overwrite ${fname} ./
                  done
                fi
              done
              popd > /dev/null
          done

      elif [ ${ID} == "debian" ]; then
          for (( i =0; i < ${n}; i++))
          do
              PACKAGE_DIST="Debian"
              PACKAGE_DIST_VER=${VERSION_ID}
              PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
              REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
              prog_log_info "${arr_modules[$i]} url: ${REAL_PATH}"
              wget -A deb -m -p -E -k -K -np -q --reject-regex 'static'  ${PACKAGE_PATH}

              pushd ${PACKAGE_EXTRACT_DIR} > /dev/null
              for filename in ../${REAL_PATH}*.deb; do
                prog_log_info "extract ${filename}"
                dpkg -x --force-overwrite ${filename} ./
                if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                  pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                  for pkg in ${sub_pkg_to_extract[@]}
                  do
                    local fname=$(ls -1 ./var/cntoolkit-${pure_ver}/${pkg}* | grep -E "${pkg}[^[:alnum:]][0-9].*")
                    prog_log_info "extract ${fname}"
                    dpkg -x --force-overwrite ${fname} ./
                  done
                fi
              done
              popd > /dev/null
          done

      elif [ ${ID} == "centos" ]; then
          for (( i =0; i < ${n}; i++))
          do
              PACKAGE_DIST="CentOS"
              PACKAGE_DIST_VER=${VERSION_ID}
              PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
              REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
              prog_log_info "${arr_modules[$i]} url: ${REAL_PATH}"
              wget -A rpm -m -p -E -k -K -np -q --reject-regex 'static' ${PACKAGE_PATH}

              pushd ${PACKAGE_EXTRACT_DIR} > /dev/null
              for filename in ../${REAL_PATH}*.rpm; do
                prog_log_info "extract ${filename}"
                rpm2cpio $filename | cpio -u -di
                if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                  pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                  for pkg in ${sub_pkg_to_extract[@]}
                  do
                    local fname=$(ls -1 ./var/cntoolkit-${pure_ver}/${pkg}* | grep -E "${pkg}[^[:alnum:]][0-9].*")
                    prog_log_info "extract ${fname}"
                    rpm2cpio ${fname} | cpio -u -di
                  done
                fi
              done
              popd > /dev/null
          done
      elif [ ${ID} == "kylin" ]; then
          for (( i =0; i < ${n}; i++))
          do
              PACKAGE_DIST="Kylin"
              PACKAGE_DIST_VER=${VERSION_ID}
              PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
              echo "PACKAGE_PATH: $PACKAGE_PATH"
              REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
              echo "real_path $REAL_PATH"
              wget -A rpm -m -p -E -k -K -np --reject-regex 'static' ${PACKAGE_PATH}

              pushd ${PACKAGE_EXTRACT_DIR}
              for filename in ../${REAL_PATH}*.rpm; do
                echo "filename: $filename"
                rpm2cpio $filename | cpio -div
                if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                  pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                  echo "pure_ver: ${pure_ver}"
                  for lib in var/${arr_modules[$i]}"-"${pure_ver}/*.rpm; do
                    rpm2cpio $lib | cpio -div
                  done
                fi
              done
              popd
          done
      fi
  fi

  pushd ${PACKAGE_EXTRACT_DIR} > /dev/null
  ln -sfvT usr/local/neuware neuware
  popd > /dev/null
  export NEUWARE_HOME=${PWD}/${PACKAGE_EXTRACT_DIR}/neuware
  prog_log_note "NEUWARE_HOME:\t${NEUWARE_HOME}"
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
      --enable-bang-memcheck)
          shift
          export MLUOP_BUILD_BANG_MEMCHECK="ON"
          ;;
      --enable-static)
          shift
          export MLUOP_BUILD_STATIC="ON"
          ;;
      --filter)
        shift
        export MLUOP_BUILD_SPECIFIC_OP=$1
        prog_log_note "Build libmluop.so with OP: ${MLUOP_BUILD_SPECIFIC_OP} only."
        shift
        ;;
      -j | --jobs)
        shift
        export BUILD_JOBS=$1
        shift
        ;;
      -t)
        shift
        export RELEASE_TYPE=$1
        export MLUOP_PACKAGE_INFO_SET="ON"
        prog_log_note "Build to generate package."
        shift
        ;;
      --no_prepare)
        shift
        export MLUOP_BUILD_PREPARE="OFF"
        prog_log_note "Skip dependency download."
        ;;
      --prepare)
        shift
        export MLUOP_BUILD_PREPARE_ONLY="ON"
        prog_log_note "Prepare dependency only."
        ;;
      --disable-gtest)
        shift
        export MLUOP_BUILD_GTEST="OFF"
        prog_log_note "Disable gtest."
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
pushd $script_path  > /dev/null
BUILD_VERSION=$(cat build.property|grep "version"|cut -d ':' -f2|cut -d '-' -f1|cut -d '"' -f2|cut -d '.' -f1-3)
popd > /dev/null
MAJOR_VERSION=$(echo ${BUILD_VERSION}|cut -d '-' -f1|cut -d '.' -f1)
prog_log_info "build_version = $BUILD_VERSION"
prog_log_info "major_version = ${MAJOR_VERSION}"
prog_log_info "BUILD_MODE = ${BUILD_MODE}"
prog_log_info "MLUOP_BUILD_COVERAGE_TEST = ${MLUOP_BUILD_COVERAGE_TEST}"
prog_log_info "MLUOP_BUILD_ASAN_CHECK = ${MLUOP_BUILD_ASAN_CHECK}"
prog_log_info "MLUOP_BUILD_BANG_MEMCHECK = ${MLUOP_BUILD_BANG_MEMCHECK}"
prog_log_info "MLUOP_MLU_ARCH_LIST = ${MLUOP_MLU_ARCH_LIST}"
prog_log_info "MLUOP_TARGET_CPU_ARCH = ${MLUOP_TARGET_CPU_ARCH}"
prog_log_info "BUILD_JOBS = ${BUILD_JOBS}"
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
  source /opt/rh/devtoolset-7/enable && prog_log_warn "devtoolset-7 activated" \
    || ( prog_log_warn "source devtoolset-7 failed, ignore this info if you have set env TOOLCHAIN_ROOT, TARGET_C_COMPILER, TARGET_CXX_COMPILER properly (see more details in README.md)" && sleep 4 ) # I hope user will see it
fi

if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

if [ "${MLUOP_BUILD_PREPARE_ONLY}" = "ON" ]; then
  prog_log_info "You have called prepare cntoolkit explicitly."
  prepare_cntoolkit
  exit 0
elif [ "${MLUOP_BUILD_PREPARE}" = "ON" ]; then
  prepare_cntoolkit
  build_requires_version_check
else
  build_requires_version_check
fi

if [ ! -z "${NEUWARE_HOME}" ]; then
  prog_log_info "using NEUWARE_HOME = ${NEUWARE_HOME}"
else
  prog_log_error "You forget to set env 'NEUWARE_HOME', I will guess one."
  sleep 1 # I hope user will see it.
  if [ -x "${PACKAGE_EXTRACT_DIR}/neuware/bin/cncc" ]; then
    export NEUWARE_HOME=${PWD}/${PACKAGE_EXTRACT_DIR}/neuware
  else
    prog_log_error "NEUWARE_HOME is null, refer README.md to prepare NEUWARE_HOME environment."
    exit -1
  fi
  prog_log_warn "NEUWARE_HOME(guessed):\t${NEUWARE_HOME}"
fi
export PATH=${NEUWARE_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:$LD_LIBRARY_PATH

prog_log_info "generate ${MLUOP_SYMBOL_VIS_FILE} file."
prog_log_info "python3 ${GEN_SYMBOL_VIS_FILE_PY} ${BUILD_PATH}/${MLUOP_SYMBOL_VIS_FILE} ${TARGET_SYMBOL_FILE}"
python3 ${GEN_SYMBOL_VIS_FILE_PY} ${BUILD_PATH}/${MLUOP_SYMBOL_VIS_FILE} ${TARGET_SYMBOL_FILE}

pushd ${BUILD_PATH} > /dev/null
  prog_log_info "Rmove cmake cache ${PWD}"
  find . -maxdepth 1 -type f -not -name ${MLUOP_SYMBOL_VIS_FILE} -exec rm -f {} \;
  rm -rf CMakeFiles mlu_op_gtest || :

  ${CMAKE}  ../ -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
                -DNEUWARE_HOME="${NEUWARE_HOME}" \
                -DMLUOP_BUILD_COVERAGE_TEST="${MLUOP_BUILD_COVERAGE_TEST}" \
                -DBUILD_VERSION="${BUILD_VERSION}" \
                -DMAJOR_VERSION="${MAJOR_VERSION}" \
                -DMLUOP_BUILD_ASAN_CHECK="${MLUOP_BUILD_ASAN_CHECK}" \
                -DMLUOP_BUILD_BANG_MEMCHECK="${MLUOP_BUILD_BANG_MEMCHECK}" \
                -DMLUOP_MLU_ARCH_LIST="${MLUOP_MLU_ARCH_LIST}" \
                -DMLUOP_TARGET_CPU_ARCH="${MLUOP_TARGET_CPU_ARCH}" \
                -DMLUOP_BUILD_SPECIFIC_OP="${MLUOP_BUILD_SPECIFIC_OP}" \
                -DMLUOP_SYMBOL_VIS_FILE="${MLUOP_SYMBOL_VIS_FILE}" \
                -DMLUOP_PACKAGE_INFO_SET="${MLUOP_PACKAGE_INFO_SET}" \
                -DMLUOP_BUILD_GTEST="${MLUOP_BUILD_GTEST}" \
                -DMLUOP_BUILD_STATIC="${MLUOP_BUILD_STATIC}"

popd > /dev/null
${CMAKE} --build ${BUILD_PATH} --  -j${BUILD_JOBS}

if [ "${MLUOP_PACKAGE_INFO_SET}" = "ON" ]; then
  BUILD_DIR="build"
  PACKAGE_DIR="package/usr/local/neuware"
  mkdir -p ${PACKAGE_DIR}
  mkdir -p ${PACKAGE_DIR}/include
  mkdir -p ${PACKAGE_DIR}/lib64
  mkdir -p ${PACKAGE_DIR}/samples

  cp -rf ${BUILD_DIR}/lib/libmluops.so* ${PACKAGE_DIR}/lib64
  cp -r samples/* ${PACKAGE_DIR}/samples
  cp mlu_op.h ${PACKAGE_DIR}/include

  TEST_DIR="test_workspace/mluops"
  mkdir -p ${TEST_DIR}/build
  mkdir -p ${TEST_DIR}/lib
  mkdir -p ${TEST_DIR}/test

  cp -rf ${BUILD_DIR}/test ${TEST_DIR}/build
  cp -rf ${BUILD_DIR}/lib/libgtest_shared.a ${TEST_DIR}/lib
  cp -rf ${BUILD_DIR}/lib/libmluop_test_proto.a ${TEST_DIR}/lib
  cp -rf test/* ${TEST_DIR}/test

  DEPS_DIR=`echo ${PACKAGE_SERVER} | awk -F '//' '{print $2}'`
  rm -rf $DEPS_DIR
fi
