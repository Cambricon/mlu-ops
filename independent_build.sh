#! /bin/bash
set -e

usage() {
  echo "Usage: $0 -t RELEASE_TYPE"
  echo "-t  release or daily. By default release. If release, need to give module version"
}

python2 json_parser.py

output='dependency.txt'
RELEASE_TYPE="release"
MODULE_VERSION=""
PACKAGE_ARCH="$(uname -m)"
# get the package type
while getopts "t:" opt
do
  case $opt in
    t) RELEASE_TYPE=$OPTARG;;
    *) usage; exit 1;;
  esac
done
echo "======================================="
echo "RELEASE_TYPE: "$RELEASE_TYPE
echo "========================================"

if [ "$RELEASE_TYPE" = "release" ]; then
  build_version=$(cat build.property|grep "version"|cut -d ':' -f2|cut -d '-' -f1|cut -d '"' -f2|cut -d '.' -f1-3)
else
  shortversion=$(cat build.property|grep "version"|cut -d ':' -f2|cut -d '-' -f1|cut -d '"' -f2|cut -d '.' -f1-3)
  datetime=$(date +'%Y%m%d_%H%M%S')
  commit_commithash=$(git describe --tags --always)
  build_version=${shortversion}-${datetime}-${commit_commithash}
fi
echo "build version $build_version for $RELEASE_TYPE"

# dep-package-version
PACKAGE_MODULES=`cat $output | awk -F ':' '{print $1}'`
echo "PACKAGE_MODULES: $PACKAGE_MODULES"

PACKAGE_BRANCH=`cat $output | awk -F ':' '{print $2}'`
echo "PACKAGE_BRANCH: $PACKAGE_BRANCH"

PACKAGE_MODULE_VERS=`cat $output | awk -F ':' '{print $3}'`
echo "PACKAGE_MODULE_VERS: $PACKAGE_MODULE_VERS"

PACKAGE_SERVER="http://daily.software.cambricon.com"
PACKAGE_OS="Linux"


arr_modules=(`echo $PACKAGE_MODULES`)
arr_branch=(`echo $PACKAGE_BRANCH`)
arr_vers=(`echo $PACKAGE_MODULE_VERS`)

n=${#arr_vers[@]}

echo "number of dependency: $n"
PACKAGE_EXTRACT_DIR="dep_libs_extract"

export NEUWARE_HOME=${PWD}/${PACKAGE_EXTRACT_DIR}/usr/local/neuware
export PATH=${PWD}/${PACKAGE_EXTRACT_DIR}/usr/local/neuware/bin:$PATH
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:$LD_LIBRARY_PATH


if [ -f "/etc/os-release" ]; then
    source /etc/os-release
    if [ ${ID} == "ubuntu" ]; then

        for (( i =0; i < ${n}; i++))
        do
            PACKAGE_DIST="Ubuntu"
            PACKAGE_DIST_VER=${VERSION_ID}
            PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
            echo "PACKAGE_PATH: $PACKAGE_PATH"
            REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
            echo "real_path $REAL_PATH"
            wget -A deb -m -p -E -k -K -np ${PACKAGE_PATH}
            mkdir -p ${PACKAGE_EXTRACT_DIR}
            pushd ${PACKAGE_EXTRACT_DIR}
            for filename in ../${REAL_PATH}*.deb; do
              echo "filename: $filename"
              dpkg -X ${filename} .
              echo "test succeuss"
              if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                echo "pure_ver: ${pure_ver}"
                for lib in var/${arr_modules[$i]}"-"${pure_ver}/*.deb; do
                  dpkg -X $lib ./
                done
              fi
            done
            popd
        done

    elif [ ${ID} == "debian" ]; then
        for (( i =0; i < ${n}; i++))
        do
            PACKAGE_DIST="Debian"
            PACKAGE_DIST_VER=${VERSION_ID}
            echo "PACKAGE_FILE: $PACKAGE_FILE"
            PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
            echo "PACKAGE_PATH: $PACKAGE_PATH"
            REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
            echo "real_path $REAL_PATH"
            wget -A deb -m -p -E -k -K -np ${PACKAGE_PATH}
            mkdir -p ${PACKAGE_EXTRACT_DIR}
            pushd ${PACKAGE_EXTRACT_DIR}
            for filename in ../${REAL_PATH}*.deb; do
              echo "filename: $filename"
              dpkg -X ${filename} ./
              if [ ${arr_modules[$i]} == "cntoolkit" ]; then
                pure_ver=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
                echo "pure_ver: ${pure_ver}"
                for lib in var/${arr_modules[$i]}"-"${pure_ver}/*.deb; do
                  dpkg -X $lib ./
                done
              fi
            done
            popd
        done

    elif [ ${ID} == "centos" ]; then
        for (( i =0; i < ${n}; i++))
        do
            PACKAGE_DIST="CentOS"
            PACKAGE_DIST_VER=${VERSION_ID}
            PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
            echo "PACKAGE_PATH: $PACKAGE_PATH"
            REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
            echo "real_path $REAL_PATH"
            wget -A rpm -m -p -E -k -K -np ${PACKAGE_PATH}
            mkdir -p ${PACKAGE_EXTRACT_DIR}
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
    elif [ ${ID} == "kylin" ]; then
        for (( i =0; i < ${n}; i++))
        do
            PACKAGE_DIST="Kylin"
            PACKAGE_DIST_VER=${VERSION_ID}
            PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"
            echo "PACKAGE_PATH: $PACKAGE_PATH"
            REAL_PATH=`echo ${PACKAGE_PATH} | awk -F '//' '{print $2}'`
            echo "real_path $REAL_PATH"
            wget -A rpm -m -p -E -k -K -np ${PACKAGE_PATH}
            mkdir -p ${PACKAGE_EXTRACT_DIR}
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

./build.sh --sub_module=bangc

if [ $? != 0 ]; then
  echo "[independent_build.sh] bangc-ops-build failed"
  exit 1
fi
echo "[independent_build.sh] bangc-ops-build success"

BUILD_DIR="bangc-ops/build"
BANGCOPS_DIR="bangc-ops"
PACKAGE_DIR="package/usr/local/neuware"
mkdir -p ${PACKAGE_DIR}
mkdir -p ${PACKAGE_DIR}/include
mkdir -p ${PACKAGE_DIR}/lib64

cp -rf ${BUILD_DIR}/lib/libmluops.so* ${PACKAGE_DIR}/lib64
cp bangc-ops/mlu_op.h bangc-ops/mlu_op_kernel.h ${PACKAGE_DIR}/include

TEST_DIR="test_workspace/mluops"
mkdir -p ${TEST_DIR}/build
mkdir -p ${TEST_DIR}/lib
mkdir -p ${TEST_DIR}/test

cp -rf ${BUILD_DIR}/test ${TEST_DIR}/build
cp -rf ${BUILD_DIR}/lib/libgtest_shared.a ${TEST_DIR}/lib
cp -rf ${BUILD_DIR}/lib/libmluop_test_proto.a ${TEST_DIR}/lib
cp -rf ${BANGCOPS_DIR}/test/* ${TEST_DIR}/test

DEPS_DIR=`echo ${PACKAGE_SERVER} | awk -F '//' '{print $2}'`
rm -rf $DEPS_DIR

rm dependency.txt
