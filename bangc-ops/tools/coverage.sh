#!/bin/bash
set -e

readonly RED='\033[1;31m'
readonly NC='\033[0m'
readonly CARD_NUM_DEFAULT=8

DEVICE_ID=0
test_cmd_=''
temp_dir_='.'
extra_test_dir_=''
lib_path_="../lib/libmluops.so"

function usage () {
    echo
    echo "Coverage is a code coverage testing tool of bangC for mlu-ops."
    echo "Usage: ./coverage.sh COMMAND [options]"
    echo "COMMAND: gtest command."
    echo "[options]: "
    echo "     -t [optional]: temporary dir to store raw files. optional, default is current dir."
    echo "     -e [optional]: extra gtest dir, optional, meaning that you can run other gtest."
    echo "     -l [optional]: path of libmluops.so, optional, default is ../lib/libmluops.so."
    echo "**********************example*********************"
    echo "./coverage.sh \"./mluops_gtest --gtest_filter=*add*\""
    echo "**************************************************"
}

# Handle options
function parse_args () {
    if [ $# != 0 ]; then
        test_cmd_=$1
        shift
        # options
        while [ $# != 0 ]; do
            case "$1" in
                -e | --extra)
                    extra_test_dir_=$2
                    echo "specify extra_test_dir_ = ${extra_test_dir_}."
                    shift 2
                    ;;
                -h | --help)
                    usage
                    exit 0
                    ;;
                -l | --library)
                    lib_path_=$2
                    echo "specify lib_path_ = ${lib_path_}."
                    shift 2
                    ;;
                -t | --temporary)
                    temp_dir_=$2
                    echo "specify temp_dir = ${temp_dir_}."
                    shift 2
                    ;;
                *)
                    printf "${RED} ERROR: Unknown options ${1}, use -h or --help\n${NC}"
                    usage
                    exit -1
                    ;;
            esac
        done
    fi
}

function run_extra_test () {
    printf "============= Coverage: run extra test ==============\n"
    export LLVM_PROFILE_FILE=${temp_dir_}/output/host_extra.profraw
    readonly new_case="--cases_dir=${extra_test_dir_}"
    for arg in ${test_cmd_}; do
        if [[ ${arg} == "--cases_dir="* || ${arg} == "--case_path="* || ${arg} == "--cases_list="* ]]; then
            readonly old_case=${arg}
        fi
    done
    if [[ -v old_case ]]; then
        test_cmd_=${test_cmd_/${old_case}/${new_case}}
    else
        test_cmd_="${test_cmd_} ${new_case}"
    fi
    ${test_cmd_}
}

function process () {
    # run gtest
    readonly mluops_dir=$(dirname ${lib_path_})
    export LD_LIBRARY_PATH="${mluops_dir}":$LD_LIBRARY_PATH
    export CNRT_DUMP_PGO=1
    mkdir -p ${temp_dir_}
    export CNRT_PGO_OUTPUT_DIR=${temp_dir_}/output
    export LLVM_PROFILE_FILE=${temp_dir_}/output/host.profraw
    ${test_cmd_}
    if [[ ! -z ${extra_test_dir_} ]]; then
        run_extra_test
    fi
    # merge to a single indexed profile data file
    mkdir -p ${temp_dir_}/profdata
    llvm-profdata merge ${temp_dir_}/output/host*.profraw -o ${temp_dir_}/profdata/host.profdata

    # export coverage data of the binaries
    mkdir -p ${temp_dir_}/info
    llvm-cov export ${lib_path_}  -instr-profile=${temp_dir_}/profdata/host.profdata -host-only -format=lcov \
      > ${temp_dir_}/info/host_info

    # device info
    find ${temp_dir_}/output/ -maxdepth 1 -type f -name "CNPGO*"  > ${temp_dir_}/profdata/profile_list.txt
    llvm-profdata merge  -f ${temp_dir_}/profdata/profile_list.txt -o ${temp_dir_}/profdata/device.profdata
    local arch=$(awk -F '_' '{print $(NF-3)}' ${temp_dir_}/profdata/profile_list.txt | sort | uniq)
    local kernels=$(awk -F '_' '{print $(NF-2)}' ${temp_dir_}/profdata/profile_list.txt | sort | uniq)
    for k in ${kernels}
    do
        llvm-cov export ${lib_path_} -instr-profile=${temp_dir_}/profdata/device.profdata -kernel-hash-id=${k} \
          -bang-mlu-arch=${arch} -format=lcov > "${temp_dir_}/info/device_info_${k}"
    done

    # generate report
    genhtml ${temp_dir_}/info/* -o result

    for arg in ${test_cmd_}; do
        if [[ ${arg} == "--gtest_filter="* ]]; then
            op_dir_name=${arg##*=}
        fi
    done
    # show the coverage test report
    if [[ ! -z ${op_dir_name} ]]; then
        html2text ${temp_dir_}/result/kernels/${op_dir_name}/index.html
    else
        html2text ${temp_dir_}/result/index.html
    fi
}

function main () {
    if [[ -v NEUWARE_HOME ]]; then
        echo " NEUWARE_HOME:  "${NEUWARE_HOME}
        export PATH="${NEUWARE_HOME}/bin":$PATH
        export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64":$LD_LIBRARY_PATH
        export MLUOP_GTEST_FILL_RAM=OFF
    else
        printf "${RED} ERROR: please export NEUWARE_HOME variable first!\n${NC}"
        exit 1
    fi
    if [[ "$#" -lt 1 ]]; then
        printf "${RED} ERROR: arguments too few !\n${NC}"
        usage && exit 1
    fi

    if [[ "$(which genhtml)" == "" ]]; then
        echo "--Coverage test failed, please install genhtml"
        exit 0
    fi

    if [[ "$(which html2text)" == "" ]]; then
        echo "--Coverage test failed, please install html2text."
        exit 0
    fi

    parse_args "$@"
    export MLU_VISIBLE_DEVICES=${DEVICE_ID}
    process
}

main "$@"
