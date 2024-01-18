import sys
import json
import os
from distutils.version import LooseVersion

build_file = "build.property"
required_version = {}

# cntoolkit, cnnl
modules = ["cntoolkit", "cnnl"]
# NEUWARE_HOME = env_vars["NEUWARE_HOME"]
env_vars = dict(os.environ)


def get_build_requires(print_mode=1):
    global required_version
    with open(build_file) as build_property:
        data = json.load(build_property)
        required_version = data["build_requires"]
        for key in modules:
            if print_mode == 1:
                print(
                    "%s %s %s"
                    % (key, required_version[key][0], required_version[key][1])
                )
            required_version[key] = required_version[key][1].split("-")[0]


def check_cntoolkit():
    toolkit_ver_path = env_vars["NEUWARE_HOME"] + "/version.txt"
    if not os.path.exists(toolkit_ver_path):
        print("Not found toolkit")
        exit(2)

    # check cntoolkit
    with open(toolkit_ver_path) as tk_f:
        data = tk_f.readlines()
        for line in data:
            if "Neuware Version" in line:
                cur_tk_ver = line.strip("\n").split(" ")[-1]
                if LooseVersion(required_version["cntoolkit"]) > LooseVersion(
                    cur_tk_ver
                ):
                    print(
                        "The version of cntoolkit must be at least "
                        + required_version["cntoolkit"]
                        + ", but local version is "
                        + cur_tk_ver
                    )
                    exit(1)


def check_cnnl():
    cnnl_ver_pre = env_vars["NEUWARE_HOME"] + "/lib64/"
    if not os.path.exists(cnnl_ver_pre):
        print("Not found cnnl")
        exit(2)

    # check cnnl
    for filePath in os.listdir(cnnl_ver_pre):
        if "libcnnl.so." in filePath:
            tmp = filePath.split(".")
            if len(tmp) > 3:
                cur_cnnl_ver = filePath[11:]
                if LooseVersion(required_version["cnnl"]) > LooseVersion(cur_cnnl_ver):
                    print(
                        "The version of cnnl must be at least "
                        + required_version["cnnl"]
                        + ", but local version is "
                        + cur_cnnl_ver
                    )
                    exit(1)


def check_driver():
    sys_out = os.popen("cnmon version").readline()
    if len(sys_out) == 0:
        print("Warning: not found cnmon.")
        print("If compilation failed, please check driver version")
        return

    sys_out = sys_out.strip("\n").split(":")[-1]
    if LooseVersion(required_version["driver"]) > LooseVersion(sys_out):
        print(
            "The version of driver must be at least "
            + required_version["driver"]
            + ", but local version is "
            + sys_out
        )
        exit(1)


def check_protoc():
    sys_out = os.popen("protoc --version").readline()
    if len(sys_out) == 0:
        print("Not found protoc")
        exit(2)

    sys_out = sys_out.strip("\n").split(" ")[-1]
    if LooseVersion(required_version["protoc"]) < LooseVersion(sys_out):
        print(
            "The version of protoc must be at most "
            + required_version["protoc"]
            + ", but local version is "
            + sys_out
        )
        exit(1)


def check_libxml2():
    sys_out = os.popen("xml2-config --version").readline()
    if len(sys_out) == 0:
        print("Not found libxml2")
        exit(2)

    sys_out = sys_out.strip("\n")
    if LooseVersion(required_version["libxml2"]) > LooseVersion(sys_out):
        print(
            "The version of libxml2 must be at least "
            + required_version["libxml2"]
            + ", but local version is "
            + sys_out
        )
        exit(1)


def check_eigen3():
    if os.path.exists("/usr/local/include/eigen3/Eigen/src/Core/util/Macros.h"):
        h_file = open("/usr/local/include/eigen3/Eigen/src/Core/util/Macros.h")
    elif os.path.exists("/usr/include/eigen3/Eigen/src/Core/util/Macros.h"):
        h_file = open("/usr/include/eigen3/Eigen/src/Core/util/Macros.h")
    else:
        print("Not found eigen3")
        exit(2)

    line = h_file.readline()
    eigen_ver = ""
    while len(line) > 0:
        if "Eigen version and basic defaults" in line:
            line = h_file.readline()
            line = h_file.readline()
            eigen_ver = h_file.readline()[28:-1]
            eigen_ver += "." + h_file.readline()[28:-1]
            eigen_ver += "." + h_file.readline()[28:-1]
            break
        line = h_file.readline()

    if LooseVersion(required_version["eigen3"]) > LooseVersion(eigen_ver):
        print(
            "The version of eigen3 must be at least "
            + required_version["eigen3"]
            + ", but local version is "
            + eigen_ver
        )
        exit(1)


def check_build_requires():
    get_build_requires(0)
    check_cntoolkit()
    check_cnnl()
    check_driver()
    check_protoc()
    check_libxml2()
    check_eigen3()


argvs = sys.argv[1:]
if len(argvs) == 1:
    eval(argvs[0])()
elif len(argvs) == 2:
    eval(argvs[0])(argvs[1])
elif len(argvs) == 3:
    eval(argvs[0])(argvs[1], argvs[2])
else:
    exit(3)
