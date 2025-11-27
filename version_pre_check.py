import sys
import json
import os
import re
# version_check Module: packaging 1, distutils 0
version_check_module = 1

build_file = "build.property"
required_version = {}

# cntoolkit, cnnl
modules = ["cntoolkit", "cnnl"]
# NEUWARE_HOME = env_vars["NEUWARE_HOME"]
env_vars = dict(os.environ)

# version check status
version_status = {"not_found_version":2, "version_check_failed": 1,
                  "success": 0}

# version(str1) > version(str2)
def neVersion(str1, str2):
    global version_check_module

    if version_check_module == 1:
        try:
            from packaging import version
            return version.parse(str1) != version.parse(str2)
        except ImportError:
            print("packaging not exists, try import distutils")
            version_check_module = 0
        except Exception as e1:
            print(f"Warning: Failed to parse version: {e1}")

    if version_check_module == 0:
        try:
            from distutils.version import LooseVersion
            return LooseVersion(str1) != LooseVersion(str2)
        except ImportError:
            print("distutils not exists, version check failed")
            version_check_module = -1
        except Exception as e1:
            print(f"Warning: Failed to parse version: {e1}")

    return False


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

def check_zstd():
    sys_out = os.popen("zstd --version").readline().strip()

    if len(sys_out) == 0:
        print("Warning: Not found zstd (command not exists)")
        return version_status["not_found_version"]

    return version_status["success"]


def check_nlohmann_json():
    header_paths = [
        "/usr/include/nlohmann/json.hpp",
        "/usr/local/include/nlohmann/json.hpp",
        "/usr/include/x86_64-linux-gnu/nlohmann/json.hpp",
    ]

    for path in header_paths:
        if os.path.exists(path):
            return version_status["success"]

    print("Warning: nlohmann-json header file 'json.hpp' not found in known include paths.")
    return version_status["not_found_version"]


def check_cntoolkit():
    toolkit_ver_path = env_vars["NEUWARE_HOME"] + "/version.txt"
    if not os.path.exists(toolkit_ver_path):
        print("Warning: Not found toolkit version")
        return version_status["not_found_version"]

    # check cntoolkit
    with open(toolkit_ver_path) as tk_f:
        data = tk_f.readlines()
        for line in data:
            if "Neuware Version" in line:
                cur_tk_ver = line.strip("\n").split(" ")[-1]
                if neVersion(required_version["cntoolkit"], cur_tk_ver):
                    print(
                        "Warning: The version of cntoolkit is recommended to be "
                        + required_version["cntoolkit"]
                        + ", but local version is "
                        + cur_tk_ver
                    )
                    return version_status["version_check_failed"]

    return version_status["success"]


def check_cnnl():
    cnnl_ver_pre = env_vars["NEUWARE_HOME"] + "/lib64/"
    if not os.path.exists(cnnl_ver_pre):
        print("Warning: Not found cnnl version")
        return version_status["not_found_version"]

    # check cnnl
    for filePath in os.listdir(cnnl_ver_pre):
        if "libcnnl.so." in filePath:
            tmp = filePath.split(".")
            if len(tmp) > 3:
                cur_cnnl_ver = filePath[11:]
                if neVersion(required_version["cnnl"], cur_cnnl_ver):
                    print(
                        "Warning: The version of cnnl is recommended to be "
                        + required_version["cnnl"]
                        + ", but local version is "
                        + cur_cnnl_ver
                    )
                    return version_status["version_check_failed"]

    return version_status["success"]


def check_driver():
    sys_out = os.popen("cnmon version").readline()
    if len(sys_out) == 0:
        print("Warning: Not found driver version.")
        return version_status["not_found_version"]

    sys_out = sys_out.strip("\n").split(":")[-1]
    if neVersion(required_version["driver"], sys_out):
        print(
            "Warning: The version of driver is recommended to be "
            + required_version["driver"]
            + ", but local version is "
            + sys_out
        )
        return version_status["version_check_failed"]

    return version_status["success"]


def check_protoc():
    sys_out = os.popen("protoc --version").readline()
    if len(sys_out) == 0:
        print("Warning: Not found protoc version")
        return version_status["not_found_version"]

    sys_out = sys_out.strip("\n").split(" ")[-1]
    if neVersion(sys_out, required_version["protoc"]):
        print(
            "Warning: The version of protoc is recommended to be "
            + required_version["protoc"]
            + ", but local version is "
            + sys_out
        )
        return version_status["version_check_failed"]

    return version_status["success"]

def check_fmt():
    sys_out = os.popen("pkg-config --modversion fmt").readline()
    if len(sys_out) == 0:
        print("Warning: Not found fmt version")
        return version_status["not_found_version"]

    sys_out = sys_out.strip("\n")
    if neVersion(sys_out, required_version["fmt"]):
        print(
            "Warning: The version of fmt is recommended to be "
            + required_version["fmt"]
            + ", but local version is "
            + sys_out
        )
        return version_status["version_check_failed"]
    return version_status["success"]

def check_libxml2():
    sys_out = os.popen("xml2-config --version").readline()
    if len(sys_out) == 0:
        print("Warning: Not found libxml2 version")
        return version_status["not_found_version"]

    sys_out = sys_out.strip("\n")
    if neVersion(required_version["libxml2"], sys_out):
        print(
            "Warning: The version of libxml2 is recommended to be "
            + required_version["libxml2"]
            + ", but local version is "
            + sys_out
        )
        return version_status["version_check_failed"]

    return version_status["success"]


def check_eigen3():
    if os.path.exists("/usr/local/include/eigen3/Eigen/src/Core/util/Macros.h"):
        h_file = open("/usr/local/include/eigen3/Eigen/src/Core/util/Macros.h")
    elif os.path.exists("/usr/include/eigen3/Eigen/src/Core/util/Macros.h"):
        h_file = open("/usr/include/eigen3/Eigen/src/Core/util/Macros.h")
    else:
        print("Warning: Not found eigen3 version")
        return 2

    line = h_file.readline()
    eigen_ver = ""
    while len(line) > 0:
        if "EIGEN_WORLD_VERSION" in line:
            eigen_ver = line[28:-1]
        if "EIGEN_MAJOR_VERSION" in line:
            eigen_ver += "." + line[28:-1]
        if "EIGEN_MINOR_VERSION" in line:
            eigen_ver += "." + line[28:-1]
            break
        line = h_file.readline()

    if neVersion(required_version["eigen3"], eigen_ver):
        print(
            "Warning: The version of eigen3 is recommended to be "
            + required_version["eigen3"]
            + ", but local version is "
            + eigen_ver
        )
        return 1

    return 0


def check_build_requires():
    get_build_requires(0)
    if check_cntoolkit() != version_status["success"]:
        print("If compilation failed, please check cntoolkit version")
    if check_cnnl() != version_status["success"]:
        print("If compilation failed, please check cnnl version")
    if check_driver() != version_status["success"]:
        print("If compilation failed, please check driver version")
    if check_protoc() != version_status["success"]:
        print("If compilation failed, please check protoc version")
    if check_libxml2() != version_status["success"]:
        print("If compilation failed, please check libxml2 version")
    if check_eigen3() != version_status["success"]:
        print("If compilation failed, please check eigen3 version")
    if check_fmt() != version_status["success"]:
        print("If compilation failed, please check fmt version")
    if check_zstd() != version_status["success"]:
        print("If compilation failed, please check zstd version")
    if check_nlohmann_json() != version_status["success"]:
        print("If compilation failed, please check nlohmann-json version")

argvs = sys.argv[1:]
if len(argvs) == 1:
    eval(argvs[0])()
elif len(argvs) == 2:
    eval(argvs[0])(argvs[1])
elif len(argvs) == 3:
    eval(argvs[0])(argvs[1], argvs[2])
else:
    exit(3)
