import os
import functools
import subprocess
import numpy as np
import logging
import struct


def available_nvidia_smi():
    nvidia_smi_ret = subprocess.getoutput("nvidia-smi")
    if "failed to initialize" in nvidia_smi_ret.lower():
        logging.warn(nvidia_smi_ret)
        logging.warn(subprocess.getoutput("dmesg | grep NVRM | tail -n4"))
        logging.warn(subprocess.getoutput("cat /proc/driver/nvidia/version"))
        logging.warn(
            "`nvidia-smi` not work, skip set env CUDA_VIAIBLE_DEVICES automatically.")
        return False
    if "not found" in nvidia_smi_ret:
        logging.warn(
            "no `nvidia-smi`, may need to install nvidia-utils-xxx libnvidia-compute-xxx, or you are using CPU not GPU")
        return False
    return True


def available_GPU():
    nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU | wc -l"))
    total_GPU_str = subprocess.getoutput(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
    total_GPU = total_GPU_str.split("\n")
    total_GPU = np.array([int(device_i) for device_i in total_GPU])
    avail_GPU_str = subprocess.getoutput(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
    avail_GPU = avail_GPU_str.split("\n")
    avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
    avail_GPU = avail_GPU / total_GPU
    return np.argmax(avail_GPU)


def getFileFromDirBase(rootdir, filter):
    file_list = []

    def traverse(rootdir):
        for root, dirs, files in os.walk(rootdir):
            for file in files:
                if filter(file):
                    file_list.append(os.path.join(root, file))
            for dir in dirs:
                traverse(dir)
    traverse(rootdir)
    return file_list


def endsWithJson(filename):
    return filename.endswith(".json")


def endsWithPrototxt(filenname):
    return filenname.endswith(".prototxt")


def filterTrue(filename):
    return True


getJsonFileFromDir = functools.partial(getFileFromDirBase, filter=endsWithJson)
getPrototxtFileFromDir = functools.partial(
    getFileFromDirBase, filter=endsWithPrototxt)
getFileFromDir = functools.partial(getFileFromDirBase, filter=filterTrue)


def str2dtype(x, dtype):
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", int(x, 16)))[0]
    elif dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", int(x, 16)))[0]
    elif dtype == "bool":
        return struct.unpack("<?", struct.pack("<B", int(x, 16)))[0]
    elif dtype == "uint8":
        return struct.unpack("<B", struct.pack("<B", int(x, 16)))[0]
    elif dtype == "int8":
        return struct.unpack("<b", struct.pack("<b", int(x, 16)))[0]
    elif dtype == "uint16":
        return struct.unpack("<H", struct.pack("<H", int(x, 16)))[0]
    elif dtype == "int16":
        return struct.unpack("<h", struct.pack("<h", int(x, 16)))[0]
    elif dtype == "uint32":
        return struct.unpack("<I", struct.pack("<I", int(x, 16)))[0]
    elif dtype == "int32":
        return struct.unpack("<i", struct.pack("<i", int(x, 16)))[0]
    elif dtype == "uint64":
        return struct.unpack("<Q", struct.pack("<Q", int(x, 16)))[0]
    elif dtype == "int64":
        return struct.unpack("<q", struct.pack("<q", int(x, 16)))[0]
    else:
        raise Exception("### str2dtype: failed. Please check datatype.")
