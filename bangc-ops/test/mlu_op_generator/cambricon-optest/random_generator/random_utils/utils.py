import os
import functools
import itertools
import random


def getRandomValues(random_values):
    assert("mode" in random_values)
    assert("params" in random_values)
    mode = random_values["mode"]
    params = random_values["params"]
    val_list = []
    if mode == 0:
        val_list = params
    elif mode == 1:
        val_list = list(range(params[0], params[1], params[2]))
    else:
        val_list.append(random.randint(params[0], params[1]))
    return val_list


def getTotalRandomShape(random_params):
    assert("dim_range" in random_params)
    assert("value_range" in random_params)
    assert("size" in random_params)

    dim_range = random_params["dim_range"]
    value_range = random_params["value_range"]
    random_num = random_params["size"]
    assert(random_num > 0)

    shape_list = []
    for i in range(random_num):
        shape = []
        len = random.randint(dim_range[0], dim_range[1])
        for i in range(len):
            shape.append(random.randint(value_range[0], value_range[1]))
        shape_list.append(tuple(shape))
    return shape_list


def getPartRandomShape(random_params):
    assert("part_shape" in random_params)
    assert("random_values" in random_params)
    if (random_params["part_shape"].count(0) != len(random_params["random_values"])):
        raise Exception("part_shape and random_values mismatch")
    else:
        data = []
        j = 0
        part_shape = random_params["part_shape"]
        random_values = random_params["random_values"]
        for i in range(len(part_shape)):
            dim = []
            if part_shape[i] != 0:
                dim.append(part_shape[i])
            else:
                dim = getRandomValues(random_values[j])
                j += 1
            data.append(dim)
    return data


def getRandomParamList(random_info):
    param_list = []
    if "total_random" in random_info:
        param_list = getTotalRandomShape(random_info["total_random"])
    elif "part_random" in random_info:
        combination_dims = getPartRandomShape(random_info["part_random"])
        assert(len(combination_dims) != 0)
        param_list = list(itertools.product(*combination_dims))
    elif "shape" in random_info:
        param_list.append(tuple(random_info["shape"]))
    elif isinstance(random_info, list):
        param_list = random_info
    else:
        pass
    return param_list


def mkdir_folder(save_path, op_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    folder_name = save_path + "/" + op_name
    if os.path.exists(folder_name):
        pass
    else:
        os.makedirs(folder_name)
    return folder_name


def getFileFromDirBase(rootdir, filter):
    file_list = []

    def bianli(rootdir):
        for root, dirs, files in os.walk(rootdir):
            for file in files:
                if filter(file):
                    file_list.append(os.path.join(root, file))
            for dir in dirs:
                bianli(dir)
    bianli(rootdir)
    return file_list


def endsWithJson(filename):
    return filename.endswith(".json")


def filterTrue(filename):
    return True


getJsonFileFromDir = functools.partial(getFileFromDirBase, filter=endsWithJson)
getFileFromDir = functools.partial(getFileFromDirBase, filter=filterTrue)
