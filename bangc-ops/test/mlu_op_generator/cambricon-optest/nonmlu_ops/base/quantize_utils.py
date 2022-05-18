from os import replace
import numpy as np


def caseDataNode(dst_data, src_data):
    if not dst_data.dtype_.exists():
        return
    dst_data.setData(src_data.data_.astype(dst_data.dtype_.getNumpyStr()))


def fillNanInf(np_value, dtype, contain_nan, contain_inf, rate=4):
    '''
    Fill nan or inf data in float type data. and the position is by random mode .
    '''
    if np_value.size == 0:
        return np_value
    special_value_list = []
    if contain_inf:
        special_value_list += [-np.inf, np.inf]
    if contain_nan:
        special_value_list += [np.nan]
    special_value_list = {
        "float16": np.float16,
        "float32": np.float32,
    }.get(dtype.getNumpyStr, np.float64)(special_value_list)

    id_inf_nan = np.random.choise(
        np_value.size, np_value.size // rate + 1, replace=False)
    np_value.flat[id_inf_nan] = np.random.choise(
        special_value_list, id_inf_nan.size)

    return np_value
