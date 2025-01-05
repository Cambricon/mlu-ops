"""
    implement of package `h5_creator`
"""

__all__ = (
    "gen_h5",
)

import sys
import os
import re
import tqdm
import logging
from multiprocessing import Pool

from analysis_suite.core.gtest_parser import case_info
from analysis_suite.core.h5_creator.network_info import NetworkInfo

def walk_dir(networks_dir):
    paths = []
    for root, dirs, files in os.walk(networks_dir, followlinks=True):
        for fn in files:
            if fn.endswith(".prototxt") or fn.endswith(".pb"):
                absolute_path = os.path.join(root, fn)
                paths.append(absolute_path)
    return paths

def gen_h5(cases_dir, cpu_count):
    if cases_dir:
        import h5py
        import pandas as pd

        # get file path
        paths = walk_dir(cases_dir)
        # get md5 for removing duplicates
        with Pool(cpu_count) as pool:
            nodes_info = \
                list(
                    tqdm.tqdm(
                        pool.imap(case_info.resolve_case, paths, chunksize=10),
                        total=len(paths),
                        ncols=80
                    )
                )

        # get count for every case-network
        unique_filenames = {}
        networks = set()
        for i in range(len(nodes_info)):
            absolute_path = paths[i]
            '''
            network_info is the absolute path of mluops_benchmark_config.json, which is samelevel as op path
            network_info is like
            xxx
                |-mluops_benchmark_config.json
                |-op_name
            '''
            network_info_dir = absolute_path.split('/')[0:-2]
            network_info_dir.append('mluops_benchmark_config.json')
            network_info = '/'.join(network_info_dir)
            if not nodes_info[i]:
                continue
            networks.add(network_info)
            md5 = nodes_info[i]['md5']
            cur_cnt = 1
            pttn = re.search(r"\.repeat-([0-9]+)\.", paths[i])
            if pttn:
                cur_cnt = int(pttn.group(1))
            if md5 not in unique_filenames:
                unique_filenames[md5] = [paths[i], {network_info: cur_cnt}]
            else:
                unique_filenames[md5][1][network_info] = \
                    unique_filenames[md5][1].get(network_info, 0) + cur_cnt

        # check whether pbName is unique
        path_count = {}
        for path in paths:
            pb_name = path.split('/')[-1]
            path_count[pb_name] = path_count.get(pb_name, 0) + 1
        duplicated_path = []
        for path in paths:
            pb_name = path.split('/')[-1]
            if path_count[pb_name] > 1:
                duplicated_path.append(path)
        if len(duplicated_path) > 0:
            raise Exception("PbNames have duplicated values")
        logging.info("network num: {}.".format(len(networks)))
        logging.info("test case num : {}.".format(len(unique_filenames)))

        # get network properties
        network_list = pd.DataFrame(dtype=object, columns=[])
        networks = list(networks)
        for i in range(len(networks)):
            network_json_file = networks[i]
            network_info = NetworkInfo()
            network_info_dict = network_info.analyse_json_config(network_json_file)
            network_info_dict["id"] = str(i)
            network_info_dict["whole_name"] = network_json_file
            network_info_df = pd.DataFrame([network_info_dict])
            network_list = pd.concat([network_list, network_info_df], ignore_index=True)
        network_list.set_index("whole_name", inplace=True)
        network_list['up_to_date'] = \
            -network_list.duplicated(
                [
                    "case_source",
                    "framework",
                    "is_complete_network",
                    "network_name",
                    "batchsize",
                    "precision_mode",
                    "card_num",
                    "mlu_platform",
                    "network_property"
                ],
                keep='last'
            )
        #print(network_list.T)
        # generat .h5 file
        network_info = NetworkInfo()
        network_info_params = network_info.validators.keys()
        h5_filename = "mluops_benchmark_raw_data.h5"
        with h5py.File(h5_filename, "w") as f:
            for md5, v in unique_filenames.items():
                pb_name = v[0].split('/')[-1]
                pb_group = f.create_group(pb_name)
                pb_group.create_dataset("file_path", data=v[0])
                for network, count in v[1].items():
                    network_group = pb_group.create_group(network_list.at[network,"id"])
                    for key in network_info_params:
                        network_group.create_dataset(key,data=network_list.at[network,key])
                    network_group.create_dataset(
                        "up_to_date",
                        data=network_list.at[network,"up_to_date"]
                    )
                    network_group.create_dataset("count", data=count)
        logging.info("h5 file is " + os.getcwd() + "/" + h5_filename)
