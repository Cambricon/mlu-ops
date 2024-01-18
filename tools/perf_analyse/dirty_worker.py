# Copyright (C) [2024] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init
#!/usr/bin/env python3
#coding:utf-8

import pandas as pd
import re
import os
import tqdm
import subprocess
import logging
from multiprocessing import Pool
from pathlib import Path

from parser import Parser
from config import Config
from processor import Processor

class DirtyWorker:

    def __init__(self, args):
        self.args_ = args

    def walk_dir(networks_dir):
        paths = []
        for root, dirs, files in os.walk(networks_dir, followlinks=True):
            for fn in files:
                if fn.endswith(".prototxt") or fn.endswith(".pb"):
                    absolute_path = os.path.join(root, fn)
                    paths.append(absolute_path)

        return paths

    # the format of network_name is: framework_name_mode_batchsize(option)_other(option)_version_date
    def get_platforms_for_name(network_name,
                               framework_name=None,
                               additional=None):
        try:
            # see wiki 76995583
            info = network_name.split("_")
            if info[2] == "O0" or info[2] == "O1":
                platforms = "MLU290"
            elif info[1] == "cpm" and info[2] != "apex-O0":
                platforms = "MLU290"
            elif "mluOpbenchmak-290" in network_name:
                platforms = "MLU290"
            elif "mluOpbenchmak-all-cloud" in network_name:
                platforms = "MLU290 MLU370 MLU590"
            elif "tf32" in network_name:
                platforms = "MLU590"
            else:
                platforms = "MLU370 MLU590"

        except Exception as e:
            print(e)
            platforms = "MLU370-S4 MLU370-X4 MLU370-X8"

        return platforms

    def generator_h5(self, cases_dir, cpu_count):
        import h5py
        # get file path
        paths = DirtyWorker.walk_dir(cases_dir)
        # get md5 for removing duplicates
        with Pool(cpu_count) as pool:
            nodes_info = list(
                tqdm.tqdm(pool.imap(Parser.resolve_case, paths, chunksize=10),
                          total=len(paths),
                          ncols=80))
        # get count for every case-network
        unique_filenames = {}
        networks = set()
        for i in range(len(nodes_info)):
            absolute_path = paths[i]
            network, operator = absolute_path.split('/')[-3:-1]
            networks.add(network)
            md5 = nodes_info[i]['md5']
            if md5 not in unique_filenames:
                unique_filenames[md5] = [paths[i], {network: 1}]
            else:
                unique_filenames[md5][1][
                    network] = unique_filenames[md5][1].get(network, 0) + 1
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
        # get network properties
        network_list = pd.DataFrame(sorted(list(networks)),
                                    columns=['whole_name'])
        network_list['name'] = [
            '_'.join(i.split('_')[:-2]) for i in network_list['whole_name']
        ]
        network_list['up_to_date'] = -network_list.duplicated(['name'],
                                                              keep='last')
        mluop_network_list_dict = network_list.to_dict('list')
        # generator h5
        h5_filename = "mluop_benchmark_raw_data.h5"
        with h5py.File(h5_filename, "w") as f:
            for md5, v in unique_filenames.items():
                pb_name = v[0].split('/')[-1]
                pb_group = f.create_group(pb_name)
                pb_group.create_dataset("file_path", data=v[0])
                for network, count in v[1].items():
                    network_group = pb_group.create_group(network)
                    network_info = network.split("_")
                    network_group.create_dataset("framework",
                                                 data=network_info[0])
                    network_group.create_dataset("network_name",
                                                 data=network_info[1])
                    network_group.create_dataset("mode", data=network_info[2])
                    batchsize = float(re.findall(
                        r'bs(\d+)', network)[0]) if len(
                            re.findall(r'bs(\d+)', network)) > 0 else 0
                    network_group.create_dataset("batchsize", data=batchsize)
                    network_group.create_dataset("version",
                                                 data=network_info[-2])
                    additional = "_".join(network.split("_")[3:-2])
                    if "bs" in network:
                        bs_str = "bs"
                        if len(re.findall(r'bs(\d+)', network)) > 0:
                            bs_str = bs_str + re.findall(r'bs(\d+)',
                                                         network)[0]
                        additional = additional.replace(bs_str, "").strip("_")
                    network_group.create_dataset("additional", data=additional)
                    network_index = mluop_network_list_dict['whole_name'].index(
                        network)
                    network_group.create_dataset(
                        "up_to_date",
                        data=mluop_network_list_dict['up_to_date']
                        [network_index])
                    platforms = set([
                        i.split("-")[0]
                        for i in DirtyWorker.get_platforms_for_name(
                            network, network_info[0], additional).split(" ")
                    ])
                    platforms = ' '.join(platforms)
                    network_group.create_dataset("mlu_platform",
                                                 data=platforms)
                    network_group.create_dataset("count", data=count)
        print("h5 file is " + os.getcwd() + "/" + h5_filename)

    def get_code_size(self, so_path):
        # get file size of operator.a and libmluOp.so
        lib_path = os.path.abspath(so_path)
        cmd_args = ["readelf", "-e", lib_path]
        operator = []
        sizes = []

        cmd_ret = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE)
        so_size = re.findall(r"cn_fatbin(.*?) \[\d+\]", str(cmd_ret.stdout))[0]
        so_size = int(re.findall(r"\w+", so_size)[4], 16)
        operator.append('libmluOp.so')
        sizes.append(os.path.getsize(lib_path))
        operator.append('cn_fatbin')
        sizes.append(so_size)

        data = {'operator': operator, 'size': sizes}
        df = pd.DataFrame(data)
        return df

    def compare_code_size(self, code_size_bl, code_size_cmp):
        code_size_compare = pd.merge(code_size_bl,
                                     code_size_cmp,
                                     suffixes=Config.suffix,
                                     on=['operator'])
        code_size_compare['size提升(Bytes)'] = code_size_compare[
            'size' + Config.suffix[1]] - code_size_compare['size' +
                                                           Config.suffix[0]]
        code_size_compare['size提升比例(Bytes)'] = code_size_compare[
            'size提升(Bytes)'] / code_size_compare['size' + Config.suffix[1]]
        code_size_compare['size提升比例(Bytes)'] = code_size_compare[
            'size提升比例(Bytes)'].apply("{:.2%}".format)
        return code_size_compare

    def run(self):
        logging.info("DirtyWorker run start")
        # generate h5
        if self.args_.cases_dir:
            self.generator_h5(self.args_.cases_dir, self.args_.cpu_count)

        if self.args_.so_path:
            code_size = self.get_code_size(self.args_.so_path)
            Processor.dfs_to_excel([code_size], ['code_size'],
                                   "code_size.xlsx")
        if self.args_.so_path_compare:
            code_size_cmp = self.get_code_size(self.args_.so_path_compare)
            code_size_diff = self.compare_code_size(code_size, code_size_cmp)
            Processor.dfs_to_excel([code_size_diff], ['code_size_compare'],
                                   "code_size_compare.xlsx")

        logging.info("DirtyWorker run end")
        return 0
