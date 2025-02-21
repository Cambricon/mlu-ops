'''
    external interface of package `perf_analyser`
        1. 接收上层传来的参数
        2. 调度 perf 和 tpi 包中的子模块
'''

__all__ = (
    "Scheduler"
)

import sys
import re
import os
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

from analysis_suite.utils import json_helper
from analysis_suite.cfg.config import Config, ColDef, PerfConfig

def get_frameworks_names(input_fw):
    frameworks = []
    frameworks = input_fw.split(",")
    all_frameworks = ("pytorch", "tf", "pt1.13")

    for i in frameworks:
        if i not in all_frameworks:
            raise Exception("The framework name entered is incorrect, incorrect name is {}".format(i))

    return frameworks

def get_version_numer(log_path, compare_path):
    log_filename = os.path.basename(log_path)
    compare_filename = os.path.basename(compare_path)

    version_compare = \
        re.findall(r'\_\d+\.\d+\.\d+.*', log_filename.replace(".xml", "")) + \
        re.findall(r'\_\d+\.\d+\.\d+.*', compare_filename.replace(".xml", ""))
    return version_compare

def get_future_lst_result_and_clear(no_return_future_lst):
    for future in no_return_future_lst:
        future.result()
    no_return_future_lst.clear()

class Scheduler:
    def __init__(self, *args):
        # TODO(init): try to be more elegant
        if len(args) > 0:
            self.args_ = args[0]
        else:
            raise ValueError("Unexpected input arguments")

        # log_path 用于本函数中获取 version_compare
        # log_path_base 用于本函数中获取
        #   xlsx_path, tpi_path, tpi_compare_path, tar_name, tar_compare_name, simple_tpi_path, tpi_compare_simple_path
        self.log_path = os.path.abspath(self.args_.log_path)
        log_path_base = self.args_.log_path.split("/")[-1].replace(".xml", "")

        # compare_path 用于本函数中获取 version_compare
        if self.args_.compare_path:
            self.args_.compare_path = os.path.abspath(self.args_.compare_path)

        if self.args_.xlsx_path is None:
            # not handle json
            self.args_.xlsx_path = os.path.abspath(log_path_base + ".xlsx")

        if self.args_.tpi:
            self.tpi_path = os.path.abspath(log_path_base + "_tpi.xlsx")
            self.tpi_compare_path = os.path.abspath(log_path_base + "_comparison_tpi.xlsx")
            self.tar_name = os.path.abspath(log_path_base + "_network_tpi.tar")
            self.tar_compare_name = os.path.abspath(log_path_base + "_network_compare_tpi.tar")

            if self.args_.simple_tpi:
                # frameworks_name 由 args_.frameworks 获取
                self.frameworks_name = get_frameworks_names(self.args_.frameworks)
                self.simple_tpi_path = self.tpi_path.replace("_tpi.xlsx", "_simple_tpi.xlsx")

                if self.args_.compare_path:
                    self.version_compare = get_version_numer(self.args_.log_path, self.args_.compare_path)
                self.tpi_compare_simple_path = self.tpi_compare_path.replace("_tpi.xlsx", "_simple_tpi.xlsx")
            else:
                if self.args_.compare_path:
                    self.version_compare = Config.suffix

        self.perf_config = PerfConfig()

    def tpi_pipeline(self, dfs):
        logging.info("{} start".format(sys._getframe().f_code.co_name))

        from analysis_suite.core.perf_analyser.tpi import get_tpi, compare_tpi, get_simple_tpi, compare_simple_tpi, tpi_utils

        float_to_percentage_cols = \
            [
                [
                    ColDef.operator_devices_time_sum_ratio_in_all_network_zh,
                    ColDef.operator_counts_ratio_in_all_networks_zh,
                    ColDef.io_bottleneck_ratio_zh,
                    ColDef.operator_devices_time_ratio_in_network_sum_zh,
                    ColDef.operator_counts_ratio_in_network_sum_zh
                ],
                [
                    ColDef.operator_devices_time_sum_ratio_in_all_network_zh,
                    ColDef.operator_counts_ratio_in_all_networks_zh,
                    ColDef.io_bottleneck_ratio_zh,
                    ColDef.operator_devices_time_ratio_in_network_sum_zh,
                    ColDef.operator_counts_ratio_in_network_sum_zh
                ],
                [
                    ColDef.io_bottleneck_ratio_zh
                ]
            ]
        dfs_len = len(dfs)
        if 2 == dfs_len:
            # use list to capture the future of async tasks
            no_return_future_lst = []
            # start process pool
            with ProcessPoolExecutor(max_workers=5) as pool:
                # TODO: more efficient scheduling
                # get tpi data
                tpi_future_A = pool.submit(get_tpi.get_tpi_data, dfs[0])
                tpi_future_B = pool.submit(get_tpi.get_tpi_data, dfs[1])
                try:
                    # get result from get_tpi.get_tpi_data
                    case_run, tpi_dfs, tpi_sheet_names = tpi_future_A.result()
                    case_run_bl, tpi_dfs_bl, _ = tpi_future_B.result()
                except Exception:
                    raise

                # dump table(framework_xxx and summary) to excel
                no_return_future_lst.append(pool.submit(tpi_utils.dump_tpi_excel,
                        tpi_dfs[0:3],
                        tpi_sheet_names[0:3],
                        self.tpi_path,
                        float_to_percentage_cols
                    ))
                # dump network tables to tar
                # TODO: dump to .db file
                dic_to_txt = dict(zip(tpi_sheet_names[3:], tpi_dfs[3:]))
                no_return_future_lst.append(pool.submit(tpi_utils.get_txt_excel_to_tar,
                        dic_to_txt, self.tar_name
                    ))
                # compare tpi data
                tpi_compare_future = pool.submit(compare_tpi.compare_tpi,
                        case_run,
                        case_run_bl,
                        tpi_dfs,
                        tpi_dfs_bl,
                        tpi_sheet_names,
                        self.tpi_compare_path,
                        self.version_compare,
                        self.tar_compare_name
                    )
                try:
                    # get result from tpi_utils.dump_tpi_excel & tpi_utils.get_txt_excel_to_tar
                    get_future_lst_result_and_clear(no_return_future_lst)
                    # get result from compare_tpi.compare_tpi
                    tpi_comp_dfs, tpi_comp_sheet_names = tpi_compare_future.result()
                except Exception:
                    raise

                # begin simple tpi
                if self.args_.simple_tpi:
                    # get simle tpi data and dump it to excel
                    dic = dict(zip(tpi_sheet_names, tpi_dfs))
                    no_return_future_lst.append(pool.submit(get_simple_tpi.dump_to_simple_tpi_network_excel,
                            dic,
                            self.simple_tpi_path,
                            self.frameworks_name
                        ))
                    # get compare simle tpi data and dump it to excel
                    dic = dict(zip(tpi_comp_sheet_names, tpi_comp_dfs))
                    no_return_future_lst.append(pool.submit(compare_simple_tpi.dump_to_simple_comparision_tpi_excel,
                            dic,
                            self.tpi_compare_simple_path,
                            self.frameworks_name,
                            self.version_compare
                        ))
                    try:
                        get_future_lst_result_and_clear(no_return_future_lst)
                    except Exception:
                        raise
        elif 1 == dfs_len:
            case_run, tpi_dfs, tpi_sheet_names = get_tpi.get_tpi_data(dfs[0])
            tpi_utils.dump_tpi_excel(
                    tpi_dfs[0:3],
                    tpi_sheet_names[0:3],
                    self.tpi_path, float_to_percentage_cols
                )
            dic_to_txt = dict(zip(tpi_sheet_names[3:], tpi_dfs[3:]))
            tpi_utils.get_txt_excel_to_tar(
                    dic_to_txt,
                    self.tar_name
                )
            if self.args_.simple_tpi:
                dic = dict(zip(tpi_sheet_names, tpi_dfs))
                get_simple_tpi.dump_to_simple_tpi_network_excel(
                        dic,
                        self.simple_tpi_path,
                        self.frameworks_name
                    )
        else:
            raise ValueError("The XML parsing result is incorrect.")

        logging.info("{} end".format(sys._getframe().f_code.co_name))

    def perf_pipeline(self, dfs):
        logging.info("{} start".format(sys._getframe().f_code.co_name))

        from analysis_suite.core.perf_analyser.perf import get_data, compare_data, perf_utils
        from analysis_suite.core.perf_analyser.perf import summary_to_database

        compare_xlsx_path = self.args_.xlsx_path.replace(".xlsx", "_comparison.xlsx")
        pic_path = self.args_.xlsx_path.replace(".xlsx", ".png")

        json_ops = None
        if self.args_.json_file:
            json_ops = set(json_helper.read_json(self.args_.json_file))

        dfs_len = len(dfs)
        if 2 == dfs_len:
            # use list to capture the future of async tasks
            no_return_future_lst = []
            # start process pool
            with ProcessPoolExecutor(max_workers=5) as pool:
                process_future_A = pool.submit(get_data.process,
                        dfs[0],
                        perf_config     = self.perf_config,
                        is_release      = self.args_.is_release,
                        use_db          = self.args_.use_db,
                        json_ops        = json_ops,
                        is_pro          = self.args_.is_pro,
                        need_case_info  = self.args_.need_case_info
                    )
                process_future_B = pool.submit(get_data.process,
                        dfs[1],
                        perf_config     = self.perf_config,
                        is_release      = self.args_.is_release,
                        use_db          = self.args_.use_db,
                        json_ops        = json_ops,
                        is_pro          = self.args_.is_pro,
                        need_case_info  = self.args_.need_case_info
                    )
                # get result from process_future
                try:
                    summary, sheet_names_new = process_future_A.result()
                    summary_bl, sheet_names_old = process_future_B.result()
                except Exception:
                    raise

                process_compare_future = pool.submit(compare_data.compare_process,
                        summary,
                        sheet_names_new,
                        summary_bl,
                        sheet_names_old,
                        self.args_.need_case_info
                    )
                no_return_future_lst.append(pool.submit(perf_utils.dump_perf_result_to_excel,
                        summary,
                        sheet_names_new,
                        self.args_.xlsx_path,
                        self.args_.deduplication
                    ))
                try:
                    get_future_lst_result_and_clear(no_return_future_lst)
                    # get result from process_compare_future
                    compare_dfs, sheet_names = process_compare_future.result()
                except Exception:
                    raise

                no_return_future_lst.append(pool.submit(perf_utils.dump_compare_result_to_excel,
                        compare_dfs,
                        sheet_names,
                        compare_xlsx_path
                    ))
                """
                if self.args_.case_run:
                    no_return_future_lst.append(pool.submit(summary_to_database.update_database,
                            summary,
                            sheet_names_new,
                            self.args_.truncate_case_run,
                            self.perf_config
                        ))
                """
                if self.args_.generate_pic:
                    no_return_future_lst.append(pool.submit(perf_utils.generate_pic,
                            compare_dfs[0],
                            pic_path
                        ))
                try:
                    get_future_lst_result_and_clear(no_return_future_lst)
                except Exception:
                    raise
        elif 1 == dfs_len:
            summary, sheet_names_new = get_data.process(
                dfs[0],
                perf_config     = self.perf_config,
                is_release      = self.args_.is_release,
                use_db          = self.args_.use_db,
                json_ops        = json_ops,
                is_pro          = self.args_.is_pro,
                need_case_info  = self.args_.need_case_info
            )

            perf_utils.dump_perf_result_to_excel(
                summary,
                sheet_names_new,
                self.args_.xlsx_path,
                self.args_.deduplication
            )

            """
            if self.args_.case_run:
                summary_to_database.update_database(
                    summary,
                    sheet_names_new,
                    self.args_.truncate_case_run,
                    self.perf_config
                )
            """
        else:
            raise ValueError("The XML parsing result is incorrect.")
        logging.info("{} end".format(sys._getframe().f_code.co_name))

    def analyse_dataframes(self, dfs: List[Optional[pd.DataFrame]]):
        logging.info("{} start".format(sys._getframe().f_code.co_name))

        # use list to capture the future of async tasks
        no_return_future_lst = []
        # start process pool
        with ProcessPoolExecutor(max_workers=2) as pool:
            if self.args_.tpi:
                no_return_future_lst.append(pool.submit(self.tpi_pipeline, dfs))
            no_return_future_lst.append(pool.submit(self.perf_pipeline, dfs))

            try:
                get_future_lst_result_and_clear(no_return_future_lst)
            except Exception as e:
                raise e

        logging.info("{} end".format(sys._getframe().f_code.co_name))
