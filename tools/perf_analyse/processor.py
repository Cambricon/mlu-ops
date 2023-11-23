
import re
import os
import sys
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, types

from config import Config, PerfConfig

class Processor:

    def __init__(self, *args):
        # TODO(init): try to be more elegant
        if len(args) > 0:
            self.args_ = args[0]
            if self.args_.log_path:
                self.args_.log_path = os.path.abspath(self.args_.log_path)
            if self.args_.compare_path:
                self.args_.compare_path = os.path.abspath(
                    self.args_.compare_path)
            if self.args_.xlsx_path == None:
                # not handle json
                if self.args_.log_path:
                    self.args_.xlsx_path = os.path.abspath(
                        self.args_.log_path.split("/")[-1].replace(".xml", "") +
                        ".xlsx")
            if self.args_.tpi:
                self.tpi_path = os.path.abspath(
                    self.args_.log_path.split("/")[-1].replace(".xml", "") +
                    "_tpi.xlsx")
                self.tpi_compare_path = os.path.abspath(
                    self.args_.log_path.split("/")[-1].replace(".xml", "") +
                    "_comparison_tpi.xlsx")

            if self.args_.simple_tpi:
                self.frameworks_name = self.get_frameworks_names(
                    self.args_.frameworks)
                self.simple_tpi_path = self.tpi_path.replace(
                    "_tpi.xlsx", "_simple_tpi.xlsx")
                self.version_compare = self.get_version_numer(
                    self.args_.log_path, self.args_.compare_path)
                self.tpi_compare_simple_path = self.tpi_compare_path.replace(
                    "_tpi.xlsx", "_simple_tpi.xlsx")
            else:
                self.version_compare = ['_new', '_baseline']

        if len(args) > 1:
            self.db_ = args[1]
        self.perf_config = PerfConfig()

    def get_max_length(series, col_name):
        """
        get max length of series, used to auto set width
        """
        # avoid nan
        series = series.fillna('-')
        str_list = series.to_list()
        len_list = []
        for ele in str_list + [col_name]:
            # change str to list
            ele_split = list(ele)
            length = 0
            for c in ele_split:
                if ord(c) <= 256:
                    length += 1
                else:
                    # chinese character will be fat
                    length += 2
            len_list.append(length + 2)

        max_length = max(len_list)
        max_length = min(max_length, 50)
        max_length = max(max_length, 12)
        return max_length

    # helper function for beautifing excel
    # df.columns can not be repeated
    def auto_width(df, writer, sheet_name, cols_list):
        for i in range(0, len(cols_list)):
            col = cols_list[i]
            letter = chr(i + 65)
            max_len = Processor.get_max_length(df[col].astype(str), col)
            wb = writer.book
            ws = writer.sheets[sheet_name]
            fmt = wb.add_format({'align': 'left'})
            ws.set_column(i, i, max_len, fmt)

    def dfs_to_excel(dfs, sheet_names, xlsx_path):
        with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
            for i in range(0, len(dfs)):
                dfs[i].to_excel(writer, sheet_name=sheet_names[i], index=False)
                Processor.auto_width(dfs[i], writer, sheet_names[i],
                                     dfs[i].columns)

    def dfs_to_excel_deduplication(dfs, sheet_names, xlsx_path):
        with pd.ExcelWriter(xlsx_path) as writer:
            for i in range(0, len(dfs)):
                if 'md5' in dfs[i].columns:
                    dfs[i] = dfs[i][-dfs[i].duplicated(subset='md5')]
                    dfs[i].to_excel(writer, sheet_name=sheet_names[i], index=False)
                    Processor.auto_width(dfs[i], writer, sheet_names[i],
                                        dfs[i].columns)
                else:
                    dfs[i].to_excel(writer, sheet_name=sheet_names[i], index=False)
                    Processor.auto_width(dfs[i], writer, sheet_names[i],
                                        dfs[i].columns)

    def get_network_summary(self, df):
        logging.info('get_network_summary start')
        # do not handle cases not in benchmark
        if not df.loc[0, 'file_path'].startswith('/SOFT_TRAIN/benchmark'):
            logging.info("case {} not in /SOFT_TRAIN/benchmark".format(df.loc[0, 'file_path']))
            return pd.DataFrame(), None

        mluop_case_run = self.append_network_info(df)
        mluop_case_run['mlu_hardware_time'] = mluop_case_run[
            'mlu_hardware_time'] * mluop_case_run['count']

        network_summary = self.get_summary(mluop_case_run, 'network_id')
        # append mlu_platforms
        network_summary['mlu_platform'] = df.loc[0, 'mlu_platform']
        network_summary['mluop_version'] = df.loc[0, 'mluop_version']
        network_summary['date'] = df.loc[0, 'date']
        network_summary['test_time'] = df.loc[0, 'test_time']
        network_summary['commit_id'] = df.loc[0, 'commit_id']
        network_summary['mluop_branch'] = df.loc[0, 'mluop_branch']
        network_summary['is_release'] = self.args_.is_release
        # add network info
        network_summary = pd.merge(
            network_summary,
            self.db_.network_list[Config.network_info_keys],
            on=['network_id'])

        # add network_summary in database
        # TODO(network_summary): what aboud multi version
        if df.loc[0, 'mlu_platform'] == 'MLU590-M9':
            network_summary = pd.merge(network_summary,
                                       self.db_.network_summary[self.db_.network_summary['mlu_platform']=='MLU590-M9U'].drop(columns=['mlu_platform']),
                                       how='left',
                                       on=['network_id'],
                                       suffixes=["", "_database"])
        else:
            network_summary = pd.merge(network_summary,
                                       self.db_.network_summary,
                                       how='left',
                                       on=['network_id', 'mlu_platform'],
                                       suffixes=["", "_database"])

        logging.info('get_network_summary end')
        return network_summary, mluop_case_run[[
            'protoName', 'network_id', 'count'
        ]]

    def get_operator_summary(self, df):
        # after ignore df may not have data
        mlu_platform = df.loc[0, 'mlu_platform']
        mluop_version = df.loc[0, 'mluop_version']
        date = df.loc[0, 'date']
        test_time = df.loc[0, 'test_time']
        commit_id = df.loc[0, 'commit_id']
        mluop_branch = df.loc[0, 'mluop_branch']
        # ignore small case
        df = df[df['mlu_hardware_time'] > self.perf_config.attrs['ignore_case']
                ['mlu_hardware_time']]

        if not df.empty:
            # should append resources mlu_platforms mluop_version date ? repeat_key
            operator_summary = self.get_summary(df, 'operator')
            operator_summary['mlu_platform'] = mlu_platform
            operator_summary['mluop_version'] = mluop_version
            operator_summary['date'] = date
            operator_summary['commit_id'] = commit_id
            operator_summary['mluop_branch'] = mluop_branch
            operator_summary['is_release'] = self.args_.is_release
            operator_summary['test_time'] = test_time
            return operator_summary
        else:
            logging.warn("The test cases are all small cases(mlu_hardware_time < 30).")
            return pd.DataFrame()

    # summary.columns = group_key + case_number + mlu_io_efficiency_mean
    #                 + mlu_compute_efficiency_mean + mlu_hardware_time_sum
    #                 + good_rate + qualified_rate + unqualified_rate
    def get_summary(self, df, group_key):
        group = df.groupby(group_key)
        # assure column order
        summary = pd.DataFrame()
        summary['case_number'] = group.apply(lambda x: x['file_path'].count())

        # compute mean efficiency by the bottleneck side
        # add bottleneck efficiency column?
        summary['mlu_io_efficiency_mean'] = group.apply(
            lambda x: x[x['is_io_bound']]['mlu_io_efficiency'].mean())
        summary['mlu_compute_efficiency_mean'] = group.apply(
            lambda x: x[~x['is_io_bound']]['mlu_compute_efficiency'].mean())
        summary['mlu_hardware_time_sum'] = group.apply(
            lambda x: x['mlu_hardware_time'].sum())
        # what about invalid
        for k in self.perf_config.attrs['criterion'].keys():
            summary[k +
                    '_rate'] = group.apply(lambda x: (x['status'] == k).sum())
            summary[k +
                    '_rate'] = summary[k + '_rate'] / summary['case_number']
        # efficiency can be nan because of all case is io bound
        summary.fillna(0, inplace=True)
        summary.reset_index(inplace=True)

        return summary

    def append_network_info(self, df):
        # merge database info
        mluop_case_run = pd.merge(df,
                                 self.db_.case_list[['case_id', 'protoName']],
                                 on=['protoName'])
        mluop_case_run = pd.merge(mluop_case_run,
                                 self.db_.case_in_network,
                                 on=['case_id'])
        mluop_case_run = pd.merge(mluop_case_run,
                                 self.db_.network_list,
                                 on=['network_id'])

        # only handle mlu_platform in xml
        platform_in_db = Config.platform_map[df.loc[0, 'mlu_platform']]
        mluop_case_run = mluop_case_run[mluop_case_run[platform_in_db] == 1]
        # drop MLU270_X5K columns
        drop_columns = set(self.db_.network_list.columns) - set(
            Config.network_info_keys)
        mluop_case_run.drop(columns=drop_columns, inplace=True)
        return mluop_case_run

    def compare(self, df_new, df_bl, on, columns_, info_columns,
                promotion_columns):
        # info_columns can not be hashable
        columns = columns_ + [on]
        df_compare = pd.merge(df_new[columns],
                              df_bl[columns],
                              on=on,
                              suffixes=Config.suffix)
        # for some circumstance, such as small cases ignorance or original cases diffrence,
        # df_new and df_bl operators number will be different; so we preserve index first.
        index_arr = []
        for index in range(len(df_new[on].to_list())):
            item = df_new[on].to_list()[index]
            if item in df_bl[on].to_list():
                index_arr.append(index)
        # update df_compare info column using preserved index
        for i in info_columns:
            df_compare[i] = [df_new[i].to_list()[index] for index in index_arr]
        # compute promotion, can be new - baseline or baseline - new
        for column in promotion_columns:
            df_compare[column + "_promotion"] = df_compare[
                column + Config.suffix[1]] - df_compare[column +
                                                        Config.suffix[0]]
            df_compare[column + "_promotion_ratio"] = df_compare[
                column + "_promotion"] / df_compare[column + Config.suffix[1]]
        return df_compare

    def compare_cases(self, df_new, df_bl):
        columns = [
            'mlu_hardware_time', 'mlu_io_efficiency', 'mlu_compute_efficiency',
            'mlu_interface_time', 'mlu_workspace_size', 'mlu_kernel_names'
        ]
        info_columns = [
            'operator', 'file_path', 'input', 'output', 'params', 'is_io_bound'
        ]
        promotion_columns = ['mlu_hardware_time']
        cases_compare = self.compare(df_new, df_bl, 'protoName', columns,
                                     info_columns, promotion_columns)

        # assure column order
        columns_ = [
            'operator', 'mlu_hardware_time_promotion',
            'mlu_hardware_time_promotion_ratio'
        ]
        columns_ = columns_ + [m + n for m in columns for n in Config.suffix]
        columns_ = columns_ + [
            'file_path', 'input', 'output', 'params', 'is_io_bound'
        ]
        return cases_compare[columns_].sort_values(
            by=['mlu_hardware_time_promotion'], ascending=False)

    def compare_operator_summary(self, df_new, df_bl):
        info_columns = ['operator']
        promotion_columns = ['mlu_hardware_time_sum']
        operator_summary_compare = self.compare(df_new, df_bl, 'operator',
                                                Config.summary_columns,
                                                info_columns,
                                                promotion_columns)
        # assure column order
        columns_ = [
            'operator', 'mlu_hardware_time_sum_promotion',
            'mlu_hardware_time_sum_promotion_ratio'
        ]
        columns_ = columns_ + [
            m + n for m in Config.summary_columns for n in Config.suffix
        ]
        return operator_summary_compare[columns_].sort_values(
            by=['mlu_hardware_time_sum_promotion'], ascending=False)

    def compare_network_summary(self, df_new, df_bl):
        promotion_columns = ['mlu_hardware_time_sum']
        network_summary_compare = self.compare(df_new, df_bl, 'network_id',
                                               Config.summary_columns,
                                               Config.network_info_keys,
                                               promotion_columns)
        network_summary_compare['network_time'] = df_new["mlu_hardware_time_sum_database"]
        network_summary_compare['ops_promotion_to_network'] = network_summary_compare['mlu_hardware_time_sum_promotion'] / network_summary_compare['network_time'] if network_summary_compare['network_time'] is not None else None
        # assure column order
        columns_ = [
            'network_id', 'mlu_hardware_time_sum_promotion',
            'mlu_hardware_time_sum_promotion_ratio', 'ops_promotion_to_network'
        ]
        columns_ = columns_ + [
            m + n for m in Config.summary_columns for n in Config.suffix
        ]
        columns_ = columns_ + [
            'network_name','network_time',  'framework', 'mode', 'batchsize',
            'network_additional_information', 'version'
        ]
        return network_summary_compare[columns_].sort_values(
            by=['mlu_hardware_time_sum_promotion'], ascending=False)

    def compare_process(self, dfs_new, sheet_names_new, dfs_bl, sheet_names_bl):
        dfs = []
        sheet_names = []

        dfs.append(self.compare_cases(dfs_new[0], dfs_bl[0]))
        sheet_names.append("cases_compare")

        tmp_idx = 1

        if 'operator_summary' in sheet_names_new and 'operator_summary' in sheet_names_bl:
            dfs.append(self.compare_operator_summary(dfs_new[tmp_idx], dfs_bl[tmp_idx]))
            sheet_names.append("operator_summary_compare")
            tmp_idx += 1
        else:
            logging.warn("baseline or new's operator summary is empty, ignore to comapre!")

        if 'network_summary' in sheet_names_new and 'network_summary' in sheet_names_bl:
            dfs.append(self.compare_network_summary(dfs_new[tmp_idx], dfs_bl[tmp_idx]))
            sheet_names.append("network_summary_compare")
            tmp_idx += 1
        else:
            logging.warn("baseline or new's network summary is empty, ignore to comapre!")

        return dfs, sheet_names

    def generate_pic(self, df_compare_info, pic_path):
        # the following two sentences support Chinese SimHei
        # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        # matplotlib.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 12), dpi=1000)
        #show two data
        plt.subplot(311)
        plt.title("mlu_hardware_time")
        plt.plot(df_compare_info['mlu_hardware_time' +
                                 Config.suffix[0]].values,
                 color='green',
                 label='mlu_hardware_time' + Config.suffix[0] + '(us)')
        plt.plot(df_compare_info['mlu_hardware_time' +
                                 Config.suffix[1]].values,
                 color='red',
                 label='mlu_hardware_time' + Config.suffix[1] + '(us)')
        plt.legend()

        #show data1-data2

        plt.subplot(312)
        plt.title('mlu_hardware_time reduction')
        plt.bar(range(0, len(df_compare_info)),
                df_compare_info['mlu_hardware_time_promotion'].values,
                label='mlu_hardware_time_promotion (us)',
                width=1)
        plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
        plt.legend()

        #show data1/data2
        plt.subplot(313)
        plt.title('mlu_hardware_time reduction percentage')
        plt.bar(range(0, len(df_compare_info)),
                (df_compare_info['mlu_hardware_time_promotion_ratio'] *
                 100.0).values,
                label='mlu_hardware_time_promotion_ratio(%)',
                width=1)
        plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pic_path)

    def process(self, df, use_db=1):
        dfs = []
        sheet_names = []
        df['test_time'] = df['timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['is_release'] = self.args_.is_release

        # assure column order
        columns_ = [
            'operator', 'mlu_hardware_time', 'mlu_io_efficiency',
            'mlu_compute_efficiency', 'mlu_interface_time',
            'mlu_workspace_size', 'file_path',
            'input', 'output', 'params', 'mlu_theory_ios', 'mlu_theory_ops',
            'is_io_bound', 'date', 'commit_id', 'mluop_branch',
            'test_time', 'mlu_platform',
            'mluop_version', 'protoName', 'status', 'is_release'
        ]
        if 'md5' in df.columns:
            columns_.append('md5')

        if 'driver_version' in df.columns and 'cnrt_version' in df.columns:
            columns_.append('driver_version')
            columns_.append('cnrt_version')
        if 'mlu_kernel_names' in df.columns:
            columns_.append('mlu_kernel_names')

        dfs.append(df[columns_].sort_values(by=['mlu_hardware_time'],
                                            ascending=False))
        sheet_names.append('cases')

        summary = self.get_operator_summary(df)
        if not summary.empty:
            dfs.append(summary)
            sheet_names.append('operator_summary')

        # means use database
        if use_db:
            network_time, case_in_network = self.get_network_summary(df)
            if not network_time.empty:
                dfs.append(network_time)
                sheet_names.append('network_summary')
                dfs.append(case_in_network)
                sheet_names.append('case_in_network')

        return dfs, sheet_names

    def dump_to_simple_tpi_network_excel(self, df, xlsx_path, frameworks):
        important_network_keys = Config.important_network_keyword
        all_network_sheets = df.keys()
        important_network_names = self.get_important_network_names(
            all_network_sheets, important_network_keys, frameworks)
        #   select important network row
        all_network_rows = df['summary']['网络名称'].to_list()
        important_network = self.get_important_network_names(
            all_network_rows, important_network_keys, frameworks)
        summary = df['summary']
        summary = summary.loc[summary.网络名称.isin(important_network)]
        summary = summary.reset_index(drop=True)
        #   select important network sheets
        self.get_important_network_sheet(df, important_network_names)
        #   compute device_time_percentage for every sheet
        dfs = []
        for sheet_name in df.keys():
            all_device_time = df[sheet_name]['总device时间(us)'].sum()
            df[sheet_name]['device_time_percentage'] = df[sheet_name][
                '总device时间(us)'] / all_device_time
            df[sheet_name] = df[sheet_name].sort_values(by=[
                ('device_time_percentage')
            ],
                                                        ascending=False)
            df[sheet_name]['device_time_percentage'] = df[sheet_name][
                'device_time_percentage'].apply('{:.2%}'.format)
            df[sheet_name] = self.move_column_location(
                df[sheet_name], 1, 'device_time_percentage')
            df[sheet_name] = df[sheet_name].reset_index(drop=True)
            df[sheet_name].rename(
                columns={'device_time_percentage': 'device时间占比'}, inplace=True)
            dfs.append(df[sheet_name])

        #   get top20 operators from important network
        #   filter metrics:'总device时间占比'
        top20_ops = pd.DataFrame()
        all_ops_sheet = pd.concat(dfs, ignore_index=True, sort=True)
        all_ops_sheet['IO瓶颈比例'] = all_ops_sheet['IO瓶颈比例'].str.strip(
            "%").astype(float) / 100
        group_by_op = all_ops_sheet.groupby('算子名称')

        data_columns = [
            '平均IO效率', '平均计算效率', '平均workspace(Bytes)', '平均device时间(us)', '总个数',
            '总device时间(us)', 'IO瓶颈比例', '总workspace(Bytes)', '总host时间(us)',
            '平均host时间(us)'
        ]
        for col in data_columns:
            if col in [
                    '总个数', '总device时间(us)', '总workspace(Bytes)', '总host时间(us)'
            ]:
                top20_ops[col] = group_by_op.agg({col: ['sum']})[col]['sum']
            else:
                top20_ops[col] = group_by_op.agg({col: ['mean']})[col]['mean']

        all_ops_time = top20_ops['总device时间(us)'].sum()
        top20_ops[
            'device_time_per'] = top20_ops['总device时间(us)'] / all_ops_time
        top20_ops['device_time_per'] = top20_ops['device_time_per'].apply(
            "{:.2%}".format)
        top20_ops['IO瓶颈比例'] = top20_ops['IO瓶颈比例'].apply("{:.2%}".format)
        top20_ops = top20_ops.sort_values(by=[('总device时间(us)')],
                                          ascending=False)
        top20_ops = top20_ops.reset_index()
        #   move 'device_time_percentage' column after '算子名称'
        top20_ops = self.move_column_location(top20_ops, 1, 'device_time_per')
        top20_ops.rename(columns={'device_time_per': 'device时间占比'},
                         inplace=True)
        top20_ops = top20_ops.head(20)

        #   df to excel
        with pd.ExcelWriter(xlsx_path) as writer:
            top20_ops.to_excel(writer,
                               sheet_name='top20_ops_data',
                               index=False)
            Processor.auto_width(top20_ops, writer, 'top20_ops_data',
                                 top20_ops.columns)
            summary.to_excel(writer, sheet_name='summary', index=False)
            Processor.auto_width(summary, writer, 'summary', summary.columns)
            for i in df.keys():
                df[i].to_excel(writer, sheet_name=i, index=False)
                Processor.auto_width(df[i], writer, i, df[i].columns)
        print("the ouput simple excel is " + xlsx_path)

    def dump_to_simple_comparision_tpi_excel(self, df, xlsx_path, frameworks,
                                             version_compare):
        important_network_keys = Config.important_network_keyword
        all_network_sheets = df.keys()
        important_network_names = self.get_important_network_names(
            all_network_sheets, important_network_keys, frameworks)
        important_network_names.append('summary')

        all_network_rows = df['summary']['网络名称'].to_list()
        important_network = self.get_important_network_names(
            all_network_rows, important_network_keys, frameworks)
        df['summary'] = df['summary'].loc[df['summary'].网络名称.isin(
            important_network)]
        df['summary'] = df['summary'].reset_index(drop=True)
        self.get_important_network_sheet(df, important_network_names)

        columns = [
            'mlu_hardware_time_new', 'mlu_hardware_time_baseline',
            'mlu_interface_time_new', 'mlu_interface_time_baseline',
            'mlu_workspace_size_new', 'mlu_workspace_size_baseline',
            'count_new', 'count_baseline'
        ]
        columns_bs = [
            'mlu_hardware_time', 'mlu_interface_time', 'mlu_workspace_size',
            'count'
        ]
        columns_bs_ch = ['device时间', 'interface时间', 'workspace大小', '个数']
        update_columns = [
            'operator', 'device_time_per_new', 'device_time_per_baseline'
        ]
        update_columns_ch = [
            '算子名称', 'device时间占比' + version_compare[0],
            'device时间占比' + version_compare[1]
        ]

        static_suffix = ['_new', '_baseline', '_promotion_ratio']
        columns_promotion_ratio = []
        ver_suffix = version_compare
        promotion = '_提升比例'
        #   generate new final columns names
        for i in range(0, len(columns_bs)):
            update_columns.append(columns_bs[i] + static_suffix[0] + '_sum')
            update_columns.append(columns_bs[i] + static_suffix[1] + '_sum')
            update_columns.append(columns_bs[i] + '_sum' + static_suffix[2])
            columns_promotion_ratio.append(columns_bs[i] + '_sum' +
                                           static_suffix[2])
            update_columns_ch.append('总' + columns_bs_ch[i] + ver_suffix[0])
            update_columns_ch.append('总' + columns_bs_ch[i] + ver_suffix[1])
            update_columns_ch.append('总' + columns_bs_ch[i] + promotion)
            if columns_bs[i] not in ['count']:
                update_columns.append(columns_bs[i] + static_suffix[0] +
                                      '_mean')
                update_columns.append(columns_bs[i] + static_suffix[1] +
                                      '_mean')
                update_columns.append(columns_bs[i] + '_mean' +
                                      static_suffix[2])
                columns_promotion_ratio.append(columns_bs[i] + '_mean' +
                                               static_suffix[2])
                update_columns_ch.append('平均' + columns_bs_ch[i] +
                                         ver_suffix[0])
                update_columns_ch.append('平均' + columns_bs_ch[i] +
                                         ver_suffix[1])
                update_columns_ch.append('平均' + columns_bs_ch[i] + promotion)
        for i in ['io_bound_percentage_new', 'io_bound_percentage_baseline']:
            update_columns.append(i)
        for i in ['IO瓶颈比例' + ver_suffix[0], 'IO瓶颈比例' + ver_suffix[1]]:
            update_columns_ch.append(i)

        # generate sum and mean data to groupby op
        data_col_sum = []
        data_col_mean = []
        data_columns_all = []
        for i in columns:
            data_col_sum.append(i + '_sum')
            data_columns_all.append(i + '_sum')
            if i not in ['count_new', 'count_baseline']:
                data_columns_all.append(i + '_mean')
                data_col_mean.append(i + '_mean')

        #   deal with 'summary' sheet
        top20_ops = pd.DataFrame()
        all_sheets = []
        dfs = pd.DataFrame()
        for i in important_network_names:
            if i not in ('summary'):
                dfs = pd.DataFrame()
                for col in columns:
                    group_by_op = df[i].groupby('operator')
                    dfs[col + '_sum'] = group_by_op.agg({col:
                                                         ['sum']})[col]['sum']
                    if col not in ['count_new', 'count_baseline']:
                        dfs[col + '_mean'] = group_by_op.agg({col: ['mean']
                                                              })[col]['mean']
                for i in range(0, 3):
                    dfs[columns_promotion_ratio[2 * i]] = (
                        dfs[data_columns_all[4 * i + 2]] -
                        dfs[data_columns_all[
                            (4 * i)]]) / dfs[data_columns_all[4 * i + 2]]
                    dfs[columns_promotion_ratio[2 * i + 1]] = (
                        dfs[data_columns_all[4 * i + 3]] -
                        dfs[data_columns_all[
                            (4 * i + 1)]]) / dfs[data_columns_all[4 * i + 3]]

                # compute io_bound_percentage
                dfs['counts_new'] = group_by_op.agg({'count_new': ['sum']
                                                     })['count_new']['sum']
                dfs['io_bound_percentage_new'] = group_by_op.apply(
                    lambda x: x[x['is_io_bound_new']]['count_new'].sum(
                    )) / dfs['counts_new']
                dfs['counts_baseline'] = group_by_op.agg(
                    {'count_baseline': ['sum']})['count_baseline']['sum']
                dfs['io_bound_percentage_baseline'] = group_by_op.apply(
                    lambda x: x[x['is_io_bound_baseline']][
                        'count_baseline'].sum()) / dfs['counts_baseline']

                dfs['count_sum_promotion_ratio'] = (
                    dfs['count_baseline_sum'] -
                    dfs['count_new_sum']) / dfs['count_baseline_sum']
                dfs = dfs.sort_values(by=[('mlu_hardware_time_new_sum')],
                                      ascending=False)
                dfs_ops_time_new = dfs['mlu_hardware_time_new_sum'].sum()
                dfs_ops_time_baseline = dfs[
                    'mlu_hardware_time_baseline_sum'].sum()
                dfs['device_time_per_new'] = (
                    dfs['mlu_hardware_time_new_sum']) / dfs_ops_time_new
                dfs['device_time_per_baseline'] = (
                    dfs['mlu_hardware_time_baseline_sum']
                ) / dfs_ops_time_baseline
                dfs['device_time_per_new'] = (
                    dfs['device_time_per_new']).apply("{:.2%}".format)
                dfs['device_time_per_baseline'] = (
                    dfs['device_time_per_baseline']).apply("{:.2%}".format)
                dfs = dfs.reset_index()
                dfs = dfs[update_columns]
                all_sheets.append(dfs)
                del dfs

        top20_ops = pd.DataFrame()
        all_ops_sheet = pd.concat(all_sheets, ignore_index=True, sort=True)
        #   gen top_20_ops sheet
        for i in columns_promotion_ratio:
            data_columns_all.append(i)

        group_by_op = all_ops_sheet.groupby('operator')
        for column in data_columns_all:
            if column in data_col_sum:
                top20_ops[column] = group_by_op.agg({column:
                                                     ['sum']})[column]['sum']
            else:
                top20_ops[column] = group_by_op.agg({column:
                                                     ['mean']})[column]['mean']

        top20_ops = top20_ops.sort_values(by=[('mlu_hardware_time_new_sum')],
                                          ascending=False)

        all_ops_time_new = top20_ops['mlu_hardware_time_new_sum'].sum()
        all_ops_time_baseline = top20_ops[
            'mlu_hardware_time_baseline_sum'].sum()
        top20_ops['io_bound_percentage_new'] = group_by_op.agg(
            {'io_bound_percentage_new':
             ['mean']})['io_bound_percentage_new']['mean']
        top20_ops['io_bound_percentage_baseline'] = group_by_op.agg(
            {'io_bound_percentage_baseline':
             ['mean']})['io_bound_percentage_baseline']['mean']
        top20_ops['device_time_per_new'] = (
            top20_ops['mlu_hardware_time_new_sum']) / all_ops_time_new
        top20_ops['device_time_per_baseline'] = (
            top20_ops['mlu_hardware_time_baseline_sum']
        ) / all_ops_time_baseline
        for i in ([
                'io_bound_percentage_new', 'io_bound_percentage_baseline',
                'device_time_per_new', 'device_time_per_baseline'
        ] + columns_promotion_ratio):
            top20_ops[i] = top20_ops[i].apply("{:.2%}".format)

        top20_ops = top20_ops.reset_index()
        top20_ops = top20_ops[update_columns]
        top20_ops.rename(columns=dict(zip(update_columns, update_columns_ch)),
                         inplace=True)
        top20_ops = top20_ops.head(20)

        #   modify columns and trans %
        for i in range(0, len(all_sheets)):
            for j in (
                    columns_promotion_ratio +
                ['io_bound_percentage_new', 'io_bound_percentage_baseline']):
                all_sheets[i][j] = all_sheets[i][j].apply("{:.2%}".format)
        for i in range(0, len(all_sheets)):
            all_sheets[i].rename(columns=dict(
                zip(all_sheets[i].keys(), update_columns_ch)),
                                 inplace=True)

        with pd.ExcelWriter(xlsx_path) as writer:
            top20_ops.to_excel(writer,
                               sheet_name='top20_ops_data',
                               index=False)
            df['summary'].to_excel(writer, sheet_name='summary', index=False)
            Processor.auto_width(top20_ops, writer, 'top20_ops_data',
                                 top20_ops.columns)
            Processor.auto_width(df['summary'], writer, 'summary',
                                 df['summary'].columns)
            for i in range(0, len(all_sheets)):
                all_sheets[i].to_excel(writer,
                                       sheet_name=important_network_names[i],
                                       index=False)
                Processor.auto_width(all_sheets[i], writer,
                                     important_network_names[i],
                                     all_sheets[i].columns)
        print("the ouput excel is " + xlsx_path)

    def move_column_location(self, df, loc, column_name):
        df_tmp = df[column_name]
        df = df.drop(column_name, axis=1)
        df.insert(loc, column_name, df_tmp)
        return df

    #   select important network sheet
    def get_important_network_sheet(self, df, important_network_names):
        not_important_names = []
        for sheet_name in df.keys():
            if sheet_name not in important_network_names:
                not_important_names.append(sheet_name)
        [df.pop(x) for x in not_important_names]

    def get_frameworks_names(self, input_fw):
        frameworks = []
        frameworks = input_fw.split(",")
        all_frameworks = ['pytorch', 'tf']
        for i in frameworks:
            if i not in all_frameworks:
                print(
                    "The framework name entered is incorrect, incorrect name is",
                    i)
                exit()
        return frameworks

    def get_version_numer(self, log_path, compare_path):
        log_filename = os.path.basename(log_path)
        compare_filename = os.path.basename(compare_path)
        version_compare = re.findall(r'\_\d+\.\d+\.\d+',
                                     log_filename) + re.findall(
                                         r'\_\d+\.\d+\.\d+', compare_filename)
        return version_compare

    def get_important_network_names(self, all_network,
                                    important_network_keyword,
                                    framework_names):
        network_names = []
        for item in important_network_keyword:
            for fw_name in framework_names:
                for key in all_network:
                    if re.search(item, key) and re.search(fw_name, key):
                        network_names.append(key)
        return network_names

    # process networks which not in the database
    def proc_special_network(self, all_network, network_time):
        all_network_name = all_network['whole_name'].to_list()
        network_in_database = network_time['whole_name'].to_list()
        special_network = all_network.loc[~all_network.whole_name.
                                          isin(network_in_database)].copy()
        special_network['mlu_hardware_time_sum'] = 0
        network_time = pd.concat(([network_time, special_network]),
                                 ignore_index=True)
        return network_time

    def get_tpi_data(self, df):
        dfs = []
        sheet_names = []

        engine = create_engine(
            "mysql+pymysql://cambricon:Myql-cam198@10.101.9.21/training_solution"
        )
        case_in_network = pd.read_sql_table('mluop_case_in_network_test',
                                            engine)
        case_list = pd.read_sql_table('mluop_case_information_benchmark_test',
                                      engine)
        network_list = pd.read_sql_table('mluop_network_list_test', engine)
        whole_name_columns = [
            'network_name', 'framework', 'mode', 'batchsize',
            'network_additional_information', 'version'
        ]
        network_list.fillna(0, inplace=True)
        network_list['whole_name'] = network_list[whole_name_columns].apply(
            lambda x: '_'.join(
                [str(i) for i in x[whole_name_columns] if i != None]),
            axis=1)
        case_in_network = pd.merge(case_in_network,
                                   network_list[[
                                       'network_id', 'whole_name',
                                       'network_name', 'up_to_date'
                                   ]],
                                   on=['network_id'])

        case_in_network = pd.merge(case_in_network, case_list, on=['case_id'])

        df['mlu_platform'] = [
            i.split('[')[0] for i in df['mlu_platform'].to_list()
        ]
        platform = df.loc[0, 'mlu_platform']
        if platform not in Config.platform_map:
            print(
                "generating tpi failed, platform not supported, check file_path"
            )
            exit()

        # append network
        case_in_network = case_in_network[case_in_network['up_to_date'] == 1]
        case_in_network = case_in_network[case_in_network[
            Config.platform_map[platform]] == 1]

        mluop_case_run = pd.merge(df,
                                 case_in_network[[
                                     'protoName', 'count', 'whole_name',
                                     'network_name', 'network_id'
                                 ]],
                                 on=['protoName'])

        if mluop_case_run.empty:
            print(
                "generating tpi failed, mluop_case_run is empty, check file_path"
            )
            exit()

        # some hard-code
        tpi_columns = [
            'mlu_hardware_time', 'mlu_interface_time', 'mlu_workspace_size',
            'mlu_io_efficiency', 'mlu_compute_efficiency'
        ]

        origin_columns = [
            'whole_name', 'mlu_io_efficiency', 'mlu_compute_efficiency',
            'mlu_workspace_size', 'mlu_workspace_size_mean',
            'mlu_interface_time', 'mlu_interface_time_mean',
            'mlu_hardware_time', 'mlu_hardware_time_mean', 'counts',
            'io_bound_percentage'
        ]

        columns = [
            '网络名称',
            '平均IO效率',
            '平均计算效率',
            '总workspace(Bytes)',
            '平均workspace(Bytes)',
            '总host时间(us)',
            '平均host时间(us)',
            '总device时间(us)',
            '平均device时间(us)',
            '总个数',
            'IO瓶颈比例',
        ]

        epi = 0.0001

        for column in tpi_columns:
            mluop_case_run[
                column] = mluop_case_run[column] * mluop_case_run['count']

        group_by_network = mluop_case_run.groupby("whole_name")

        # assure column order
        tpi_network = pd.DataFrame()
        tpi_network['mlu_io_efficiency'] = group_by_network.apply(
            lambda x: x[x['is_io_bound']]['mlu_io_efficiency'].sum() / max(
                x[x['is_io_bound']]['count'].sum(), epi))
        tpi_network['mlu_compute_efficiency'] = group_by_network.apply(
            lambda x: x[~x['is_io_bound']]['mlu_compute_efficiency'].sum(
            ) / max(x[~x['is_io_bound']]['count'].sum(), epi))

        for i in [
                'mlu_workspace_size', 'mlu_hardware_time', 'mlu_interface_time'
        ]:
            tpi_network[i] = group_by_network.agg({i: ['sum']})[i]['sum']

        tpi_network['counts'] = group_by_network.agg({'count':
                                                      ['sum']})['count']['sum']

        for i in [
                'mlu_workspace_size', 'mlu_hardware_time', 'mlu_interface_time'
        ]:
            tpi_network[i + '_mean'] = tpi_network[i] / tpi_network['counts']

        tpi_network['io_bound_percentage'] = group_by_network.apply(
            lambda x: x[x['is_io_bound']]['count'].sum(
            )) / tpi_network['counts']
        tpi_network['io_bound_percentage'] = tpi_network[
            'io_bound_percentage'].apply("{:.2%}".format)

        tpi_network = tpi_network.reset_index()
        tpi_network = pd.merge(tpi_network,
                               network_list[['whole_name', 'network_id']],
                               on=['whole_name'])
        tpi_network.rename(columns=dict(zip(origin_columns, columns)),
                           inplace=True)

        dfs.append(tpi_network)
        sheet_names.append("summary")

        # change for operator
        origin_columns[0] = 'operator'
        columns[0] = '算子名称'

        for network, index in group_by_network.groups.items():
            if len(network) > 27:
                sheet_name = network[:27]
            else:
                sheet_name = network

            group_by_op = mluop_case_run.loc[index].groupby('operator')
            tpi = pd.DataFrame()
            tpi['mlu_io_efficiency'] = group_by_op.apply(
                lambda x: x[x['is_io_bound']]['mlu_io_efficiency'].sum() / max(
                    x[x['is_io_bound']]['count'].sum(), epi))

            tpi['mlu_compute_efficiency'] = group_by_op.apply(
                lambda x: x[~x['is_io_bound']]['mlu_compute_efficiency'].sum(
                ) / max(x[~x['is_io_bound']]['count'].sum(), epi))

            for i in [
                    'mlu_workspace_size', 'mlu_hardware_time',
                    'mlu_interface_time'
            ]:
                tpi[i] = group_by_op.agg({i: ['sum']})[i]['sum']

            tpi['counts'] = group_by_op.agg({'count': ['sum']})['count']['sum']

            for i in [
                    'mlu_workspace_size', 'mlu_hardware_time',
                    'mlu_interface_time'
            ]:
                tpi[i + '_mean'] = tpi[i] / tpi['counts']

            tpi['io_bound_percentage'] = group_by_op.apply(
                lambda x: x[x['is_io_bound']]['count'].sum()) / tpi['counts']
            tpi['io_bound_percentage'] = tpi['io_bound_percentage'].apply(
                "{:.2%}".format)

            tpi = tpi.reset_index()
            tpi.rename(columns=dict(zip(origin_columns, columns)),
                       inplace=True)

            dfs.append(tpi)
            sheet_names.append(sheet_name.lower())

        # remove duplicate
        name2idx = {}
        for idx, name in enumerate(sheet_names):
            if name not in name2idx:
                name2idx[name] = []
            name2idx[name].append(idx)

        for name, idxes in name2idx.items():
            if len(idxes) > 1:
                for i in range(len(idxes)):
                    sheet_names[
                        idxes[i]] = sheet_names[idxes[i]] + "_" + str(i)

        return mluop_case_run, dfs, sheet_names

    def dump_tpi_excel(self, dfs, sheet_names, tpi_path):
        Processor.dfs_to_excel(dfs, sheet_names, tpi_path)
        print("the tpi excel is " + tpi_path)

    def compare_tpi(self, case_run, case_run_bl, tpi_dfs, tpi_baseline,
                    sheet_names, tpi_compare_path, version_compare):
        summary_compare = pd.merge(tpi_dfs[0],
                                   tpi_baseline[0],
                                   suffixes=version_compare,
                                   on=['网络名称', 'network_id'])
        dfs = []
        summary_compare['device时间提升(us)'] = summary_compare[
            '总device时间(us)' +
            version_compare[1]] - summary_compare['总device时间(us)' +
                                                  version_compare[0]]
        summary_compare['device时间提升比例'] = summary_compare[
            'device时间提升(us)'] / summary_compare['总device时间(us)' +
                                                version_compare[1]]
        summary_compare = summary_compare.sort_values(by=[('device时间提升比例')],
                                                      ascending=False)
        summary_compare['device时间提升比例'] = summary_compare[
            'device时间提升比例'].apply("{:.2%}".format)
        summary_compare['host时间提升(us)'] = summary_compare[
            '总host时间(us)' +
            version_compare[1]] - summary_compare['总host时间(us)' +
                                                  version_compare[0]]
        summary_compare['host时间提升比例'] = summary_compare[
            'host时间提升(us)'] / summary_compare['总host时间(us)' +
                                              version_compare[1]]
        summary_compare['host时间提升比例'] = summary_compare['host时间提升比例'].apply(
            "{:.2%}".format)

        summary_compare['workspace提升(Bytes)'] = summary_compare[
            '总workspace(Bytes)' +
            version_compare[1]] - summary_compare['总workspace(Bytes)' +
                                                  version_compare[0]]
        summary_compare['workspace提升比例'] = summary_compare[
            'workspace提升(Bytes)'] / summary_compare['总workspace(Bytes)' +
                                                    version_compare[1]]
        summary_compare['workspace提升比例'] = summary_compare[
            'workspace提升比例'].apply("{:.2%}".format)
        # more hard code
        columns = [
            '网络名称', 'device时间提升(us)', 'device时间提升比例', 'host时间提升(us)',
            'host时间提升比例', 'workspace提升(Bytes)', 'workspace提升比例', 'network_id'
        ]
        for i in [
                '总device时间(us)', '总host时间(us)', '总workspace(Bytes)', '平均IO效率',
                '平均计算效率', '总个数', '平均device时间(us)', '平均host时间(us)',
                '平均workspace(Bytes)', 'IO瓶颈比例'
        ]:
            for j in version_compare:
                columns.append(i + j)
        dfs.append(summary_compare[columns])

        group_by_network = case_run.groupby("whole_name")
        group_by_network_bl = case_run_bl.groupby("whole_name")

        # assume cases are the same
        for network, index in group_by_network.groups.items():
            one_network = case_run[case_run['whole_name'] == network]
            one_network_bl = case_run_bl[case_run_bl['whole_name'] == network]
            merge_columns = ['protoName', 'operator', 'file_path']
            tpi_compare = pd.merge(one_network,
                                   one_network_bl,
                                   suffixes=Config.suffix,
                                   on=merge_columns)
            tpi_compare['mlu_hardware_time_promotion'] = tpi_compare[
                'mlu_hardware_time' +
                Config.suffix[1]] - tpi_compare['mlu_hardware_time' +
                                                Config.suffix[0]]
            tpi_compare['mlu_hardware_time_promotion_ratio'] = tpi_compare[
                'mlu_hardware_time_promotion'] / tpi_compare[
                    'mlu_hardware_time' + Config.suffix[1]]
            tpi_compare['mlu_hardware_time_promotion_ratio'] = tpi_compare[
                'mlu_hardware_time_promotion_ratio'].apply("{:.2%}".format)

            tpi_compare['mlu_interface_time_promotion'] = tpi_compare[
                'mlu_interface_time' +
                Config.suffix[1]] - tpi_compare['mlu_interface_time' +
                                                Config.suffix[0]]
            tpi_compare['mlu_interface_time_promotion_ratio'] = tpi_compare[
                'mlu_interface_time_promotion'] / tpi_compare[
                    'mlu_interface_time' + Config.suffix[1]]
            tpi_compare['mlu_interface_time_promotion_ratio'] = tpi_compare[
                'mlu_interface_time_promotion_ratio'].apply("{:.2%}".format)

            tpi_compare['mlu_workspace_size_promotion'] = tpi_compare[
                'mlu_workspace_size' +
                Config.suffix[1]] - tpi_compare['mlu_workspace_size' +
                                                Config.suffix[0]]
            tpi_compare['mlu_workspace_size_promotion_ratio'] = tpi_compare[
                'mlu_workspace_size_promotion'] / tpi_compare[
                    'mlu_workspace_size' + Config.suffix[1]]
            tpi_compare['mlu_workspace_size_promotion_ratio'] = tpi_compare[
                'mlu_workspace_size_promotion_ratio'].apply("{:.2%}".format)
            # arange excel columns
            columns_ = [
                'operator', 'mlu_hardware_time_promotion',
                'mlu_hardware_time_promotion_ratio',
                'mlu_interface_time_promotion',
                'mlu_interface_time_promotion_ratio',
                'mlu_workspace_size_promotion',
                'mlu_workspace_size_promotion_ratio', 'count_new',
                'count_baseline', 'mlu_hardware_time_new',
                'mlu_hardware_time_baseline', 'mlu_interface_time_new',
                'mlu_interface_time_baseline', 'mlu_workspace_size_new',
                'is_io_bound_new', 'is_io_bound_baseline',
                'mlu_workspace_size_baseline', 'file_path'
            ]
            dfs.append(tpi_compare[columns_])

        Processor.dfs_to_excel(dfs, sheet_names, tpi_compare_path)
        print("the tpi comparison excel is " + tpi_compare_path)
        return dfs, sheet_names

    def mapping_df_types(self, df):
        dtype_dict = {}
        for i, j in zip(df.columns, df.dtypes):
            if i in Config.case_info_keys:
                dtype_dict.update({i: types.JSON()})
        return dtype_dict

    def update_database(self, dfs, is_truncate, sheet_names):
        mluop_case_run = self.append_network_info(dfs[0])

        # when all cases recorded in db, md5 not exists
        if 'md5' in mluop_case_run.columns:
            mluop_case_run.drop(columns=['md5'], inplace=True)

        type_info = self.mapping_df_types(dfs[0])
        action = "replace" if is_truncate else "append"
        mluop_case_run.to_sql('mluop_case_run_test',
                             con=self.db_.engine_rainbow,
                             if_exists=action,
                             dtype=type_info,
                             index=False)
        print("update mluop_case_run_test successfully")

        # TODO(operator_summary): what about no cases
        tmp_idx = 1
        if 'operator_summary' in sheet_names:
            operator_summary = pd.merge(dfs[tmp_idx], self.db_.owner_resources, how="left").fillna(value="unknown")
            operator_summary.to_sql("mluop_operator_summary_test",
                                    con=self.db_.engine_rainbow,
                                    if_exists=action,
                                    index=False)
            tmp_idx += 1
            print("update mluop_operator_summary_test successfully")
        else:
            logging.warn("The test cases are all small cases(mlu_hardware_time< 30), ignore to update mluop_operator_summary_test")

        if 'network_summary' in sheet_names:
            network_summary = dfs[tmp_idx].copy()
            if "mlu_hardware_time_sum_database" in network_summary.columns:
                network_summary.drop(columns=['mlu_hardware_time_sum_database'],
                                inplace=True)
            network_summary.to_sql("mluop_network_summary_test",
                                con=self.db_.engine_rainbow,
                                if_exists=action,
                                index=False)
            tmp_idx += 1
            print("update mluop_network_summary_test successfully")
        else:
            logging.warn("warning: empty network_summary get, ignore to update mluop_network_summary_test!")

    def run(self, dfs):
        logging.info("Processor run start")
        if len(dfs) > 0:
            summary, sheet_names_new = self.process(dfs[0], self.args_.use_db)
            # remove the cases that the scales are duplicated
            if self.args_.deduplication:
                print("self.args_.deduplication=", self.args_.deduplication)
                Processor.dfs_to_excel_deduplication(summary, sheet_names_new, self.args_.xlsx_path)
            else:
                Processor.dfs_to_excel(summary, sheet_names_new, self.args_.xlsx_path)

            print("the ouput excel is " + self.args_.xlsx_path)

            if self.args_.case_run:
                self.update_database(summary, self.args_.truncate_case_run, sheet_names_new)

            if self.args_.tpi:
                #TODO(tpi): refactor get tpi and compare
                case_run, tpi_dfs, tpi_sheet_names = self.get_tpi_data(dfs[0])
                self.dump_tpi_excel(tpi_dfs, tpi_sheet_names, self.tpi_path)
                if self.args_.simple_tpi:
                    dic = dict(zip(tpi_sheet_names, tpi_dfs))
                    self.dump_to_simple_tpi_network_excel(
                        dic, self.simple_tpi_path, self.frameworks_name)

        if len(dfs) > 1:
            if self.args_.tpi:
                case_run_bl, tpi_baseline, names = self.get_tpi_data(dfs[1])
                tpi_comp_dfs, tpi_comp_sheet_names = self.compare_tpi(
                    case_run, case_run_bl, tpi_dfs, tpi_baseline,
                    tpi_sheet_names, self.tpi_compare_path,
                    self.version_compare)
                if self.args_.simple_tpi:
                    dic = dict(zip(tpi_comp_sheet_names, tpi_comp_dfs))
                    self.dump_to_simple_comparision_tpi_excel(
                        dic, self.tpi_compare_simple_path,
                        self.frameworks_name, self.version_compare)

            summary_bl, sheet_names_old = self.process(dfs[1], self.args_.use_db)
            compare_xlsx_path = self.args_.xlsx_path.replace(
                ".xlsx", "_comparison.xlsx")
            compare_dfs, sheet_names = self.compare_process(
                summary, sheet_names_new, summary_bl, sheet_names_old)
            pic_path = self.args_.xlsx_path.replace(".xlsx", ".png")
            self.generate_pic(compare_dfs[0], pic_path)
            Processor.dfs_to_excel(compare_dfs, sheet_names, compare_xlsx_path)
            print("the ouput comparison information excel is " +
                  compare_xlsx_path)
        logging.info("Processor run end")
        return 0
