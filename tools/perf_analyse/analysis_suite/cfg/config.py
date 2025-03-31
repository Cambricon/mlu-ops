#!/usr/bin/env python3
#coding:utf-8

import enum
import logging
import json

# miscellaneous config
class Config:
    # global: supported platform
    platform_map = {
        'MLU370-S4': 'MLU370',
        'MLU370-X4': 'MLU370',
        'MLU370-X8': 'MLU370',
        'MLU590': 'MLU590',
    }

    # gtest_parser: for case information
    case_info_keys = ["input", "output", "params"]
    # for database: network_info
    network_info_keys = [
        'network_id',
        'network_name',
        'framework',
        'precision_mode',
        'batchsize',
        'network_additional_information',
        'project_version'
    ]

    # for database: performance database
    environment_keys = [
        'commit_id',                # 代码commit id
        'mluop_version',            # mluops版本
        'driver_version',           # 驱动版本
        'cnrt_version',             # cnrt版本
        'time_stamp',               # 测试时间
        'mluop_branch',             # mluops分支
        'mlu_platform',             # mlu型号
        'sn',                       # mlu序列号
        'liner_memory',             # 线性内存
        'compress_memory',          # 内存压缩
        'zero_input',               # 全零输入
        'random_mlu_address',       # 随机地址
        'fast_allocate',            # fast allocate
    ]

    # gtest_parser: for parsing input
    xml_properties_map = {
        'op_name': 'operator',
        'hardware_time_mlu': 'mlu_hardware_time',
        'interface_time_mlu': 'mlu_interface_time',
        'io_efficiency_mlu': 'mlu_io_efficiency',
        'compute_efficiency_mlu': 'mlu_compute_efficiency',
        'case_path': 'file_path',
        'workspace_size_mlu': 'mlu_workspace_size',
        'kernel_names_mlu': 'mlu_kernel_names',
        'theory_ops': 'mlu_theory_ops',
        'theory_ios': 'mlu_theory_ios',
        'compute_force': 'mlu_computeforce',
        'io_bandwidth': 'mlu_iobandwidth',
        'workspace_size_gpu': 'gpu_workspace_size',
        'hardware_time_gpu': 'gpu_hardware_time',
        'io_efficiency_gpu': 'gpu_io_efficiency',
        'compute_efficiency_gpu': 'gpu_compute_efficiency',
        'repeat_num': 'repeat_num',
    }
    log_keyword_map = {
        'RUN': 'operator',
        'MLU Hardware Time': 'mlu_hardware_time',
        'MLU Interface Time': 'mlu_interface_time',
        'MLU IO Efficiency': 'mlu_io_efficiency',
        'MLU Compute Efficiency': 'mlu_compute_efficiency',
        'MLU Workspace Size': 'mlu_workspace_size',
        'MLU Kernel Name(s)': 'mlu_kernel_names',
        'MLU TheoryOps': 'mlu_theory_ops',
        'MLU TheoryIOs': 'mlu_theory_ios',
        'MLU ComputeForce': 'mlu_computeforce',
        'MLU IoBandWidth': 'mlu_iobandwidth',
        'GPU Hardware Time': 'gpu_hardware_time',
        'GPU IO Efficiency': 'gpu_io_efficiency',
        'GPU Compute Efficiency': 'gpu_compute_efficiency',
        'GPU Workspace Size': 'gpu_workspace_size',
        '^      OK': 'file_path',
        '^  FAILED': 'file_path',
        'repeat_num': 'repeat_num',
    }

    # gtest_parser: for preprocessing
    float_columns = [
        'mlu_hardware_time',
        'mlu_interface_time',
        'mlu_io_efficiency',
        'mlu_compute_efficiency',
        'mlu_workspace_size',
        'gpu_hardware_time',
        'gpu_io_efficiency',
        'gpu_compute_efficiency',
        'gpu_workspace_size',
        'mlu_theory_ios',
        'mlu_theory_ops',
        'mlu_computeforce',
        'mlu_iobandwidth',
        'repeat_num'
    ]

    # perf_analyser: for comparison
    suffix = ["_new", "_baseline"]
    promotion_suffix = ["_promotion", "_promotion_ratio"]

    # perf_analyser: for summary
    summary_columns = [
        'filtered_case_number', 'mlu_io_efficiency_mean', 'mlu_compute_efficiency_mean',
        'mlu_hardware_time_sum', 'good_rate', 'qualified_rate', 'unqualified_rate', 'poor_rate'
    ]

    # perf_analyser: for TPI
    epi = 0.0001

    # perf_analyser: for simple TPI
    important_network_keyword = [
        'resnet50v1.5',
        'ssd',
        'maskrcnn',
        'transformer',
        'bert',
        'mobilenetv2',
        'inceptionv3',
        'yolov3',
    ]

class PerfConfig:
    def __init__(self, filename="./perf.json"):
        try:
            from analysis_suite.utils import json_helper
            attrs = json_helper.read_json(filename)
            self.attrs = attrs
        except Exception:
            logging.warn("Error in reading `perf.json` file, use default configure")
            self.attrs = {
                "criterion": {
                    "good": [0.6, float('inf')],
                    "qualified": [0.3, 0.6],
                    "unqualified": [0.02, 0.3],
                    "poor": [float('-inf'), 0.02]
                },
                "ignore_case": {
                    "mlu_hardware_time": 15
                }
            }

class ColDef:
    mlu_platform                                                        = 'mlu_platform'                                       # mlu 型号； str
    mluop_version                                                      = 'mluop_version'                                     # mluops 版本号； str
    date                                                                = 'date'                                               # 测试日期； str
    time_stamp                                                          = 'time_stamp'
    test_time                                                           = 'test_time'                                          # 测试时间，是timestamp的别名； str
    is_release                                                          = 'is_release'                                         # 是否发行；bool
    operator                                                            = 'operator'                                           # 算子名； str
    operator_zh                                                         = '算子名称'                                            # operator的中文别名， str
    mlu_kernel_names                                                    = 'mlu_kernel_names'
    mlu_hardware_time                                                   = 'mlu_hardware_time'                                  # mlu 使用时间; float
    gpu_hardware_time                                                   = 'gpu_hardware_time'                                  # mlu 使用时间; float
    mlu_hardware_time_new                                               = 'mlu_hardware_time_new'
    mlu_hardware_time_baseline                                          = 'mlu_hardware_time_baseline'                         # 对比表 中新旧两个mlu使用时间；float
    mlu_hardware_time_promotion                                         = 'mlu_hardware_time_promotion'                        # 提升值，可能为负
    mlu_hardware_time_promotion_ratio                                   = 'mlu_hardware_time_promotion_ratio'                  # 提升比例
    mlu_hardware_time_zh                                                = 'device时间'
    mlu_hardware_time_sum                                               = 'mlu_hardware_time_sum'                              # 和
    mlu_hardware_time_sum_promotion                                     = 'mlu_hardware_time_sum_promotion'
    mlu_hardware_time_sum_promotion_ratio                               = 'mlu_hardware_time_sum_promotion_ratio'
    mlu_hardware_time_sum_database                                      = 'mlu_hardware_time_sum_database'
    mlu_hardware_time_mean                                              = 'mlu_hardware_time_mean'
    mlu_hardware_time_new_sum                                           = 'mlu_hardware_time_new_sum'
    mlu_hardware_time_baseline_sum                                      = 'mlu_hardware_time_baseline_sum'
    mlu_io_efficiency                                                   = 'mlu_io_efficiency'                                  # mlu io效率; float
    mlu_io_efficiency_mean                                              = 'mlu_io_efficiency_mean'
    mlu_compute_efficiency                                              = 'mlu_compute_efficiency'                             # mlu 计算效率； float
    mlu_compute_efficiency_mean                                         = 'mlu_compute_efficiency_mean'
    mlu_interface_time                                                  = 'mlu_interface_time'
    mlu_interface_time_zh                                               = 'interface时间'
    mlu_interface_time_new                                              = 'mlu_interface_time_new'
    mlu_interface_time_baseline                                         = 'mlu_interface_time_baseline'
    mlu_interface_time_mean                                             = 'mlu_interface_time_mean'
    mlu_interface_time_promotion                                        = 'mlu_interface_time_promotion'
    mlu_interface_time_promotion_ratio                                  = 'mlu_interface_time_promotion_ratio'
    mlu_workspace_size                                                  = 'mlu_workspace_size'
    mlu_workspace_size_zh                                               = 'workspace大小'
    mlu_workspace_size_new                                              = 'mlu_workspace_size_new'
    mlu_workspace_size_baseline                                         = 'mlu_workspace_size_baseline'
    mlu_workspace_size_mean                                             = 'mlu_workspace_size_mean'
    mlu_workspace_size_promotion                                        = 'mlu_workspace_size_promotion'
    mlu_workspace_size_promotion_ratio                                  = 'mlu_workspace_size_promotion_ratio'
    file_path                                                           = 'file_path'                                         # case文件路径
    input                                                               = 'input'                                             # 输入信息
    output                                                              = 'output'                                            # 输出信息
    params                                                              = 'params'                                            # 参数
    mlu_theory_ios                                                      = 'mlu_theory_ios'
    mlu_theory_ops                                                      = 'mlu_theory_ops'
    is_io_bound                                                         = 'is_io_bound'                                       # 是否IO瓶颈
    is_io_bound_new                                                     = 'is_io_bound_new'
    is_io_bound_baseline                                                = 'is_io_bound_baseline'
    commit_id                                                           = 'commit_id'
    mluop_branch                                                       = 'mluop_branch'
    protoName                                                           = 'protoName'                                          # case文件名，和file_path有关联
    status                                                              = 'status'                                             # 良好，合格，不合格
    md5                                                                 = 'md5'
    count                                                               = 'count'
    count_zh                                                            = '个数'
    count_new                                                           = 'count_new'
    count_baseline                                                      = 'count_baseline'
    counts                                                              = 'counts'
    counts_zh                                                           = '总个数'
    counts_new                                                          = 'counts_new'
    counts_baseline                                                     = 'counts_baseline'
    count_sum_promotion_ratio                                           = 'count_sum_promotion_ratio'
    count_baseline_sum                                                  = 'count_baseline_sum'
    count_new_sum                                                       = 'count_new_sum'
    count_percentage_zh                                                 = 'count占比'
    network_id                                                          = 'network_id'
    all_case_number                                                     = 'all_case_number'
    filtered_case_number                                                = 'filtered_case_number'
    io_bound_case_number                                                = 'io_bound_case_number'
    io_bound_percentage                                                 = 'io_bound_percentage'
    io_bound_percentage_new                                             = 'io_bound_percentage_new'
    io_bound_percentage_baseline                                        = 'io_bound_percentage_baseline'
    compute_bound_case_number                                           = 'compute_bound_case_number'
    case_id                                                             = 'case_id'
    network_time                                                        = 'network_time'
    ops_promotion_to_network                                            = 'ops_promotion_to_network'
    network_name                                                        = 'network_name'
    network_name_zh                                                     = '网络名称'
    framework                                                           = 'framework'
    framework_zh                                                        = '框架'
    precision_mode                                                      = 'precision_mode'
    batchsize                                                           = 'batchsize'
    network_additional_information                                      = 'network_additional_information'
    project_version                                                     = 'project_version'
    driver_version                                                      = 'driver_version'
    cnrt_version                                                        = 'cnrt_version'
    total_device_time_zh                                                = '总device时间(us)'
    device_time_per                                                     = 'device_time_per'
    device_time_per_new                                                 = 'device_time_per_new'
    device_time_per_baseline                                            = 'device_time_per_baseline'
    device_time_promotion_zh                                            = 'device时间提升(us)'
    device_time_promotion_ratio_zh                                      = 'device时间提升比例'
    device_time_percentage                                              = 'device_time_percentage'
    device_time_percentage_zh                                           = 'device时间占比'
    device_time_mean_zh                                                 = '平均device时间(us)'
    io_bottleneck_ratio_zh                                              = 'IO瓶颈比例'
    io_bottleneck_sum_zh                                                = 'IO瓶颈数量'
    io_efficiency_mean_zh                                               = '平均IO效率'
    compute_efficiency_mean_zh                                          = '平均计算效率'
    workspace_size_mean_zh                                              = '平均workspace(Bytes)'
    workspace_size_sum_zh                                               = '总workspace(Bytes)'
    workspace_size_promotion_zh                                         = 'workspace提升(Bytes)'
    workspace_size_promotion_ratio_zh                                   = 'workspace提升比例'
    host_time_sum_zh                                                    = '总host时间(us)'
    host_time_mean_zh                                                   = '平均host时间(us)'
    host_time_promotion_ratio_zh                                        = 'host时间提升比例'
    host_time_promotion_zh                                              = 'host时间提升(us)'
    whole_name                                                          = 'whole_name'                                              # 由一些字段拼接而成
    operator_counts_ratio_in_all_networks_zh                            = '算子在所有网络中count总个数占比'
    operator_devices_time_sum_ratio_in_all_network_zh                   = '算子在所有网络中device总时长占比'
    operator_devices_time_ratio_in_network_sum_zh                       = '算子在各网络中device时长占比总和'
    operator_counts_ratio_in_network_sum_zh                             = '算子在各网络中count数占比总和'
    case_source                                                         = 'case_source'
    case_source_zh                                                      = 'case来源'
    up_to_date                                                          = 'up_to_date'
    gen_date                                                            = 'gen_date'
    owner                                                               = 'owner'
    resources                                                           = 'resources'
    env_id                                                              = 'env_id'
    mlu_hardware_time_proportion                                        = 'mlu_hardware_time_proportion'
    mlu_hardware_time_sum_proportion                                    = 'mlu_hardware_time_sum_proportion'
    repeat_num                                                        = 'repeat_num'                                            # repeat time in case list

class DBConfig:
    pass_port = {
        "user_name": b'',
        "pass_word": b''
    }

    class DB_Name(enum.Enum):
        training_solution = 0
        rainbow = 1
        local_db = 2

    # real name of each database
    DB_Name_mp = {
        DB_Name.training_solution: "training_solution",
        DB_Name.rainbow: "rainbow",
        DB_Name.local_db: "mluops_perf_analyse_local.db",
    }

    class Table_Name(enum.Enum):
        case_in_network = 0,
        network_list = 1,
        case_list = 2,
        network_summary = 3,
        owner_resources = 4,
        environment_info = 5,
        performance = 6,

    # real name of each table
    Table_Name_mp = {
        Table_Name.case_in_network: "mluops_case_in_network_test", # training_solution
        Table_Name.network_list: "mluops_network_list_test", # training_solution
        Table_Name.case_list: "mluops_case_information_benchmark_test", # training_solution
        Table_Name.network_summary: "mluops_network_summary_test", # rainbow
        Table_Name.owner_resources: "mluops_owner_resources_test", # rainbow
        Table_Name.environment_info: "mluops_performance_test_environment", # rainbow
        Table_Name.performance: "mluops_performance_test_result", # rainbow
    }
