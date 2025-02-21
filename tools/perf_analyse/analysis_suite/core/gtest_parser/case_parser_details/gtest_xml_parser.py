"""
    gtest XML parser

    Input: XML file.
    Output: test_info.TestInfo. If there's some case failed, export the list.
"""

__all__ = (
    "parse_gtest_xml",
)

import re
import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple

from analysis_suite.cfg.config import Config, ColDef, PerfConfig
from analysis_suite.core.gtest_parser import test_info
from analysis_suite.utils import excel_helper, builtin_types_helper

"""

Information we need is `env_info`, `testcase_namespace` and `property_namespace`.
Structure of current XML files is:

<testsuites (env_info)>
    <testsuite name="xxx">
        <properties>
            <property .../>
            ...
        </properties>
        <testcase .../>
        ...
    </testsuite>
    ...

    <testsuite name="xxx/TestSuite">
        <properties>
            <property .../>
            ...
        </properties>

        <testcase (testcase_namespace) />
        ...
        <testcase (testcase_namespace)>
            <properties>
                <property (property_namespace)/>
                ...
            </properties>
        </testcase>
        ...
        <testcase (testcase_namespace)>
            <failure message=xxxxx> </failure>
            ...
            <properties>
                <property (property_namespace)/>
                ...
            </properties>
        </testcase>
    </testsuite>
    ...
</testsuites>

"""

# for `env_info` in
#   <testsuites (env_info)>
def handle_xml_header(testsuites) -> Dict:
    env_dict_ori = {key: value for key, value in testsuites.attrib.items()}

    # preprecess header
    # remove '['
    env_dict_ori['mlu_platform'] = env_dict_ori['mlu_platform'].split('[')[0]
    # format timestamp
    #if 'timestamp' not in df.columns:
    #    df['timestamp'] = pd.Timestamp.now()
    env_dict_ori['time_stamp'] = datetime.strptime(env_dict_ori['timestamp'], "%Y-%m-%dT%H:%M:%S")

    env_dict = {key: env_dict_ori[key] for key in Config.environment_keys if key in env_dict_ori}

    return env_dict

# for
#   <testsuite name="xxx/TestSuite">
def is_testsuite_for_mluops(test_name: str) -> bool:
    if test_name.endswith("/TestSuite"):
        return True
    return False

# for
#   <testcase (testcase_namespace)>
def is_testcase(tag_name: str) -> bool:
    if "testcase" == tag_name:
        return True
    return False

# for `failure` in
#   <failure message=xxxxx> </failure>
def is_failed(testcase: ET.Element) -> bool:
    for child in testcase:
        if "failure" == child.tag:
            return True
        return False

# for failed testcase, need case_path
#   <property name="case_path" value="xxxx" />
def handle_faied_testcase(testcase: ET.Element) -> Dict:
    for property in testcase.find('properties'):
        if "case_path" == property.attrib['name']:
            return {Config.xml_properties_map[property.attrib['name']]: property.attrib['value']}

# for `property_namespace` in
#   <properties>
#       <property (property_namespace)/>
#       ...
#   </properties>
def extract_ok_properties(properties: ET.Element) -> Dict:
    # extract property
    data = {
        Config.xml_properties_map[property.attrib['name']]: property.attrib['value']
        for property in properties if property.attrib['name'] in Config.xml_properties_map.keys()
    }
    data.setdefault(ColDef.repeat_num, 1)

    # str -> float
    for k in Config.float_columns:
        data[k] = float(data[k])

    return data

# for ok testcase, need all properties and a diff factor
def handle_ok_testcase(testcase: ET.Element) -> Dict:
    properties = extract_ok_properties(testcase.find('properties'))

    # diff factor
    match = re.search(r'\d+', testcase.attrib['name'])
    properties.update({"diff_factor": int(match.group())})

    return properties

def to_testcase_list(
        testsuites: ET.Element,
        filter_failed_cases: bool
    ) -> Tuple[List[Dict], List[Dict]]:
    ok_lst = []
    failed_lst = []

    # there are several `testuite` in `testsuites`
    for testsuite in testsuites:
        # check testsuite.attrib['name'] to select `testsuite`
        if not is_testsuite_for_mluops(testsuite.attrib['name']):
            continue

        # there are several `testcase` in `testsuite`
        for child in testsuite:
            # child.tag may be `properties` or `testcase`. need `testcase` only
            # handle `self-closing tag`
            if (not is_testcase(child.tag)) or (0 == len(child)):
                continue
            # handle `testcase`
            if True == filter_failed_cases:
                if is_failed(child):
                    failed_lst.append(handle_faied_testcase(child))
                else:
                    ok_lst.append(handle_ok_testcase(child))
            else:
                ok_lst.append(handle_ok_testcase(child))

    return ok_lst, failed_lst

# Read XML body andreturn 2 pd.DataFrame to store ok/failed cases
def handle_xml_body(
        testsuites: ET.Element,
        filter_failed_cases: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # parse testcase in testsuites
    ok_lst, failed_lst = to_testcase_list(testsuites, filter_failed_cases)

    # merge ok list
    ok_cases = {}
    for case in ok_lst:
        builtin_types_helper.merge_dict(ok_cases, case)

    failed_cases = {}
    for case in failed_lst:
        builtin_types_helper.merge_dict(failed_cases, case)

    return pd.DataFrame(ok_cases), pd.DataFrame(failed_cases)

# Entrance Function
def parse_gtest_xml(
        file_path: str,
        filter_failed_cases: bool = False,
        export_failed_cases: bool = False,
    ) -> test_info.TestInfo:
    """
    Parse XML file and return success cases.
    If `export_failed_cases`, export failed cases to excel.
    """
    # parse XML file
    tree = ET.parse(file_path)
    testsuites = tree.getroot()

    # get environment information
    env = handle_xml_header(testsuites)

    # get performance information
    ok_df, failed_df = handle_xml_body(testsuites, filter_failed_cases)

    # export failed cases
    if True == filter_failed_cases and True == export_failed_cases:
        excel_helper.to_excel_helper(failed_df, file_path.split("/")[-1].replace(".xml", "") + "_failed_cases.xlsx")

    # return test_info.TestInfo
    return test_info.TestInfo(env, ok_df)
