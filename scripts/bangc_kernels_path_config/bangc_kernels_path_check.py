#!/usr/bin/env python3
import json

import os
import logging

def find_files_in_path(relative_path):
    list_path = []
    for root, dirs, files in os.walk(relative_path):
        for file in files:
            if file.endswith('.h') or file.endswith('.cpp') or file.endswith('.mlu') or file.endswith('.mluh'):
                full_path = os.path.join(root, file)
                list_path.append(full_path)
    return list_path


def get_relative_paths(absolute_path):
    # init
    relative_paths = []
    # base_folder = os.path.basename(absolute_path)
    base_folder = ""
    for folder, dirs, files in os.walk(absolute_path):
        # get relative_folder path
        relative_folder = os.path.relpath(folder, start=absolute_path)

        # concat path
        for file in files:
            if file.endswith('.h') or file.endswith('.cpp') or file.endswith('.mlu') or file.endswith('.mluh'):
                if relative_folder == '.':
                    full_path = os.path.join(base_folder, file)
                else:
                    full_path = os.path.join(base_folder, relative_folder, file)
                relative_paths.append(full_path)
    return relative_paths

def extract_headers(json_file_path, common_flag=True, header_flag=True, other_flag=True):
    header_files = []
    op_name = []
    try:
        with open(json_file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"The JSON file at {json_file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file at {json_file_path}.")
        return []

    if 'common' in config and common_flag:
        header_files.extend(config['common'])

    if 'operators' in config:
        for operator in config['operators']:
            if 'name' in operator:
                if isinstance(operator['name'], list):
                    op_name.extend(operator['name'])
                else:
                    op_name.append(operator['name'])
            
            if 'header' in operator and header_flag:
                if isinstance(operator['header'], list):
                    header_files.extend(operator['header'])
                else:
                    header_files.append(operator['header'])
            
            if 'other' in operator and other_flag:
                if isinstance(operator['other'], list):
                    header_files.extend(operator['other'])
                else:
                    header_files.append(operator['other'])

    return header_files, op_name


def find_json_files(absolute_path):
    json_files = []
    for root, dirs, files in os.walk(absolute_path):
        for file in files:
            if file.endswith('.json'):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return json_files


def check_header_files_in_list_path(header_files, list_path):
    for header_file in header_files:
        if header_file in list_path:
            logging.info(f"Header file {header_file} is found in the list_path.")
        else:
            logging.warning(f"Header file {header_file} is not found in the list_path.")


'''
params:
    1. header_files_all:       all .h/.mlu/.mluh/.cpp paths under "../../kernels" 
    2. header_files:           all .h/.mlu/.mluh/.cpp paths form JSON
    3. header_files_unique:    Deduplicated header_files
'''
if __name__ == "__main__":
    try:
        # get header_files_all
        mlu_ops_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        header_files_all = get_relative_paths(mlu_ops_path)

        # print(header_files_all)

        # get header_files, op_name
        header_files = []
        op_name_list = []
        
        json_paths = find_json_files(os.path.dirname(__file__))
        print(os.path.dirname(__file__))
        
        print(json_paths)

        for path in json_paths:
            files_path, op_name = extract_headers(path, True, True, True)
            print(path)
            header_files.extend(files_path) if isinstance(files_path, list) else header_files.append(files_path)
            op_name_list.extend(op_name) if isinstance(op_name, list) else header_files.append(op_name)
        
        # get header_files_unique
        header_files_unique = list(set(header_files))
        assert len(header_files_unique) is len(header_files), "There are duplicate paths in JSON files {}. ".format(json_paths)
        assert len(list(set(op_name_list))) is len(op_name_list), "There are duplicate op-name in JSON files {}. ".format(json_paths)

        for path in header_files_unique:
            assert path in header_files_all, f"The file path({path}) is not in mlu-ops/. "

        print("Bangc kernels path check success.")

    except Exception as e:
        print(f"[ERROR] Check failed. : {e}") 
