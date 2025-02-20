"""
    reader for .json file
"""

__all__ = (
    "read_json",
)

import os
import json
import logging

# read json file
def read_json(file_path):
    if None == file_path:
        return None
    path = file_path

    # handle `~` in file path
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise Exception("File {} not exists".format(path))

    # parse json file
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error("Cannot find json file {}.".format(path))
        raise
    except json.JSONDecodeError:
        logging.error("Error when decoding json file {}.".format(path))
        raise
    except Exception:
        logging.error("Read json file {} failed.".format(path))
        raise
