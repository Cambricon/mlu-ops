#!/usr/bin/env python3
from __future__ import annotations

# need python3.8 or later
import sys
if sys.version_info < (3, 8, 0):
    print("ERROR: need Python >=3.8, your python version: {}".format(sys.version_info))
    sys.exit(-1)

"""
analyse log of cmake build (use Ninja not Unix Makefiles), get each object compile time
"""

import argparse
from pathlib import Path
import re
import enum
from concurrent.futures import ProcessPoolExecutor
import logging
from collections import namedtuple
import textwrap

logging.basicConfig(
    format='%(levelname)s:%(name)s [%(process)d-%(thread)d] [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    #    level=logging.DEBUG,
)
logger = logging.getLogger('analyse_compile_time')
logger.setLevel(logging.INFO)

from analysis_suite.utils import module_install_helper

MODULES = "lief pandas"
try:
    import pandas as pd
    import lief
except ImportError as e:
    logger.error(e)
    module_install_helper.try_pip_install_with_restart_once(logger, MODULES)

class StateMachine(enum.Enum):
    IDLE = 0
    MLU_MATCHED = 1
    CPP_MATCHED = 2

class ObjType(enum.IntEnum):
    MISC = 0
    MLU = 1
    CPP = 2

def match_elapsed_s(line: str) -> None | int:
    # CMake 3.23,  match 1630 in 'Elapsed time: 1630 s'
    # CMake 3.30,  match 1630 in 'Elapsed time (seconds): 1630'
    pattern_elapsed_time = re.compile('Elapsed time.*: (?P<time>\d+)( s)?')
    match = pattern_elapsed_time.search(line)
    if match:
        elapsed_s = int(match.groupdict()['time'])
        return elapsed_s
    return None

def match_generated_mlu(line: str) -> None | str:
    # match xxx.mlu.o in 'Generated xxx.mlu.o successfully'
    pattern_generated_mlu = re.compile('^Generated (?P<mlu_obj>\S+\.mlu\.o) successfully')
    match = pattern_generated_mlu.search(line)
    if match:
        mlu_obj = match.groupdict()['mlu_obj']
        return mlu_obj
    return None

CompileInfo = namedtuple('CompileInfo', ['cmd', 'obj'])
def match_generated_cpp(line: str) -> None | CompileInfo:
    # -o xxx.o -c xxx.c
    pattern_generated_cpp = re.compile('-o (?P<cpp_obj>\S+\.(cpp|cc|c)\.o) -c \S+\.(cpp|cc|c)')
    match = pattern_generated_cpp.search(line)
    if match:
        cpp_obj = match.groupdict()['cpp_obj']
        # TODO consider read compile_commands.json directly
        return CompileInfo(obj=cpp_obj, cmd=f'"{match.string}"')
    return None

def match_generating_mlu(line: str) -> None | str:
    # match xxx.mlu.o in '... Generating xxxx.mlu.o'
    pattern_generting_mlu = re.compile('Generating (?P<mlu_obj>\S+\.mlu\.o)$')
    match = pattern_generting_mlu.search(line)
    if match:
        #print("handle", match)
        mlu_obj = match.groupdict()['mlu_obj']
        return mlu_obj
    return None

def match_compile_mlu_cmd(line: str) -> None | CompileInfo:
    # match cncc compile .mlu/.dlp command
    pattern_compile_mlu_cmd = re.compile('/bin/cncc \S+\.mlu -c -o (?P<mlu_obj>\S+\.(mlu|dlp)\.o) .* --bang(-mlu)?-arch=.*')
    match = pattern_compile_mlu_cmd.search(line)
    if match:
        #print("handle", match)
        return CompileInfo(obj=match.groupdict()['mlu_obj'], cmd=f'"{match.string}"')
    return None

def fix_relative_path(obj: str, cwd: str | None = None) -> str:
    if (not Path(obj).is_absolute()) and cwd:
        return str(Path(cwd).joinpath(obj))
    return obj

class SymbolWrapper:
    @staticmethod
    def from_obj(obj, *, cwd: str | None = None):
        sym = SymbolWrapper(obj)
        sym.evaluate(cwd)
        return sym

    @property
    def cn_fatbin_bytes(self):
        return self._cn_fatbin_bytes

    @property
    def size_bytes(self):
        return self._size_bytes

    @property
    def obj(self):
        return str(self._obj)

    def __init__(self, obj):
        self._obj = obj
        self._cn_fatbin_bytes = 0
        self._size_bytes = 0

    def evaluate(self, cwd: str | None):
        if not Path(self._obj).exists():
            logger.warning(f"{self._obj} not found")
            return
        self._size_bytes = Path(self._obj).stat().st_size
        sym = lief.parse(str(self._obj))
        try:
            self._cn_fatbin_bytes = sym.get_section('.cn_fatbin').size
        except Exception as e:
            pass

class CMakeNinjaLogParser:
    def __init__(self):
        self.data_obj: list[str] = []
        self.data_obj_type: list[ObjType] = []
        self.data_elapsed_s: list[int] = []
        self.cmd_s: list[str] = []

    def record(self, obj: str, obj_type: ObjType, elapsed_s: int, cmd: str):
        self.data_obj.append(obj)
        self.data_obj_type.append(obj_type)
        self.data_elapsed_s.append(elapsed_s)
        self.cmd_s.append(cmd)
        logger.debug(f"{obj}, {elapsed_s}")

    def on_state_idle(self, line: str) -> None | StateMachine:
        # mlu_obj = match_generated_mlu(line)
        compile_info = match_compile_mlu_cmd(line)
        if compile_info:
            self.mlu_obj = compile_info.obj
            self.cmd = compile_info.cmd
            return StateMachine.MLU_MATCHED
        compile_info = match_generated_cpp(line)
        if compile_info:
            self.cpp_obj = compile_info.obj
            self.cmd = compile_info.cmd
            return StateMachine.CPP_MATCHED
        return None

    def on_state_mlu_matched(self, line: str) -> None | StateMachine:
        elapsed_s = match_elapsed_s(line)
        if elapsed_s:
            self.record(self.mlu_obj, ObjType.MLU, elapsed_s, self.cmd)
            self.mlu_obj = None
            self.cmd = None
            return StateMachine.IDLE
        return None

    def on_state_cpp_matched(self, line: str) -> None | StateMachine:
        elapsed_s = match_elapsed_s(line)
        #print("on_state_cpp_matched", line, elapsed_s)
        if elapsed_s:
            self.record(self.cpp_obj, ObjType.CPP, elapsed_s, self.cmd)
            self.cpp_obj = None
            self.cmd = None
            return StateMachine.IDLE
        return None

    def evaluate(self, output_: str, cwd: str | None):
        self.data_obj_short_names = [x.partition('CMakeFiles/')[-1] for x in self.data_obj]
        self.base_names = [Path(x).name for x in self.data_obj]
        self.data_obj = [fix_relative_path(obj, cwd) for obj in self.data_obj]
        table = pd.DataFrame(zip(
            self.data_obj_type,
            self.data_elapsed_s,
            self.base_names,
            self.data_obj_short_names,
            self.data_obj,
            self.cmd_s,
        ) , columns = [
            "obj_type",
            "elapsed_s",
            "base_name",
            "obj_short",
            "obj",
            "cmd",
        ])
        table.set_index('obj')

        syms: list[SymbolWrapper]
        with ProcessPoolExecutor() as executor:
            syms = list(executor.map(SymbolWrapper.from_obj, self.data_obj))

        for sym in syms:
            table.loc[table['obj'] == sym.obj, ('size_kb', 'cnfatbin_size_kb')] = sym.size_bytes / 1024, sym.cn_fatbin_bytes / 1024

        serializers = {
            '.csv': lambda t, o: t.to_csv(o),
            '.xls': lambda t, o: t.to_excel(o),
            '.xlsx': lambda t, o: t.to_excel(o),
        }
        serializers[Path(output_).suffix](table, output_)
        logger.info(f'write to {output_}')

def extract_compile_time_from_cmake_log(*, log_path: str, output_: str, cwd: str):
    state = StateMachine.IDLE
    parser = CMakeNinjaLogParser()

    for line in Path(log_path).read_text().splitlines():
        new_state: None | StateMachine = None
        #print("state", state)
        #print("handle", line)
        if state == StateMachine.IDLE:
            new_state = parser.on_state_idle(line)
        elif state == StateMachine.MLU_MATCHED:
            new_state = parser.on_state_mlu_matched(line)
        elif state == StateMachine.CPP_MATCHED:
            new_state = parser.on_state_cpp_matched(line)
        if new_state is not None:
            state = new_state

    parser.evaluate(output_, cwd)

class MyFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

help_text_example = textwrap.dedent("""
Example:
    python3 ./tools/Perf_Analyse/analyse_compile_time.py  --log-path /DEV_SOFT_TRAIN/(user)/compile.log  --working-directory $PWD/build

NOTE:
    You should use Ninja (ninja) instead of Unix Makefiles (make) and enable verbose build, e.g.:
    `./independent_build.sh -g Ninja -v .... 2>&1 | tee compile.log`
""")

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        prog='analyse_compile_time',
        description="analyse log of cmake build (use Ninja not Unix Makefiles), get each object compile time",
        formatter_class=MyFormatter,
        epilog=help_text_example,
    )
    parser.add_argument("--log-path", help="compile log path", required=True)
    parser.add_argument("--output", help="output file", default='size.xlsx')
    parser.add_argument("--working-directory", help="directory for build(for resolving relative path)", default='build')
    args = parser.parse_args()
    extract_compile_time_from_cmake_log(
        log_path=args.log_path,
        output_=args.output,
        cwd=args.working_directory,
    )
