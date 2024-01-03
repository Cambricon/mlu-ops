#!/usr/bin/env python3

"""wrap 'ld' command
"""

import sys
import os
import shlex
import subprocess
import uuid
import tempfile
from pathlib import Path
import argparse
import logging
import enum
import atexit
import signal

class AddMode(enum.IntEnum):
  insert = 0
  append = 1

def gen_workaround_source():
    """sample codes to be compiled temporarily
    """
    hashname = lambda: uuid.uuid4().hex

#    rand_name = hashname()[:8]
#    yield """
#    #include <stdio.h>
#    void f{0} () {{
#      size_t a = 1;
#    }}
#    """.format(rand_name)
#    yield """
#    extern void f{0} ();
#    inline int s{0}() {{
#      f{0}();
#      return 0;
#    }}
#    static __attribute__((unused)) c = s{0}();
#    """.format(rand_name)

    yield AddMode.append, """
    __attribute__((weak))
    void f{0}() {{}}
    """.format(hashname()[:8])

    yield AddMode.insert, """
    __attribute__((weak))
    void f{0}() {{}}
    """.format(hashname()[:8])

    yield AddMode.append, """
#include <stdlib.h>
size_t workaround_aarch64_ld_bug_mluOpcore_11301_random_function_{0}(char *) {{
}}
""".format(hashname())

    yield AddMode.append, """
#include <stdlib.h>

static __attribute__((noinline))
bool workaround_aarch64_ld_bug_mluOpcore_11301_random_function_{0}(size_t arg0, char * arg1) {{
  return true;
}}

static __attribute__((unused)) bool _init = workaround_aarch64_ld_bug_mluOpcore_11301_random_function_{0}(1, NULL);

""".format(hashname())

    flip = 0
    modes = [AddMode.insert, AddMode.append]
    type_handler = ["float", "char *", "bool", "int"]
    for t in type_handler:
        yield modes[flip], """
#include <stdlib.h>
template <typename T>
void workaround_aarch64_ld_bug_mluOpcore_11301_random_function_{0}(T *) {{
}}

__attribute__((constructor)) static void _ctor() {{
  workaround_aarch64_ld_bug_mluOpcore_11301_random_function_{0}<{1}>(NULL);
}}
""".format(hashname(), t)
    flip = flip ^ 1

    fname = hashname()
    yield AddMode.insert, """
    char fx{0}() {{}}
    """.format(fname)

    yield AddMode.append, """
    extern char fx{0}();
    void call{0}() {{
        fx{0}();
    }}
    """.format(fname)

    yield None, None


class Linker:
    """wrap toolchain's `ld` from binutils
    """
    def __init__(self, *, compiler, target, obj):
        self._compiler = compiler
        self._target = target
        self._workaround_src_iter = gen_workaround_source()
        self._obj_name = obj
        self._files_to_be_cleanup = {self._obj_name,}
        self._ret = None
        atexit.register(self.cleanup)
        def sig_handler(signo, frame):
            self.cleanup()
            os._exit(-1)
        signal.signal(signal.SIGTERM, sig_handler)

    def __call__(self, args):
        """call `ld` to link, if link failed with ld internal bug, try to compile some source 
        to generate new symbol and link again
        """
        self._ret = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err = self._ret.stderr.decode('utf8')

        if self._ret.returncode and '_bfd_aarch64_erratum_843419_branch_to_stub' in err:
            logging.warning("encounter ld internal bug, try to generate random object to link as a workaround")
            # workaround for error pattern:
            # > ld: ... internal error, aborting at ... elfnn-aarch64.c:4812 in _bfd_aarch64_erratum_843419_branch_to_stub
            # > ... aarch64-linux-gnu/bin/ld: Please report this bug.
            mode, source = next(self._workaround_src_iter)
            if source is None:
                logging.error("Tried many times but cannot bypass ld internal bug")
                return self.on_failure()
            return self(self.build_cmd(mode, args, self.compile(source)))
#            return self(args + [self.compile(source)])
        elif self._ret.returncode:
            logging.error("link failed")
            return self.on_failure()
        else:
            return self.on_success()

    def build_cmd(self, mode, args, objname):
        if mode == AddMode.append:
            return args + [objname]
        return [args[0]] + [objname] + args[1:]

    def compile(self, source: str) -> str:
        """based on source contents, create temp file to compile and return compiled object name
        """
        src = tempfile.NamedTemporaryFile(mode='w', prefix='.'+tempfile.gettempprefix(), suffix='.cpp', dir=os.getcwd())
        self._files_to_be_cleanup.add(src.name)
        self._files_to_be_cleanup.add(src.name + ".o")
        src.write(source)
        src.flush()
        compile_command = "{compiler} -c {srcname} -o {srcname}.o -fPIC".format(
            compiler=self._compiler, srcname=src.name)
        # print(compile_command)
        subprocess.run(shlex.split(compile_command), check=True)
        return src.name + ".o"

    def on_success(self):
        sys.stderr.write(self._ret.stderr.decode('utf8'))
        sys.stderr.flush()
        sys.stdout.write(self._ret.stdout.decode('utf8'))
        sys.stdout.flush()
        self._files_to_be_cleanup.remove(self._obj_name)
        self.cleanup()

    def on_failure(self):
        sys.stdout.write(self._ret.stdout.decode('utf8'))
        sys.stdout.flush()
        sys.stderr.write(self._ret.stderr.decode('utf8'))
        sys.stderr.flush()
        self.cleanup()
        sys.exit(-1)

    def cleanup(self):
        for f in self._files_to_be_cleanup:
            if Path(f).exists(): # python3.5 does not have `Path.unlink(missing_ok=True)`
                Path(f).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="linker", description="wrap 'ld' command as a workaround for ld internal bug")
    parser.add_argument("--compiler", help="c++ compiler", required=True)
    parser.add_argument("--target", help="cpu target", required=True)
    parser.add_argument("command", help="command to be executed", nargs='*')
    args = parser.parse_args()
    ld = Linker(compiler=args.compiler, target=args.target, obj=args.command[args.command.index("-o") + 1])
    ld(args.command)
