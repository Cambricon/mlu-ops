"""
    auto pip install python modules then reload
"""

__all__ = (
    'try_pip_install_with_restart_once',
)

import sys
import os
import shlex
import subprocess

def restart_script(*, enable_env):
    python_executable = sys.executable  # Get the path to the Python interpreter
    # script_path = os.path.abspath(__file__)
    script_path = sys.argv[0]           # Get the path to the currently executing script
    args = sys.argv[1:]                 # Get command-line arguments, excluding the script name
    os.environ[enable_env] = "TRUE"

    # Replace the current process with a new one
    os.execl(python_executable, python_executable, script_path, *args)
    # subprocess.Popen([python_executable, script_path] + args)
    # sys.exit()

def pip_install(logger, modules):
    args = shlex.split("-m pip install --user --trusted-host mirrors.cambricon.com -i http://mirrors.cambricon.com/pypi/web/simple {modules}".format(modules=modules))
    try:
        subprocess.check_call([sys.executable, *args])
    except subprocess.CalledProcessError as e:
        logger.fatal("automatically install modules failed, `python {args}` failed".format(args=" ".join(args)))
        sys.exit(-1)

def try_pip_install_with_restart_once(logger, modules):
    logger.error("Some python modules ({modules}) should be installed".format(modules=modules))
    logger.error('''You need to run \033[1;33m"pip3 install --user --trusted-host mirrors.cambricon.com -i http://mirrors.cambricon.com/pypi/web/simple {0}"\033[0m'''.format(modules))
    if os.environ.get('_PY_MODULE_RESTARTED', None) is None: # to avoid infinite exec loop...
        logger.warning('try to install python modules automatically')
        pip_install(logger, modules)
        restart_script(enable_env='_PY_MODULE_RESTARTED')
    else:
        logger.error('module list to be installed need update')
        sys.exit(-1)
