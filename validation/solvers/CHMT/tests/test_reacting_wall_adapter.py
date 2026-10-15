from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
APP=(SOURCE_ROOT / 'applications/CHMT')
def test_reacting_wall_adapter(tmp_path):
    exe=tmp_path/'reacting_wall_adapter'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(APP.parents[1]/'common'),str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_reacting_wall_adapter.cpp')),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
