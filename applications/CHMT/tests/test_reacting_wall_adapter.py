from pathlib import Path
import subprocess
APP=Path(__file__).resolve().parents[1]
def test_reacting_wall_adapter(tmp_path):
    exe=tmp_path/'reacting_wall_adapter'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(APP/'tests/test_reacting_wall_adapter.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
