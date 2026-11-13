"""Resident geometry must select the populated-host material accuracy branch."""
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_material_geometry_explicit_count(tmp_path):
    root=SOURCE_ROOT
    exe=tmp_path/"material_geometry_count"
    command=["g++","-std=c++17","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror","-pedantic","-Wno-misleading-indentation",
             "-I"+str(root/"applications/CHMT"),"-I"+str(root/"common"),
             str(root/"validation/pr42/material_geometry_count.cpp"),
             str(root/"applications/CHMT/mesh/Geometry.C"),"-o",str(exe)]
    built=subprocess.run(command,capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr
    assert "17 host/resident parity cases" in run.stdout
