from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[3]
def test_particle_node_wall_flux_balance(tmp_path):
    exe=tmp_path/'node-wall-source'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(ROOT/'validation/pr35/node_wall_sources.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
