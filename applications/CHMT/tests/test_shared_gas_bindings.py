from pathlib import Path
import subprocess
APP = Path(__file__).resolve().parents[1]

def test_shared_model_binding_and_density_conversion(tmp_path):
    target=tmp_path/'binding'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
                    '-I'+str(APP),'-I'+str(APP.parents[1]/'common'),
                    str(APP/'tests/test_shared_gas_bindings.cpp'),'-o',str(target)],check=True)
    subprocess.run([str(target)],check=True)

def test_exact_species_storage_and_transaction(tmp_path):
    target=tmp_path/'mixture'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
                    '-I'+str(APP),'-I'+str(APP.parents[1]/'common'),
                    str(APP/'tests/test_shared_mixture_storage.cpp'),'-o',str(target)],check=True)
    subprocess.run([str(target)],check=True)
