"""Serial adapter for the production gasUGKP parcel path.

Uses the production atomic Poisson pool path including RNG and sampling.
The caller configures collisional pressure/mobile packing off and owns the gas
step. Thermal wall contact is outside this gasUGKP adapter.
"""
from pathlib import Path


def fragment(root):
    root = Path(root)
    backend = (root/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    relaxation = backend[backend.index('#include "GpuParticleGasHeatRelaxation.cuh"'):backend.index('#include "../../../common/gasTransport/MixtureParticleCoupling.cuh"')]
    heat_factor = backend[backend.index('__device__ double particleHeatFactorDevice'):backend.index('#include "operators/clearSolidCell.cuh"')]
    return (Path(__file__).with_suffix('.hpp').read_text()
            .replace('/* PRODUCTION_HEAT_FACTOR */', heat_factor)
            .replace('/* PRODUCTION_RELAXATION */', relaxation))


def selftest(root, directory, sanitizers=False):
    """Build/run the forced-collision and recycling probe without pytest/CUDA."""
    import importlib.util
    import os
    import subprocess
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('steady_reference_build', root/'validation/steady_cpu/build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(directory/'gas_base')
    source = (directory/'gas_base.cpp').read_text().split('int main(')[0]
    helpers = r'''
void initialise(DeviceState&s){Input a;a.cells=2;a.local=0;a.perturbation=0;initialise(s,a);}
void ck(bool value,const char*message){if(!value){std::fprintf(stderr,"%s\n",message);std::abort();}}
void near(double a,double b,const char*message){ck(std::abs(a-b)<1e-10*std::max(1.,std::abs(b)),message);}
'''
    source += helpers + fragment(root) + Path(__file__).with_name('steady_cpu_particles_probe.cpp').read_text()
    cpp, executable = directory/'particles_probe.cpp', directory/'particles_probe'
    cpp.write_text(source)
    flags = ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitizers else ['-O2']
    subprocess.run(['g++','-std=c++17',*flags,*['-I'+str(path) for path in (root, root/'common',root/'common/gasNumerics',root/'applications/gasUGKP/private_backend',Path(__file__).parent)],str(cpp),'-o',str(executable)],check=True)
    return subprocess.run([str(executable)],check=True,text=True,capture_output=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',required=True,type=Path,help='build/output directory')
    parser.add_argument('--sanitizers',action='store_true')
    args=parser.parse_args()
    print(selftest(Path(__file__).resolve().parents[3],args.selftest,args.sanitizers).stdout,end='')
