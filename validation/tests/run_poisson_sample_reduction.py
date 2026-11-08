from pathlib import Path
import argparse,subprocess
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--arch",default="sm_89")
parser.add_argument("--build-only",action="store_true")
args=parser.parse_args();root=Path(__file__).resolve().parents[2];args.output.mkdir(parents=True,exist_ok=True)
def function(text,needle):
    a=text.index(needle);a=text.rfind("\n",0,a)+1;begin=text.index("{",a);depth=1;end=begin+1
    while depth:
        depth+=(text[end]=="{")-(text[end]=="}");end+=1
    return text[a:end]+"\n"
common=root/"common"
rng=function((common/"operators/snapshotParticleGasCouplingStateKernel.cuh").read_text(),"unsigned long long mixSeed")
rng+=(common/"operators/uniform01Device.cuh").read_text()
rng+=function((common/"operators/normalDevice.cuh").read_text(),"void normalPairDevice")
(args.output/"sample_rng.inc").write_text(rng)
binary=args.output/"poisson-sample-unit"
subprocess.run(["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","--fmad=false","-arch="+args.arch,
    "-I"+str(common),"-I"+str(args.output),str(Path(__file__).with_name("poisson_sample_reduction.cu")),"-o",str(binary)],check=True)
if not args.build_only:subprocess.run([str(binary)],check=True)
