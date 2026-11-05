"""Reproducible host/CUDA execution of the actual shared collision pool chain."""
import argparse,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def body(path,signature):
 text=(ROOT/path).read_text();start=text.index(signature);brace=text.index('{',start);depth=1;end=brace+1
 while depth:
  depth+=(text[end]=='{')-(text[end]=='}');end+=1
 return text[start:end]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda',action='store_true');p.add_argument('--before-header',type=Path);p.add_argument('--expect-loss',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 fragments=[body('common/operators/snapshotParticleGasCouplingStateKernel.cuh','__device__ unsigned long long mixSeed'),body('common/wall/GpuColdWall2DDevice.cuh','__device__ void clearColdWall2DParticleState'),body('common/thermal/GpuResidentStrict.cu','__device__ void finalizeOneThermalizedMobileParticle')]
 (out/'owner_functions.inc').write_text('\n'.join(fragments)+'\n')
 if a.before_header:shutil.copyfile(a.before_header,out/'GpuThermalParticleFinalization.cuh')
 exe=out/'pool-release';compiler='/usr/local/cuda/bin/nvcc' if a.cuda else 'g++'
 flags=['-arch=sm_89','--fmad=false'] if a.cuda else ['-x','c++','-fsanitize=address,undefined','-fno-omit-frame-pointer']
 cmd=[compiler,'-std=c++17','-O1',*flags,'-I'+str(out),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/wall'),str(ROOT/'validation/pr36/frozen_pool_release.cu'),'-o',str(exe)]
 (out/'build-command.txt').write_text(' '.join(cmd)+'\n')
 with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out/'run.log').open('w') as log:result=subprocess.run([exe],stdout=log,stderr=subprocess.STDOUT)
 print((out/'run.log').read_text(),end='');assert result.returncode==(1 if a.expect_loss else 0),result.returncode
if __name__=='__main__':main()
