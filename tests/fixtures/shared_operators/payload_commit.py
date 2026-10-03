import os
from pathlib import Path
import sys,re,subprocess
root=Path(sys.argv[1]);out=Path(sys.argv[2]);app=sys.argv[3];bits=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=True)
fixture=(root/f'tests/fixtures/thermal_workers/gather_{app}.cu.in').read_text()
fixture=fixture.replace('int nCells,','int *compactCountDevice,*particleCountDevice;\n int nCells,')
fixture=fixture.replace('const int capacity=3*n+16;', 'const int capacity=3*n+16;')
fixture=fixture.replace('alloc(s,1);','alloc(s,1);alloc(s->compactCountDevice,1);alloc(s->particleCountDevice,1);')
code=(root/'common/GpuCellLocalThermalFields.cuh').read_text()+'\n#define GPU_PARTICLE_EXTRA_FIELDS CellLocalThermalExtraFields<'+('true, true' if app=='CHT' else 'false, false')+'>\n'+(root/'common/GpuCellLocalPrimary.cuh').read_text()
code+='\ntemplate<class T>__device__ T clampRange(T x,T a,T b){return x<a?a:x>b?b:x;}\n'
if (root/'common/GpuParticleBufferCommit.cuh').exists():code+='\n#include "operators/gatherSelectedParticlesKernel.cuh"\n#include "GpuParticleBufferCommit.cuh"\n'
else:
 t=(root/f'applications/{app}/{"gpu" if app=="CHT" else "private_backend"}/GpuResidentStrict.cu').read_text();code+='\n#include "operators/swapParticlePointerDevice.cuh"\n'
 for name in ['gatherSelectedParticlesKernel','commitCellLocalParticleBuffersKernel','commitSelectedParticleBuffersKernel']:
  a=t.index('__global__ void '+name);i=t.index('{',a)+1;depth=1
  while depth:depth+=(t[i]=='{')-(t[i]=='}');i+=1
  code+='\n'+t[a:i]+'\n'
fixture=fixture.replace('@PARTICLE_COPY_HOOK@',code).replace('@REAL_TYPE@','float' if bits==32 else 'double')
start=fixture.index(' if(segmented) gatherThermalSegmentedParticlesKernel')
end=fixture.index(' auto e=cudaDeviceSynchronize()',start)
fixture=fixture[:start]+" if(cudaDeviceSynchronize()!=cudaSuccess)exit(12);for(int dst=0;dst<(int)expected.size();dst++)s->sortedParticleIndex[dst]=expected[dst];*s->compactCountDevice=expected.size();gatherSelectedParticlesKernel<<<32,128>>>(s);\n"+fixture[end:]

pairs=re.findall(r'cudaFree\(s->(p\w+)\);cudaFree\(s->(compactP\w+)\);',fixture)
pairs=[(x,y) for x,y in pairs if app!='FSH' or x!='pContactAge']
checks=''
for x,y in pairs:checks+=f'auto old_{x}=s->{x};auto old_{y}=s->{y};\n'
checks+='commitSelectedParticleBuffersKernel<<<1,32>>>(s);if(cudaDeviceSynchronize()!=cudaSuccess)exit(7);if(*s->particleCountDevice!=(int)expected.size())exit(8);\n'
for x,y in pairs:
 flag='(flags&2)' if 'Cold2D' in x else '(flags&1)' if 'Cold' in x else 'true'
 checks+=f'if(s->{x}!=({flag}?old_{y}:old_{x})||s->{y}!=({flag}?old_{x}:old_{y})){{printf("swap fail {x}\\n");exit(9);}}\n'
checks+='commitCellLocalParticleBuffersKernel<<<1,32>>>(s);if(cudaDeviceSynchronize()!=cudaSuccess)exit(10);\n'
for x,y in pairs:checks+=f'if(s->{x}!=old_{x}||s->{y}!=old_{y})exit(11);\n'
checks+='swapParticleBuffersDevice(*s);\n'
for x,y in pairs:
 flag='(flags&2)' if 'Cold2D' in x else '(flags&1)' if 'Cold' in x else 'true'
 checks+=f'if(s->{x}!=({flag}?old_{y}:old_{x})||s->{y}!=({flag}?old_{x}:old_{y}))exit(13);\n'
checks+='swapParticleBuffersDevice(*s);\n'
for x,y in pairs:checks+=f'if(s->{x}!=old_{x}||s->{y}!=old_{y})exit(14);\n'
checks+='cudaFree(s->compactCountDevice);cudaFree(s->particleCountDevice);\n'
fixture=fixture.replace(' cudaFree(s->px);',checks+' cudaFree(s->px);',1)
f=out/'payload_commit.cu';f.write_text(fixture);exe=out/'payload_commit';cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','--fmad=false','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'-I'+str(root/'common'),'-I'+str(root/'gpu/thermal'),str(f),'-o',str(exe)]
with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
if q.returncode:print((out/'build.log').read_text()[-5000:]);sys.exit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(app,bits,len(pairs),q.stdout,q.stderr);sys.exit(q.returncode)
