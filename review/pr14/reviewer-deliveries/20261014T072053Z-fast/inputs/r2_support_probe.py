from r2_runner import *
for p in [pathlib.Path('/usr/local/cuda/include/cuda_runtime_api.h'),pathlib.Path('/usr/local/cuda/include/crt/host_runtime.h')]:
 s=p.read_text();lines=s.splitlines();print('FILE',p)
 for i,x in enumerate(lines):
  if '__cudaLaunchKernel' in x:print('\n'.join(f'{j+1}:{lines[j]}' for j in range(max(0,i-6),min(len(lines),i+10))))
print('COLDcoupler',*[str(p.relative_to(ROOT)) for p in ROOT.rglob('GpuSolidThermalCoupler.C')],sep='\n')
for p in [D/'artifacts/extra/SST-SSPRK2/exchange-1.csv',D/'artifacts/extra/SST-SSPRK3/exchange-1.csv',OLD/'inputs/native/chmt/thermal-full/constant/chmtProperties',ROOT/'applications/CHMT/io/Output.C']:
 if p.exists():print('FILE',p,'\n',p.read_text()[:8000])
