from r2_runner import *
for p in pathlib.Path('/usr/local/cuda/include').rglob('*.h'):
 lines=p.read_text(errors='replace').splitlines()
 for i,x in enumerate(lines):
  if '__cudaLaunchKernel(' in x or '__cudaLaunchKernel_helper' in x:
   print('FILE',p);print('\n'.join(f'{j+1}:{lines[j]}' for j in range(max(0,i-5),min(len(lines),i+14))))
