from r2_runner import *
p=pathlib.Path('/usr/local/cuda/include/cuda_runtime_api.h');lines=p.read_text().splitlines()
for i,x in enumerate(lines):
 if 'cudaKernelGetName' in x:print('\n'.join(f'{j+1}:{lines[j]}' for j in range(max(0,i-12),min(len(lines),i+4))))
print('old cases',*[p.name for p in (OLD/'inputs/native/chmt').iterdir()],sep='\n')
