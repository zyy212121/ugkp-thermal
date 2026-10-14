from r3_runner import *
for name in ['SharedGasTrialPolicy.cuh','GpuGasAdvance.cuh']:
 for p in ROOT.rglob(name):
  if 'work' in p.relative_to(ROOT).parts or 'review' in p.relative_to(ROOT).parts:continue
  lines=p.read_text().splitlines();print('PATH',p)
  start,end=(260,340) if name.startswith('Shared') else (650,805)
  print('\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1,min(end,len(lines)))))
