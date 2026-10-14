import shutil
from r2_runner import *
src=D/'inputs/segv-observer.cpp';shutil.copy2(pathlib.Path(__file__).parent/'r2_signal.cpp',src);lib=ROOT/'work/segv-observer.so'
t=run('BUILD-HOST-SEGV-OBSERVER',cmd('g++','-g','-shared','-fPIC',src,'-o',lib),[10,14,26],expected='External SIGSEGV observer only; no CUDA replacement.',inputs=[src],kind='BUILD',variant='diagnostic-host-observer')
if not t['exit_code']:
 b=ROOT/'work/fixtures/chemistry/chemistry_trial'
 t=run('CHEM-OWNER-HOST-SEGV-OBSERVER',cmd('env','LD_PRELOAD='+str(lib),b),[10,14,26],expected='Locate host crash on same original binary.',binary=b,inputs=[lib],variant='diagnostic-host-observer',kind='DIAGNOSTIC')
 print((D/t['log']).read_text())
