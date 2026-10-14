from r2_runner import *
b=ROOT/'work/fixtures/chemistry/chemistry_trial'
run('CHEM-OWNER-HOST-BACKTRACE',cmd('gdb','-batch','-ex','run','-ex','bt','--args',b),[10,14,26],expected='Diagnose original fixture hostSIGSEGV without replacing GPU kernels.',binary=b,kind='DIAGNOSTIC',variant='diagnostic-gdb')
