from r2_runner import *
b=ROOT/'work/fixtures/chemistry/chemistry_trial'
run('CHEM-OWNER-CUDA-GDB',cmd('cuda-gdb','-batch','-ex','run','-ex','bt','-ex','info registers','-ex','x/10i $pc-20','--args',b),[10,14,26],expected='Real hostCUDA binary crash diagnosis; no mock.',binary=b,kind='DIAGNOSTIC',variant='diagnostic-cuda-gdb')
run('CHEM-OWNER-ADDR2LINE',cmd('addr2line','-Cfipe',b,'0x5dac8','0x5e741','0x5eb56','0x3fc98'),[10,14,26],expected='Resolve actual native crash stack symbols.',binary=b,kind='ANALYSIS')
print((D/'logs/CHEM-OWNER-CUDA-GDB.log').read_text())
