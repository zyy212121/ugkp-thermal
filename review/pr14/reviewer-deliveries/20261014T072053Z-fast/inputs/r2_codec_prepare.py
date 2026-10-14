from r2_runner import *
src=D/'inputs/checkpoint_actual_reader_S207.cpp';b=ROOT/'work/fixtures/checkpoint-ASan-S207-2'
run('BUILD-HOST-CHECKPOINT-ASAN-S207-2',cmd('g++','-std=c++17','-g','-O1','-fno-omit-frame-pointer','-fsanitize=address,undefined','-DUGKWP_GAS_SPECIES=2','-I'+str(ROOT/'applications/CHMT'),src,ROOT/'applications/CHMT/restart/Checkpoint.C','-o',b),[5,6,8,9,14,19,22,26,28],900,'Actualunchanged productioncodec withadditional realinventory/ledger fieldprintf; hostASan/UBSan.',inputs=[src,ROOT/'applications/CHMT/restart/Checkpoint.C'],kind='BUILD')
