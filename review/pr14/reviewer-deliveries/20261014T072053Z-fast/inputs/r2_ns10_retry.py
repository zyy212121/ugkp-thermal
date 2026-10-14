import time,json
from r2_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nNs10编译预算变更：第一次完整原生编译因900s外部deadline终止（多大CUDA translation unit并行期间），原完整log/command保留，非数值缺陷。完成其他编译及Ns2非profiling样本后，Ns10单独重复同源/同flags/同入口，外部预算1800s；不改变求解器或数值case预算300s。\n')
while not (D/'artifacts/timing/summary.json').exists():time.sleep(5)
t=run('BUILD-CHMT10-SERIAL-RETRY',cmd('env','UGKWP_GAS_SPECIES=10',f'CHMT_BUILD={ROOT}/work/build/CHMT10',f'FOAM_USER_APPBIN={ROOT}/work/install/ns10/bin','bash',ROOT/'applications/CHMT/Allwmake'),[10,24,25,26],1800,'Samefixedsource/flags genuine complete nativeNs10build, isolatedserial retry afterdeadline.',inputs=[ROOT/'applications/CHMT/Allwmake'],kind='BUILD')
if not t['exit_code']:
 from r2_native import chmt
 chmt('chemistry-full');chmt('chemistry-control')
