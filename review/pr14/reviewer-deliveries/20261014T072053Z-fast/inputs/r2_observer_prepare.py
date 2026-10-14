from r2_runner import *
import shutil,difflib
source=D/'inputs/actual-chemistry-volume-observer-v2.cpp';shutil.copy2(P/'r2_volume_observer_v2.cpp',source)
old=D/'inputs/actual-chemistry-volume-observer.cpp'
(D/'patches/S206-CUDA13-kernel-handle-observer.patch').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),source.read_text().splitlines(True),fromfile='a/actual-chemistry-volume-observer.cpp',tofile='b/actual-chemistry-volume-observer-v2.cpp')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS206诊断支持修正：保留原ACTUAL-MOVING-CHEM-VOLUMES零记录FAIL及全部输出完全相同的结果。nm确认本机CUDA13生成代码实际调用__cudaLaunchKernel(cudaKernel_t,...), 而原观察器只挂接cudaLaunchKernel(constvoid*,...)。使用已安装crt/device_functions.h真实签名和cuda.h/cuKernelGetName记录实际handle对应kernel名字/参数。新增只读8cell化学volume D2H与RKblend参数观察，不改变参数/算式；额外同步/IO不纳入效率。原移动chem输入/步数不变、各版CSV/checkpoint/geometry应全bytes相同。观察器编译900s，实际案例300s。\n')
lib=ROOT/'work/actual-chemistry-volume-observer-v2.so'
run('BUILD-ACTUAL-CHEM-VOLUME-OBSERVER-S206',cmd('g++','-std=c++17','-O2','-shared','-fPIC','-I/usr/local/cuda/include',source,'-L/usr/local/cuda/lib64','-lcudart','-L/usr/lib/wsl/lib','-lcuda','-ldl','-o',lib),[8,10,14,26,29],900,'External actualCUDA13kernelhandle ABI observer; noproduction changes.',inputs=[source],kind='BUILD',variant='diagnostic-volume-S206')
