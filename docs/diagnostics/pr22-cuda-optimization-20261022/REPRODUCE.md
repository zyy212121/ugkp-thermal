# 复现PR22效率优化

代码提交：`cee8dcaec4d761708b4f0ca59e48bce97a62ec7b`。工作区：`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018`。运行前从新目录复制case-inputs，避免覆盖证据中的已完成案例。性能脚本拒绝覆盖既有case目录。

## 正式构建

```bash
cd /home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018
source /opt/openfoam10/etc/bashrc
runtime="$PWD/work/pr22-native-cuda-20261021/cuda-opt-20261022/final-runtime"
mkdir -p "$runtime"/{bin,lib,tmp,cuda-cache}
export FOAM_USER_APPBIN="$runtime/bin"
GAS_UGKP_FMAD=0 UGKWP_CUDA_ARCH=sm_89 NVCC_APPEND_FLAGS='-Ofc=min -Xptxas=-v' bash tool/gasUGKP/build_private_backend.sh
```

源环境脚本在本机可能返回1，但已设置OpenFOAM变量；上述交互命令不启用set -e。完整实际构建命令和退出状态见final-production-build.json/log。Ofc=min不意味着允许fast-math或更改精度；本轮保持--fmad=false和原FP64。

## 原生和冻结参考

```bash
GAS_UGKP_CUDA_STACK_BYTES=131072 NVCC_APPEND_FLAGS=-Ofc=min python3 validation/cuda/wall_coupling/reproduce.py --output NEW_QUALIFICATION_DIRECTORY --cuda /usr/local/cuda --arch sm_89 --sanitize --math-reference work/pr22-native-cuda-20261021/reference/pr22-math-reference-20261021T0941Z
```

证据含两个sequence-owner*.bin与独立冻结final-A.csv及哈希；reproduce.py还期待原CPU参考包的既有目录布局。fixture头、ABI和比较器来自本次代码提交。资格构建发生于诊断目录元数据修复之前，但四个生产数值文件与最终源逐字节一致；官方最终完整构建与整应用检查发生于修复之后。

## 整应用

```bash
GAS_UGKP_CUDA_STACK_BYTES=131072 FOAM_SIGFPE=false python3 validation/cuda/wall_coupling/run_case.py --case NEW_MIXED_CASE --runtime "$runtime" --output NEW_AUDIT_DIRECTORY
# 三种工具分别使用新Euler案例目录
GAS_UGKP_CUDA_STACK_BYTES=131072 FOAM_SIGFPE=false python3 validation/cuda/wall_coupling/run_case.py --case NEW_EULER_CASE --runtime "$runtime" --output NEW_EULER_AUDIT --sanitize memcheck
python3 validation/cuda/wall_coupling/audit_euler.py --case NEW_EULER_CASE --input work/pr22-native-cuda-20261021/mixed-euler-input --audit NEW_EULER_AUDIT --dt 2e-7 --output NEW_BUDGET.json
```

完整20步逐事件计时、旧/新相同输入验证、缓存反例测试、Nsight采集和CPU/GPU差异计算见归档scripts/。其中gpu_opt_lib.py保存原始绝对路径和探索输出名，重跑探索时须换新输出根目录；无需重新进行已撤回的慢方案。

旧版本来源2f4a66f，支持补丁保存在old-build-support.patch。固定相同31输入，FMA关闭、相同线程栈和sm89，同时保留mid历史runtime对照及两端min构建。独占GPU串行计时，统计3–19步；插桩诊断及sanitizer不纳入主性能数据。

## 证据内容

archive.json列出完整ZIP、逐文件SHA256；如文件分片，则按parts顺序连接后校验ZIP SHA256，再解压。包括全部成功、失败、撤回／超时日志，独立冻结回放输入与CPU表，两组原生CPU/GPU完整4000行表，整应用输入／终态／面源数组，测试基线复测，源码patch及各实验关键头文件。可执行文件／对象／静态库、swap和缓存未归档，二进制以SHA256定位。
