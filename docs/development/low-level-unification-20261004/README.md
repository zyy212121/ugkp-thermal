# 底层算子归并：一致性与守恒验证记录（2026-10-04）

基线为 `fa71f4777aefb6457e662c42c72db73ec3778343`。本轮仅修改
`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`。
当前完成源码整理、CPU 行为检查、CUDA 编译、机器码比较及本轮相关 GPU 一致性与守恒回归。
109 次 GPU 程序调用及 7 个补充物理回归程序通过；验证范围见下文，不等同于整个历史测试集或完整案例验收。
本轮没有效率计时，没有 k1/k2/k8 运行，没有修改生产双库或 E 盘。

## 实施范围

1. `common/GpuBlockComponentReduction.cuh` 单点维护分量归约。gas 的 pruned tree 和热应用的 full tree 由编译期策略选择；原有 lane 判据、第二级 offset、同步、求和顺序及精度保持。
2. 清零主体与采样准备接入公共算子。gas 的 dt 签名、仅多任务单元发布概率，以及线程 0 在单元边界检查前发布诊断计数的顺序保持。接触索引追加函数移入独立头，gas 不引入接触字段依赖。
3. `common/GpuThermalLaunchOccupancy.cuh` 共用 FSH/CHT 主机流程。保留各自真实核选择、共享内存字节数、硬件查询及错误分支。CHT 冷壁网格通过应用适配设置，仍在几何打印及最终 `syncDeviceState` 前完成。
4. 删除 gas 无调用点的旧 gather 发射器和私有任务操作；删除 FSH/CHT 的旧分段 gather、CHT 旧 index 核和无 survivor 目录的旧 payload 分支；移除失去消费者的两个 CHT 私有索引线程常量。有效夹具迁至当前 survivor producer/consumer 与字段提交路径。

没有改动物理系数、截断常数、随机数位数、应用层级接口、auto 判据、归约线程或硬件阈值。
gas 的 baseOnly 和热应用的独立接触、精度策略均保留。

## 可达性与删除边界

gas 旧发射器在仓库生产源中没有调用点；冻结流体库仍持有它及其测试引用，所以本轮只清理测试热库。
`common/GpuCellLocalGather.cuh` 继续保留，避免后续手动镜像时误删流体库依赖；它已不被当前热库生产源包含。

FSH/CHT 的正常 cell-local 路径，正容量与通过验证的 occupancy 产生正工作网格，post-transport 使用 exact survivor directory。旧 gather 只出现在 `!exactSurvivorDirectory` 分支。
零容量且没有颗粒源时，外层跳过颗粒链；零容量且源使 `particlesMayBePresent` 成立时，初始 `preBaseDirectoryReady=0`，pre-transport 走 full bin，零网格 count 发射的错误在 scan、auto、任务消费及 compaction 之前返回。
此已有边界行为保留，本轮没有增加零容量新算法。CPU 模型执行了实际 host bin 函数的错误顺序；实际 CUDA 零网格断言已在三个热应用消费者程序中执行通过。

`thermal_workers` 轻量夹具现在按当前全局 payload 核检查字段闭合、既定 survivor 顺序和精确壁面发布；完整 backend 夹具继续覆盖 FSH fused 搬运、CHT 独立 payload、L1/L2、压力及 commit。
`l2_goal` 的 filtered/all-live 两项迁至真实 survivor count/scatter 与当前 payload 消费者，保留过滤、空单元、覆盖、顺序和输出尾部哨兵检查。
`payload_commit` 的字段换指针、条件冷壁字段和壁面检查仍保留。

## 已执行的验证

- 36 项相关 CPU 行为/源码约束检查，以及 1 项更新后的 gas 发射配置检查通过，共 37 项。清零用旧、新实际函数逐字段逐位比较，并检查缓存与诊断时序；occupancy 用实际旧、新 host 代码比较配置、核身份、共享字节、API 调用顺序、错误及同步快照。CPU 桩不模拟 CUDA 调度。
- 237 项受管清单检查、gas64/FSH64/CHT32/CHT64 字段生命周期闭合，以及 160 项入口策略编译检查通过。清单为测试库单库检查，**未进行双库镜像同步**。
- gas64、FSH64、CHT64、CHT32 四个完整 backend 对象编译通过。CUDA 13.1，`sm_89`，C++17，O3，PIC；gas/FSH fmad=true，CHT fmad=false；CHT32 保持原有命名隔离与精度头。
- 全部保留 GPU 函数的指令文本、寄存器操作数、分支 offset、原始机器字、调度字及资源记录完全相同；没有将地址差异或 symbol 改名过滤成“相同”。资源比较包含寄存器、静态 shared、stack、local、constant。动态 shared 与几何/调用顺序另由实际 host 配置及源码范围对照确认。

| 配置 | 基线函数数 | 保留函数数 | 删除旧 gather 函数 | 改变函数 | 新增函数 |
|---|---:|---:|---:|---:|---:|
| CHT32 | 159 | 146 | 13 | 0 | 0 |
| CHT64 | 155 | 142 | 13 | 0 | 0 |
| FSH64 | 141 | 133 | 8 | 0 | 0 |
| gasUGKP64 | 131 | 123 | 8 | 0 | 0 |

合计 586 → 544 个函数，删除 42 个经可达性核查的旧 gather 实例，其余全部相同。
第一阶段 1/2 单独比较时，586 个函数也全部相同。

构建中出现过两次工具链内部崩溃：stage12 CHT64 nvcc 崩溃和最终 CHT32 消费者主机编译器在系统数学头中的 ICE；同配置重试成功，原始失败日志与成功记录并存。
此外，离线编译暴露已有 CHT 压力夹具仍传整数 0，已改为现有 `FlatPressureSegment::full`，没有修改生产压力算法。
本轮未执行整个历史 pytest 集，不将以前的失败记录或以前的 GPU 成功冒充本轮结果。

## 已执行的 GPU 一致性与守恒回归

17 个程序已执行通过，总计 109 次调用：4 个归约逐位程序，3 个完整载荷程序，3 个字段提交程序，3 个当前 S1/L2 热消费者程序，以及 4 个真实 compaction→目录→auto→碰撞池程序。
归约两种策略分别对照自己的旧实现：每配置 150 项比较，覆盖 1/3/4/7/8 分量与 32/64/128/256/512/1024 线程、抵消、动态范围、正负零和 subnormal。实际公开线程配置 32/64/128/256 包含在内；不要求两种不同归约树互相逐位一致。
生产链程序另有 96 组目录/调度/实际池消费检查，保留质量、动量、能量和任务状态的原有断言。

以下第一条只校验源和产物哈希，不调用 GPU；第二条执行 GPU 正确性。本轮已在用户授权后执行第二条，不计时：

```bash
cd /home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test
python3 tools/run_prebuilt_low_level.py --bundle /home/lss/ugkp-low-level-unification-20261004
python3 tools/run_prebuilt_low_level.py --bundle /home/lss/ugkp-low-level-unification-20261004 --run
```

第二条的 109 次调用全部通过，包括 600 项各自旧、新归约逐位比较、三配置热应用载荷/commit/S1-L2/压力一致性以及 96 次真实目录→auto→碰撞池检查。池质量、三分量动量、能量逐单元与独立 CPU 参考和实际 L1 消费结果对照，随机数与选中颗粒状态一致。记录见 [GPU 调用结果](evidence/gpu-correctness/results.json)。

另执行四配置 `collision_behavior.py`，覆盖非 Poisson/Poisson、L1/L2、32/64/128/256 线程、接触元数据与共享采样准备，并验证拒绝采样不读 theta；三个热配置 `contact_conservation.py` 检查逐壁面颗粒焓变化＋壁面能量账本守恒（冷热两面、传热开/关、四种线程数），合计 48 个能量收支断言通过。结果见 [物理回归](evidence/physical-correctness/results.json) 和 [汇总](evidence/gpu-validation-summary.json)。

首次执行暴露旧测试预期：compact 压力更新后错误要求源数组也清零。压力主体与基线完全相同；夹具已改为分别验证源数组不变、compact 数组正确，以及原路径更新后的源数组，未修改压力生产代码。补充碰撞夹具也仍引用已移除的旧策略宏，现改用实际公共 Poisson 模板默认策略。原失败、旧夹具与根因记录保存在 [诊断证据](evidence/gpu-regression-diagnosis/)。
如源码改变，哈希检查会拒绝使用旧产物。可用 `tools/low_level_regression.py --build-only --integration --out <目录>` 重建 13 个基础/热消费者程序，并用原 `test_production_directory_cuda.py` 入口重建和执行四配置生产链。

目前**没有必须补做的效率测试**：保留的 GPU 生成代码及资源相同，正常有效路径的核次数、时序、同步和搬运没有改变，occupancy 配置结果逐项相同。若后续修改热路径或其调度，再对受影响消费者重做产物比较，并按差异决定小范围配对计时。

提交前清理了测试参考头及 gather 夹具的行尾空格，保持所有代码 token 不变；四配置归约程序重新编译并逐位对照通过，5 项受影响 gather 行为检查也重新通过。最终执行清单与验证输入哈希已更新。

## 证据与发布状态

精简证据在 [evidence](evidence/)；完整对象、SASS、二进制、基线快照和补丁在 `/home/lss/ugkp-low-level-unification-20261004`。
[离线机器码比较](evidence/machine-comparison.json)、[源码范围](evidence/source-scope.json)、[生产冻结检查](evidence/production-source-guard.json) 和 [执行清单](evidence/deferred-run.json) 可独立核对。
可重现比较命令：

```bash
python3 tools/compare_cuda_artifacts.py \
  --before /home/lss/ugkp-low-level-unification-20261004/before-build \
  --after /home/lss/ugkp-low-level-unification-20261004/validated-final/build \
  --out /home/lss/ugkp-low-level-unification-20261004/recomparison \
  --require-unchanged
```

生产 thermal 的 503 个受保护源/工具输入及流体库的 244 个输入哈希均保持；E 盘未操作。
用户已授权在一致性与守恒回归通过后上传 main。本记录与源码随同一提交发布；发布回执及远端提交号保存于外部证据目录 `github-upload-state.json`。

没有执行完整生产案例长时间积分、整个历史 pytest 集或效率计时；本次通过结论限于上述与归并改动相关的实际消费者、数值对照及守恒收支检查。
