# 2026-10-04：gas auto 任务状态正确性修复

本轮按 c38d3e2 审查意见修复 gasUGKP auto，并按用户后续要求统一三个应用的完整 auto 协议。只修改 WSL 测试库；生产 thermal、流体库和 E 盘保持不变。GitHub 上传父提交为 `c38d3e2b3fe9ac7214432f8afb2454f520341007`，现有案例保留。

## 确认与复现

生产调用链是目录/任务准备 → runToolB3 → clearPoissonThermalPoolKernel → launchCsrHeavyPoolReduction → 公共 worker/finalizer。旧检查借用 csrHeavyCellCount 存最大单元占用，破坏后续 finalizer 的重载单元总数。旧 L1 状态还会使前面的任务准备被跳过，启用标志更新后没有重建描述符。

修复前新增消费者回归18项全部按预期失败。4单元33/32/32/31、阈值32：已有L2时真实5任务/1重载单元被改为5任务/33重载单元；旧L1时变成0任务/33。测试在发现无效元数据后终止，没有故意运行越界 GPU 访问。

## 修复

- `common/GpuAutomaticCsrScheduleFields.inl` 统一模式、间隔、推进计数和相关存储角色。最大占用、任务数、重载单元数和队列游标互不借用。gas和FSH各增加4字节设备标量，CHT沿用自己的独立标量；声明、分配和释放闭合，CHT壁能状态仍保持首地址。
- `common/GpuAutomaticCsrSchedule.cuh` 统一检查周期、最大占用统计、严格大于硬件阈值的判定、决策发布、关闭时清计数及启用时准备任务。三个 runToolB3 只适配目录/执行参数。
- 公共任务生产者每次先使任务就绪状态失效，包括旧L1路径；成功排入 count/scan/materialize 后才发布就绪和目录类型。auto更新阈值后重建，非检查步复用已有任务，遇到未就绪或目录类型改变则补建。保留原CUDA流顺序，不增加每步主机同步或重复扫描。
- 最大占用核也共用，gas的base-only与热应用的split-no-injection明确适配。固定L0/L1/L2、检查间隔、原硬件阈值、归约线程和物理公式不改。字段提取所需测试/基准夹具只更新声明读取，未改变原预期值或数值算法。

## 验证范围

gas独立修复阶段18/18通过，随后公共协议最终回归 **75/75通过**：3项公共归属检查及72项实际CUDA消费者检查，覆盖gasUGKP64、FSH64、CHT64、CHT32 × 三目录 × 已有L2/由L1启用 × 32/64/128线程。共用一个转换/消费者夹具，每项执行9个非稳态阶段、显式L2和3次目录切换：首次启用、后续检查、检查间隔、占用下降、空目录、再次启用、总粒子数改变引起tile改变。

每次启用后检查任务总数、重载单元数、目录类型、连续范围和完整覆盖，再实际执行生产 Poisson 碰撞池 worker/finalizer。与实际L1消费者比较选中颗粒、每粒子RNG，独立CPU计算质量、三分量动量、能量与粒径矩，并核对发布计数。使用生产同样的普通主机描述符；finalizer列表只按4单元分配，避免超大测试缓冲掩盖越界。

GPU Compute Sanitizer memcheck **24/24通过，0错误**：四配置、三目录及两种初始状态，均执行完整消费者序列。最终四CUDA后端对象按原sm_89/O3/FMA策略成功编译。受管文件清单233项、四配置字段生命周期和160项策略编译检查通过。CHT32矩对照容差2e-5，三个FP64配置2e-12；颗粒选择及RNG逐项一致。

完整项目 pytest 前后对照：修复前源码快照 **528通过、175失败、25错误、38跳过**；修复后测试库 **608通过、170失败、25错误、38跳过**。没有新增失败或错误测试名称。既有失败与错误逐项名称见 [修复后结果清单](suite-after-summary.json)，前后差异见 [对照清单](suite-comparison.json)，完整输出另有gzip归档。基线是独立源码/工具/测试快照，没有Git元数据及历史交付文档，因此消失的资源存在性失败不能当作本轮数值改进。整个仓库仍不是全绿，不能据此做整体工程无条件验收。

生产库源码/链接冻结复核：thermal503项、流体244项均未变化。原c38算子与配置改善保留；此前auto检查的标志验证不足已在原报告入口更正。本轮修复不追溯改写原始日志。

## 性能边界与复现

本轮没有效率计时，不声称auto非稳态收益或整体性能不变。修复保留原定检查频率，非检查步正常复用准备好的任务；检查得到新阈值后仍启用L2时保证重建。FSH此前只在激活时补建，现在与gas/CHT统一。后续若评估auto收益，应同时包含目录检查与任务准备成本。

运行 `python3 -B -m pytest -q tests/test_shared_auto_schedule.py tests/test_gas_auto_pool_cuda.py tests/test_thermal_auto_pool_cuda.py`。CUDA夹具使用本机sm_89（可由UGKWP_CUDA_ARCH覆盖），512MiB编译栈，优化等级保持O3。内存检查命令为 `compute-sanitizer --tool memcheck --error-exitcode 99 <各配置测试二进制> <目录0/1/2> <初始L2 0/1> 32`。完整 pytest 使用已加载的OpenFOAM环境与 `UGKP_MANAGED_MIRROR_ROOT` 指向本次测试库，测试产物写在源码外。

源码补丁 [changes.patch](changes.patch)，输入白名单及哈希 [changed-inputs.json](changed-inputs.json)，四配置72组消费者输出在 actual-consumers，GPU内存检查在 memcheck。一次复测被WSL重启中断，未完整产生结果的日志不作为通过或失败证据；之后已复核生产源码冻结并重跑。该修复随后按用户授权直接提交热力学GitHub main，不同步生产库。
