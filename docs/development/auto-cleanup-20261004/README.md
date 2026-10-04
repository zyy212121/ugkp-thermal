# 2026-10-04：恢复 auto 与算子收尾清理

> 后续正确性复核发现 c38d3e2 的 gas auto 占用检查会覆盖重载单元计数，并遗漏从 L1 启用后的任务准备。下述原有4组测试只验证自动标志，未覆盖切档后消费任务，不能据此判定 c38d3e2 的完整 auto 路径验收通过。算子差分与配置验证记录仍保留为历史证据；修复与实际消费者回归见 [gas auto 正确性修复](../gas-auto-correctness-20261004/README.md)。

本轮只修改 WSL 测试库 `/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`，随后按用户授权直接提交热力学 GitHub main。生产 thermal、流体库及 E 盘未修改。上传父提交 `f03e21539a1df32f05a3cd3ee25afa95ed2d9f67`；测试库不含案例，提交时保留 GitHub 所有现有案例，尤其保留 MSS7_twoPhase_dense 的 auto 配置。

## 接口与阈值

固定算子档位仍为 L0/L1/L2；`gpuCsrLevel auto` 是已有 L1/L2 归约路径的动态选择模式，不是研究档位。E/S/T 和 `gpuResearchVariant` 仍拒绝。显式 L2 不会被自动关闭。

auto 复用已有后端判断，未修改 runToolB3、阈值公式、归约线程数或驻留优化架构。阈值为 `B3 * max(1, ceil(N/(B3*SM*residentBlocksPerSM)))`，最大单元颗粒数严格大于阈值时启用重载分段。N 是当前目录的实际颗粒总数，合并目录包含基础与注入颗粒。没有案例拟合参数。

`gpuCsrHeavyReductionAutoInterval` 仅对 auto 有效，必须为正，默认100；首次颗粒推进检查，然后每个间隔检查。它不同于 Courant 更新间隔。周期检查包含目录占用归约及主机回读，不能描述为零成本，也未在本轮量化非稳态收益。迁移工具保留旧 auto 和配置间隔；非法接口在写文件前拒绝。

## 三项清理

- 有序压力遍历复用无排序路径的面通量累积代数。累积保持局部变量，末尾写入共享数组及单元增量，保持原发布时序。
- coldWall1D 两种精度的外层输入、材料、气相热导及接触准备抽为 `GpuColdWall1DInputs.inl`。精度分支及 FP32 稳定求解策略保留。
- 删除 gas 未被生产路径调用的两个旧原子压力核定义。生产路径仍使用原有缓存及有序压力核。

本轮不合并物理上不同的边界/精度/算法策略，不改变物理系数、截断、时间精度和数值公式。

## 验证

- 配置解析、迁移及源码归属 **42/42 通过**：固定三档，auto 默认/正值/非法间隔，研究接口拒绝与迁移拒绝前不写入。额外读取 GitHub 父版本 MSS7_twoPhase_dense 的实际调度文件，原配置成功解析为 auto、1000步检查；无需改写案例。
- 实际 CUDA 差分及自动调度 **4/4 通过**：gas64、FSH64、CHT64、CHT32；完整和合并目录低→高→低及空目录，阈值相等使用L1，检查间隔保持上次选择，显式L2不变。压力面累积与发布、冷壁外层温度/焓/壁面账本/固化状态对照冻结旧实现；冷壁覆盖有限接触及沉积、气相换热开关。
- 相关守恒及归约行为 **12/12 通过**：四种后端压力平衡和碰撞轻/重分段，gas 无排序压力，FSH/CHT 两种精度的颗粒焓与壁面账本守恒。
- 四个最终 CUDA 后端对象成功编译，使用本机原有 sm_89、O3 和各应用 FMA 配置。编译器曾发生内部崩溃，最终对象编译在提高进程栈软限制后完成；未降低 GPU 优化等级或修改数值代码绕过编译。新增夹具修正了 WSL 统一内存并发主机访问，调度描述符采用生产代码的普通主机内存布局。
- 与父版本对象相比，588 个原有核中 570 个机器码逐位相同，16 个机器码变化，删除2个未调用旧核，无新增核；变化核指令数没有增加，共享内存、栈及局部内存资源不变，gas 三个压力核寄存器由64降至62，其余变化核寄存器不变。详细指令差异见 codegen-details.json。源码合并不能先验保证机器码不变，也不能用指令/资源变化直接推断实际加速比。
- 受管镜像 standalone 库存231项、四种后端字段生命周期闭合与160个入口策略编译检查通过。只检查测试库，未运行跨生产库同步。
- 生产 thermal503、流体244项源码/链接哈希与此前冻结基线相同。

本轮采用针对修改路径的回归，没有重新运行整个仓库历史测试集。上一轮完整测试的既有失败仍见 [六项算子合并记录](../operator-unification-20261004/README.md)，不能把本轮聚焦通过解释为全库全绿。

没有运行本轮效率计时。coldWall 外层提取不改变相关核机器码；压力提取和自动模式的数值行为已验证。auto 的非稳态收益及周期主机回读成本，可在后续应用效率测试中量化；本报告不声称新增加速收益。

## 复现

OpenFOAM 环境先加载 `/opt/openfoam10/etc/bashrc`。设置 `UGKP_MANAGED_MIRROR_ROOT=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`，防止管理工具默认指向生产库。运行 `python3 -B -m pytest -q tests/test_scheduling_configuration_behavior.py tests/test_operator_cleanup_ownership.py tests/test_remaining_operator_ownership.py tests/test_auto_cleanup_cuda.py`。原守恒工具 `tools/validate_shared_operators.py --output <源码外目录>` 可运行完整38项；本轮12项选用相同工具中的压力、碰撞分段与接触守恒夹具。

源码补丁见 [changes.patch](changes.patch)，上传白名单及哈希见 [changed-inputs.json](changed-inputs.json)。README 和两份维护入口文档一并补齐，历史性能记录明确标记为历史。
