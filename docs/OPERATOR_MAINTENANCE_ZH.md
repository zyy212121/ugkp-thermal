# 双库公共算子维护

gas 负责流场；FSH 增加有限接触壁面；CHT 增加双向固体耦合。应用选择由用户承担，应用内部优化必须覆盖其合法功能，不能以“没有有限接触壁面”等特殊场景为优化前提，也不允许加入测试案例得到的经验参数。

## 维护所有权与构建

两库可独立构建。`tools/managed_mirrors.json` 明确公共算子以 ugkp-thermal 为上游、gas 应用以 gpu-riemann-gkp-main 为上游；清单登记共享和库内独有文件。正常检查只读，发现镜像漂移或未登记输入会失败。单独 checkout 校验本地库存；两库并排时同时比较内容。

```sh
python3 tools/managed_mirrors.py
python3 tools/particle_field_contract.py
python3 tools/operator_policy_contract_test.py
```

修改上游后，使用 `managed_mirrors.py --sync` 按白名单更新下游；新增 common 输入需显式 `--register common/文件名`。工具不会删除未登记文件。库存排除已知的 wmake 平台输出、CUDA build/build_logs、生成的 lnInclude 和固定后台归档路径；不按任意文件后缀全局放行，生成物不能登记为受管源码。Allwmake、直接 CUDA 构建脚本和算子验证入口均执行源码检查。

## 公共协议与字段

气相 Euler/RK、边界完成、图捕获流程由 `GpuGasAdvance.cuh` 维护；`GpuGasHostPolicy.cuh` 提供时间、权重精度和壁能账本能力。packing 由 `GpuMobilePackingHost.cuh` 维护；thermal 的 bin/split host 流程由 `GpuParticleDirectoryHost.cuh` 维护。ToolB1 的 launch bundle、测量、选优及事件错误处理已有共同主体。诊断初始化和阶段计时由 `GpuDevelopmentProbeInit.cuh` / `GpuDevelopmentAdvanceProbe.cuh` 维护，FSH/CHT 采样收集由 `GpuDevelopmentThermalProbeSample.cuh` 维护；精度转换和 gas 专有采样字段保留明确的编译期边界。

入口宏必须满足 `GpuOperatorContract.cuh` 的必填、类型和有效组合检查。`GpuLaunchOptions.cuh` 为矩搬运/恢复提供不同类型的具名选项；历史 bool 包装只作为兼容入口。`GpuPressureFlatLayout.cuh` 命名 FP32 压力段和缓存槽，保持原布局。

`ParticleFieldManifest.json` 生成 payload copy/swap 注册和 thermal schema7 wire 顺序。改字段后显式运行 `particle_field_contract.py --generate`，再执行 `--cpu --output /源码目录外/路径` 检查真实声明、分配、释放、搬运和旧格式 golden。gas 的 scratch 与持久 payload 分开建模；thermal schema7 的 golden 不代表 gas 独有磁盘格式。

## CUDA 调度与验证

这里的 CUDA 启动网格指 kernel 的线程块数量，与 blockMesh 物理网格无关。普通粒子和追踪 launch 由实际 kernel 占用率与容量决定，不受 L1/L2 规约选择影响。规约块 B3 仍由用户设置，任务 tile 仍按粒子总量、SM 数、规约占用率和 B3 推导。显式 L2 不会按估计盈利自动关闭；`gpuCsrLevel auto` 则复用现有占用阈值，在首步及每个 `gpuCsrHeavyReductionAutoInterval` 周期检查 L1/L2 归约选择（默认100步，必须为正）。auto不是研究层级，研究接口仍关闭。

auto 的检查周期、占用统计、严格大于阈值的判定、决策发布和任务就绪保障由 `common/GpuAutomaticCsrSchedule.cuh` 维护。三个应用的 `runToolB3` 只适配目录和硬件策略参数。状态字段统一在 `GpuAutomaticCsrScheduleFields.inl`：`csrMaximumOccupancy` 专用于统计；任务数、重载单元数和队列游标各有独立存储，不能相互借用。CHT 的壁能账本仍位于状态首地址。

公共任务生产者每次都先使 `csrTasksReady` 失效，即使此前处于L1；成功排入当前目录的 count/scan/materialize 后才置为就绪，并记录目录类型。auto 检查更新阈值后重新准备任务；非检查步使用目录生产者刚准备好的任务，遇到未就绪或目录类型变化则补建。返回错误时调用方必须终止推进，不能继续消费任务。任务在同一CUDA流中排在消费者前面，未增加每步主机同步或重复扫描。

同一状态转换及实际碰撞池消费者夹具覆盖 gasUGKP64、FSH64、CHT64、CHT32，运行 `python3 -B -m pytest -q tests/test_shared_auto_schedule.py tests/test_gas_auto_pool_cuda.py tests/test_thermal_auto_pool_cuda.py`。统计判据仍是 `GpuHardwareReductionTile.cuh` 的硬件公式，未添加工况参数；效率评估必须计入检查周期上的任务准备成本。

任务目录保留 count、全局 exclusive scan、materialize 的单元顺序；计数清零并入 count，总任务数发布并入 materialize。`csrHeavyTaskCount` 包含所有非空单元的任务，`csrHeavyCellCount` 只记录需要多个任务的单元。Poisson 抽样、theta 读取时机和 RNG 提交由同一公共默认策略维护；概率与非 Poisson cutoff 语义保留。

```sh
python3 tools/validate_shared_operators.py --output /源码目录外/native-validation
python3 -m pytest -q tests/test_named_launch_options.py tests/test_tool_b1_protocol.py tests/test_source_contracts.py
```

效率评估使用同输入、同二进制身份、固定配对数量和均衡顺序；B2/B3 固定为 64 的效率测量与其它块大小的正确性测试分开。按用户最新要求，k2 配对加速比中位数接近 1 且低档收益瓶颈明确可接受，不要求最小值超过 1；不为跨线强行增加优化。性能记录必须绑定实际源码和二进制 SHA。

有限接触粒子/壁面局部账本与既有气粒换热缺口分别评价；不能将“缺口未恶化”写成严格能量闭合通过。成功路径的字段闭包也不等价于穷尽全部资源耗尽或损坏 restart 输入。


Poisson 访问策略由 GpuCollisionPoolParticle.cuh 统一，接受后的 RNG 提交延后，拒绝仍立即提交；不再设置应用/层级的 late-theta 开关。GpuCollisionPoolSplitParticle.cuh 是共同的兼容包装。运行 tests/test_uniform_pool_policy.py 可检查公共归属与读写顺序。S2/L2 的 legacy lightBlocksPerSm 字段使用所选规约路径的驻留能力，不能解释为固定 L1 核占用率。本轮最终性能范围按用户要求收敛为 k1/k2。

## 当前公共接口与镜像边界

thermal固定L0/L1/L2与auto选择模式由 `common/GpuSchedulingConfiguration.H` 维护，流体库研究接口保持独立。这个配置头不允许机械跨库覆盖。自动判定保留已有周期目录归约和主机回读，不能解释成零成本。生产双库冻结期间只做 standalone 库存检查；不要对生产运行 `--sync`。

本文件是当前维护入口；早期性能结果是历史阶段记录。当前收尾验证见 [自动调度及清理验证](development/auto-cleanup-20261004/README.md)，前一阶段的六项合并见 [算子合并验证](development/operator-unification-20261004/README.md)。

本轮 gas auto 正确性修复及三应用公共流程验证见 [auto 正确性与公共协议](development/gas-auto-correctness-20261004/README.md)。
