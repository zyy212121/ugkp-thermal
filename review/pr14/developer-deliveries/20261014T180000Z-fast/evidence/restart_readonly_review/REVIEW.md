# PR14 FAST4 重启温度初值差异只读审查

## 结论

本项没有足够证据要求修改成熟 FP64 热力学恢复、warm start 或原迭代容差。建议保持生产源码不变，不为逐位一致而清零温度缓存、统一冷启动、增加无意义迭代或重写历史时间积分。

严格 full-versus-restart 位一致性仍失败，原记录不能改为通过。不过，这不构成最近两轮修复必须新增热恢复代码变更的理由：相同恢复源码及旧版本重启对照已证明该限定行为不是最近修复新增。严格位一致是否继续作为本次整体验收的必要条件，应由用户决定；本审查不代替该决定，也不改变任何标准。

## 固定身份和审查范围

- 生产提交：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`
- 生产 tree：`5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`
- 证据固定提交：`0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668`
- 主要证据目录：`review/pr14/reviewer-deliveries/20261014T120103Z-fast/`
- 本项仅阅读源码、配对已有真实 GPU 原始日志及进行离线 binary64 算术。没有运行新的 GPU 计算，没有修改仓库源码、测试、门槛、README 或 Conversation。

每个实际读取文件的 Git blob、SHA256、字节数及与固定提交的逐字节比较，见 `metrics.json` 的 `fixed_references`。`recompute_metrics.py` 可从已有仓库证据重算全部指标，不作新的验收判定。

## 恢复路径与可以证明的范围

`common/operators/computeSstFaceFluxKernel.cuh:1561–1569` 从保守量计算内能，以当前 `s.Tgas[c]` 作为温度初值；成功后在 `1611` 回写温度和压力。

`common/gasTransport/MixtureThermo.H:464–466` 使用有效区间内的既有初值，fresh 的零温度在本例转入区间中点。`469–496` 计算原能量残差及 Newton 修正阈值，满足条件即返回当前已求值温度；`503–507` 执行下一次 Newton 或区间回退。因此，从不同初值在两个相邻 binary64 温度结束，并不说明存在精度或物理错误。

D303 记录了实际 currentCo 前后状态与后续 flux，支持 seed 差异经热恢复传播这一解释。但它没有记录热反演每次迭代、能量残差、修正量或具体返回分支，不能断言已经直接捕获“warm 恰在第几次由哪一行返回”，也不能声称已经排除所有其他因果因素。

## 实物差异与原数值尺度

独立配对完整 window2 的 400 条 post-currentCo 单元记录，覆盖 8 个单元和 50 个微步：

- 最大温差：`1.1368683772161603e-13 K`，约 600 K 处的 1 ULP，相对差约 `1.8948e-16`
- 最大压力差：`2.9103830456733704e-11 Pa`
- 全部 post-currentCo 的质量、总能量、体积差均为零
- 后续最大 x 动量差：`1.455185662723943e-16 kg·m/s`
- 50 条 selected-dt 记录完全相同；没有观察到时间步选择分叉
- 216/1800 条最终 face-flux 记录、4/400 条 paired 记录存在位差

原 `GasModel.H:166–174` 的热反演阈值为：能量 `1e-10 + 1e-10×|目标内能|`，Newton 温度修正 `1e-9 + 1e-10×|T|`。原 thermal 输入没有覆盖这些设置，`SharedGasDeviceStorage.cuh:384` 将实际 gas controls 传入该路径。约 600 K 时原温度修正阈值为 `6.099992807669355e-8 K`；所见温差约为其 `1.86e-6` 倍。该比值只是原数值尺度比较，修正阈值不是严格温度真误差上界，也不是新的 restart 通过标准。

使用捕获的 cell0 保守量和原 linearCp 输入离线重算：warm 温度对应的主机 binary64 能量残差为 0；fresh 温度对应残差约 `−5.820766091346741e-11`，修正约 `−7.833561092807994e-14 K`，远低于原能量和修正阈值。这只是独立主机算术佐证，不冒充 GPU 的实际 FMA、迭代路径或退出记录。

原 numerical continuation checker 在 `applications/CHMT/verification/coupled_runtime/check_case.py:230–243` 对终态 gas/material 使用 `1e-10 + 5e-12×max(|a|,|b|)`，并另行要求初始 checkpoint 及同步摘要等条件。它与证据中的全文件严格位一致性是两种不同检查。本审查没有重跑该 checker，尤其不宣称 dry checker 已证明 wet 用例通过。

## wet 差异与基线边界

`artifacts/wet-full-restart-first-difference.json` 保留了实际重启完成后 27 个终态语义字段差异。首次交换能量和导热差为 `3.469446951953614e-18 J`；气体单元 x 动量差最大约 `1.455202603382888e-16 kg·m/s`。支持冲量差为 `5.684341886080802e-14 kg·m/s`。较小净 exchangeMomentum 项的相对差可达 `3.15e-10`，所以不能把所有终态差异概括为“只有 1 ULP”；1 ULP 是热恢复的首个温差。

该 wet 证据与 D303 thermal 证据属于不同用例。不能将 thermal 的 seed 因果解释直接当作 wet 的排他根因证明，更不能用这里的小重启差异关闭 wet 的完整机械库存/反力缺口。

三个核心文件 `computeSstFaceFluxKernel.cuh`、`MixtureThermo.H`、`GasModel.H` 在 `802a11e7`、`4b5c480b`、`9d5be6ef` 三个生产版本之间 blob 完全相同。R2 的 `artifacts/chmt/old802-vs-formal-restart.json` 记录同一真实 checkpoint/input 下 10 个输出文件逐字节一致；同轮 `formal-restart-terminal-comparison.json` 已保留 full-versus-restart 严格失败。

因此可说“既有于已核实的 802 版本，非最近两轮修复新增”。最早 PR 基线 `d6110c02` 的这些文件不同，现有材料不足以把这一结论扩大为“整个 PR 之前已实物证明完全同样”。

## 最小后续范围

保持源码不变本身不需要再安排一次大规模 GPU 回归。如果 FAST4 其他修复不触及此路径，可沿用以上固定证据，不通过修改数值路径换取严格位比较变绿。

若必须把 seed 根因证明到具体返回分支，最小缺口是一次隔离、只读的真实 GPU 首受影响恢复观察：同一 checkpoint 保守量和实际 thermo controls，分别保留 warm/fresh 初值，记录每次残差、capacity、修正量、迭代数及退出分支，再核验原输出未受观察影响。若要证明 wet 同因，应在原 wet 用例取得对应首步观察。不能用主机重算或 thermal 用例代替这两项证明。

在现有证据下，新增缓存序列化、统一 seed 或改变 Newton 停止条件都超出必要修复，且可能引入格式兼容、效率和成熟 FP64 路径改变。这里没有提出或执行这些改动。

## 复算

运行 `python recompute_metrics.py --repo /workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014 --output metrics.json`。

输出仅为描述性指标，`wet_strict_bitparity` 保留为 false；任何已有失败、标准和范围都未被自动变更。交付文件自身摘要见 `SHA256SUMS`。
