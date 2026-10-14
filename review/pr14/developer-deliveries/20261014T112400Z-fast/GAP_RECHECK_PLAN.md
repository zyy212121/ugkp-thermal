# FAST3 剩余缺口与有界复验计划

PR14_REVIEW_ONLY。本表是申请和源码/既有数据分析，不是 GPU 完成声明。保持第二轮原 gap/finding IDs、原输入、原通过标准和失败/阻断记录。所有新执行结果必须由审查者填入其单一新 reviewer-deliveries 目录。

## 先完成正式修复的三条证明

1. `GPU-F006` / `wet-ablation-material`：原 wet heat/evap 输入正式复验、未选择材料零相变的热/力/film forcing/完整库存、受影响 wet restart/continue。
2. 同一 phase-volume 分支的合法非零相变控制：有效 material、真实 nonzero phase mass、原 FP64 运算及完整热/力/几何成功路径不变。
3. `TEST-R2-S1/S2` / `developer-formal-recheck`：正式 CUDA fixture 的 host/device 所有权、真实 Co accepted timeline、16 失败组合、原成功、terminal 和全语义审计；保留 S208 历史范围。

这三条通过也不自动关闭下面独立缺口。

## 1. 实际 active cold 完整组

关联 `OBS-R2-COLD16`、`FSH-cold-profile`、`R2-B-16-activegroups`。固定原输入/reader/22 个 input-output state 的核查与确切构造配方见 `evidence/cold_fixture_triage.json`；这是既有数据检查，没有新增 GPU run。

源码/不可变输入复核发现：S214 由 S211 已提交的 wall-bound 记录复制，随后 `inputs/r2_groups_explicit.py` 把其改成 `pStuck=0`、face=-1；所复制速度已经为零。16 粒子在所有保存输出仍为 mobile/no-contact，cold2D ages 全零。原每粒子无条件 profile inverse 观察器不能直接代表 active cold 的 caloric consumer。保留原 strict profile/pT 红及原输入，不能为 stale inactive profile 改生产。

优先从原 authoritative ASCII SCHEMA3 mobile t=0 入射记录建立 16 粒子的有效小型 fixture。该记录位于第一轮 `inputs/native/resident/CHT-coldWall2D-INPUT2/0/gpuResidentStrictParticles.dat`，其实际 puz=-5、z=5e-7 会覆盖 `0/Us` 的 -10；必须保留实际记录的速度、直径、质量推导、温度、RNG 来历和初始 empty cold 状态。按真实 frontend reader 的列定义复制 16 行并核对 cell ID/orig_id 与位置，不用只支持二进制的 reader 误读 ASCII；真实 cell/face 映射逐项核对，让实际接触流程产生 active 状态。原记录列为 x/y/z、ux/uy/uz、T、theta、d、cell、status、rng、orig_id、stuck、stuckFace、area，质量来自原 particleProperties 的 parcelMass=1.9054628487089897e-4。建议只把 x/y 按原 4×4 cells 布置、cell=i、orig_id=i+1，其余实际输入不变；第一写时须见真实 wall-bound/contact 初始化，后续 age 必须证明动态推进。不要再把已接触记录的标志单独清零。可用完整一致的 bound snapshot 作另一控制，但 contact/profile/age/velocity/addressing 必须相容。

记录至少两个完整热计算组的真实 active witness、组/粒子数量、接触与 consumer 路径；对 active CHT 采用原 double mean+double carry 与 pT 的 1e-9 K 标准，FSH 按其真实 double schema，另记 inactive mobile 的有效 caloric 状态。补全 paired gas/particle/wall 各阶段最终权重和热库存。现有四粒子 initialization fixture 不是这个问题的来源，也不代替动态整组证明。

## 2. Full-vs-restart 首差

关联 `OBS-R2-RESTART`、`R2-C-restart-full-bitparity`。复用 formal-vs-802 同 checkpoint 全字节相同的历史结论，不重跑跨版全套。

使用原 thermal-full 的 8 cells / 2 windows / 100 microsteps，及原 checkpoint1 的后 50 microsteps 恢复。已保存首处差异在 exchange-2.csv 第 2 行，window2/micro1/packet0/face16/stage1，energy/conductive 分别为 `-0.0014999964008330952` 与 `-0.0014999964008330941`；该首 microstep 的 packets 0/2/4/6 不同，之后 396 个匹配，终态气体只剩 8 cells 的 Px 不同。上述是既有数据定位，尚不是完整因果证明。

优先观察 fresh allocation 与 warm owner 的 `Tgas`：源码显示前者清零，uploadState 恢复 conservative state，后者保留缓存，而 currentCo recovery 将 Tgas 用作温度反演 seed。此路径在 802 已存在；它是待证伪/确认的假设，不是获准的生产修改。

用不改原输出的只读观察器，依执行顺序记录第一个 currentCo 前后 Tgas 的 hex、conservative state、volume、nextDt/lastAcceptedDt、selected dt/wall 输入、后续 T/p recovery 和第一个 heat/momentum packet。若恢复 Tgas 等值，明确推翻该解释并继续下一处 gradient/flux。不要把生产 Tgas 强制清零，不改 checkpoint 语义或放宽 bitwise 标准。

## 3. 严格动量账与全部阶段库存

关联 `OBS-R2-P`、`SST-RK-budgets`、`resident-full-budgets`。保留原 cumulative budget 严格 FAIL、原 absMass=1e-14/relMass=1e-10 比较规则及其适用量，不靠重设容差关闭。

先利用已有实际最终 face flux、实际 `ledgerDt`、packet 和 checkpoint 做算术归因。新离线重算证据位于 `evidence/strict_momentum_reanalysis/`：它只针对 SST-SSPRK2、SST-SSPRK3、dry-ablation 的 gas bulk x momentum；不代表新 GPU 运行、不覆盖 film/particle/global P。脚本应把实际浮点值精确转换后高精度累加，并按原执行顺序复现 double accumulator，区分真实通量闭合与大数累加后的 audit 舍入。真实 boundary kind 和 packet persistent face ID 不是同一套索引，不可混用。

在此基础上补尚缺 gross SST production/dissipation、全部三 resident 应用的真实 stage 权重、最终限幅 flux、全部源与 M/E/P 库存。只对缺失链条做小型 GPU 观察；已通过的 net K/omega、M/E、阶段权重不重复。即使算术归因成功，也分别报告物理 flux 结果和原 strict cumulative audit 结果，不把诊断解释伪装成生产修复或整账 PASS。

## 4. Cold-start Js / history-Co 新输入

关联 `cold-start-Js`、`R2-NOT-RUN-cold-start-Js`。需要独立首步 oracle，覆盖无历史状态的首次调用及合法流场/输入改变后的实际 Js、Co、选定 dt 和 consumer。复用 cache/operator fresh-data 单元，但不要将其等同于该物理/调度证明。保存输入、初始 state、前处理数组、实际归约和最终消费值；不新增入场数值限制。已有 host-shim scenario（`applications/CHMT/tests/test_final_gas_consumers.py`）只能作输入配方参考：实际 fresh/unpublished 0.013 → 更新 0.021 → copy 失败后 retry 0.034，随后真实 stage 发布 0.034，再供给 0.055；确认 Co 使用已有已发布 lagged mass、下一 actual stage 消费 0.055 的原策略。采用独立 host/device owner 与显式同步，记录实际 phi=-area*Js，不能人为宣布 profile ready 或修改 K/omega。至少两组真正改变流场/dt 的输入对照 fresh reference；不要因为 lagged 行为就改生产策略。

## 5. Mapper、材料和其它资源范围

- `CHT-radiation-mapper`：补 empty 与 multiple-selected face，保留已证明的 enabled/disabled DO、single/mixed 及完整固体边界热账。empty 的真实成功/无工作语义按原实现核对，不为了测试新增拒绝规则。
- `wet-ablation-material`：除首要 wet/evap/有效非零 phase 外，仍需实际 pore/material→geometry 及未覆盖库存；不能由 dry M/E 推断全部 melting/pore 模型。
- `host-sanitizers`：codec 之外的 wire、mapper、resource-owner 路径。测试编译器/真实 host 路径与实际 GPU 工作分别标记，不能用 shim 代替。
- `allocation-fault-lifecycle`：已通过 SST22/Co OOM、copy/retry/destruction 直接复用；补尚缺 free/sync/host allocation 与适用并发路径。只做安全、可控、可清理的真实错误，不危险 OOB，不新增生产资源上限/门禁。

旧 `SST22-owner-graph`、`moving-GCL`、`chem-native-full-retry-owner`、`mixed-wall-coupler`、`native-FP64-clock-lineage` 的限定关闭状态由原证据支持；不重新当作未执行。需要新依赖才扩展，`developer-formal-recheck` 仍随本轮三条证明和其他未结项追加状态。

## 6. 两项原 300 s timeout 的追加预算

原 `CHMT-chemistry-full-initcheck`（正式 Ns10 reacting 全输入）和 `EXTRA-wet-evaporation-S209-D204`（诊断 evaporation）均保留原命令、日志、输入哈希、300 s 超时及部分输出；部分 checker 的未达终点失败也保留。

审查者先在新 TEST_PLAN 写明同原输入/终点、机器与实际命令、预计 sanitizer/evap 成本和一个有限外部观察预算，建议 Ns10 full initcheck 为 1800 s，原 evaporation 输入为 900 s，单 GPU 顺序执行；这些是有限观察预算，不是完成保证。运行记录注明这是新尝试，不覆盖原尝试。Ns10 须等原完整输入的最终 initcheck 汇总；control initcheck 0 不替代。evap 使用本轮正式修复、观察 accepted timeline/进展及真实资源错误，不叠 D204，不缩物理输入或终点换绿。

超出新增预算仍未完成，保留 BLOCKED/INCONCLUSIVE 与可诊断进度，说明继续所需条件；不自动开启 PRODUCTION 或无限循环。外部时间预算不是求解器停算契约。
