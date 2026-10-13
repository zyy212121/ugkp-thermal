# PR #14 独立只读源码审查

日期：2026-10-14。审查者：`/root/source_review`。按 `requesting-code-review/code-reviewer.md` 模板进行；未编译、未执行 GPU/CPU 数值测试、未修改仓库工作树/index/HEAD、未写 GitHub、未派子代理。唯一写入是本仓库外报告。

## 固定范围与覆盖

仓库绝对路径：`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261013T182722Z-fast`。下文 `file:line` 均相对此路径；行号是固定 HEAD 的一基行号，明确写明 base 的引用除外。

- Base：`d6110c02745a8ad726208a426c4bb3a28233c1fd`。
- Head：`802a11e7dfe986488a8bf8305e86424dcc94c9e1`；tree：`db2863158139842f6c8221c0da11ba640f381336`。
- 最后生产源码：`a73b9635510fb09d7baa5339e14d1aa609f0ff87`；production tree：`e343cb9f477dcd86556f7b6595189fa2c9c1c099`。
- 已读 INDEX、PROTOCOL、33 个 topic 的开发记录/验收范围、source-index、相关 file/function sidecar、pending-findings 与下述旧审查；未逐条读取全部历史 artifact。
- 已读初始 `review/pr14/reviewer-deliveries/20261013T182722Z-fast/TEST_PLAN.md` 和 coverage；后续只读检查主代理 chemistry GPU fixture、实际日志、未改断言 checker 与 P002 patch/json，以及operator differential来源、cold2D原输出/P006日志/P007隔离补丁和三种cold2D sanitizer summary。本席位没有执行它们。
- 验证了 Backend、Checkpoint、native SharedGas/BoundaryLayer adapter、trial policy、GpuGasAdvance、codec、chemistry integrator/adapter、cold2D 等十条关键源码 SHA256 与 source-index 相同。

审查期间只读复核 HEAD/tree 均匹配上述固定值，曾见 tracked 差异仅为 `examples/thermal/singleAluminaDrop/0/` 七个输入文件。主代理说明这些为按需 LFS 解码，生产源码未改，输入复制后恢复指针；本席位最后再次读取 git diff --name-only 已为空。本席位已读 `logs/needed-lfs-inputs.json` 的 coldWall/0 七条期望/实际 SHA256 相同；没有将测试输入解码等同于生产源码变更，也没有把运行中的工作树称为完全干净。P002 仅存在隔离诊断副本，固定 HEAD 未包含修复。

本次是按类别/调用链的定向审查，**不是 426 个生产文件逐行审查**：

| 类别 | 本次直接源码覆盖 | 尚未由本审查证明 |
|---|---|---|
| CHMT CUDA / owner | 完整 Backend.cu、Backend.H、GasWindowProgram.H、Buffer.H、SharedGasDeviceStorage.cuh；prepare→launch→D2H→commit/restore | 真 GPU 异步错误、注入真实 allocator/copy/free 故障、所有并发时序 |
| 化学 / 原成功重试 | 完整 GpuGasAdvance.cuh、LocalStiffIntegrator.H、density adapter；ConservativeReactor 的全部 PR diff；完整 native SharedGasTrialPolicy 与 AdvanceGasOnly 入口 | 所有机制/刚性尺度的数值精度；整个 native TU 的逐行覆盖 |
| native 配置 / 生命周期 | 完整 SharedGasStorage/Adapter、BoundaryLayerStorage/Adapter、trial snapshot；releaseState 的相关调用 | 所有销毁分支和 persistent runtime failure 的实机行为 |
| cold1D / cold2D | 完整 1D Device/GasOnly/Advance/Inputs/Publish、ThermalCarry；完整 cold2D 1093 行及相关 native launch 段；paired producer/publisher/settlement | 真 warp 调度、多粒子 atomic 顺序、完整接触/追踪/碰撞耦合和 >256 substep 实机终端行为 |
| restart / 可见消费者 | 完整 CHMT Checkpoint.C、HostState、CoupledOutput；thermal codec 的全部 diff 与关键定义；原生持久字段/producer 使用点 | 跨主机/跨 endian、全部历史 wire 格式、所有 restart 工具完整运行 |
| GPU 测试支持 | differential、gas_auto_pool、pressure、collision、direct_dispatch 的相关真实入口/断言；CHMT check_case；cold carry/mixed fixture | 未运行任何这些测试；mock/host-lane 适配器不算 GPU 验收 |

## 旧证据的适用边界

旧记录均按原 source/dependency/实例化范围继承，不能把“approved”解释为整个固定 HEAD 的 GPU 批准：

| 旧证据 | 能支持的范围与限制 |
|---|---|
| E014-c0eb1b4b9d88b077 | a164 全局分类审查：136 direct、49 scoped、241 lexical，合计 426 生产文件；不是 426 文件逐行。无 GPU。 |
| E014-89628f0f69e95895 | thermal 九个生产路径、3225 dependency/hash 记录、有限 native host/CUDA 编译；不证明完整 app/device link 或运行。 |
| E014-24d1004f94dd2618 | resident 七文件最终 delta、1035 记录，有限实际 gas main/focused CUDA；不能推广为所有 native 路径。 |
| E014-e99a5a8c7d6fa711 | 最后单 header C++14 tag-dispatch codec delta；不覆盖整个 restart/app。 |
| E014-df24bc6578459988 | a73 单 mapper delta，普通成功顺序保持；已披露的 supplied-address `operator[]` 风险没有被本审查重新包装为新缺陷。 |
| E014-3eecf953992eb814 | CW-1 的 stationary bracket 修复；原 multiply/40 bisection 保留。四个 CUDA object 是 modeled wrapper/selected header 编译，无完整真实 owner link/GPU。 |
| E014-4a4e29978ed36f1b | particle-terminal 的十个窄 TU；41 quoted-include 闭包中 cold 依赖曾变更，编译证据与运行证据不可混用。普通 heat/radiation 的严格 carry 入口仍有独立既有边界。 |
| E014-0c5d006124907712 | B5 真实 wall profile/exchange/packet/positivity 源码与窄编译。报告主动把 B6/B7 owner/final-status 留给后续，不能用其同一 stage 的成功结论替代当前终端消费者验证。 |

本次未重新报已修复的 signed-R、DO sweep、parser copy、CW-1、pool publication、resident volume mirror、film boundary、codec C++14、mapper scope 问题。CHMT Buffer::release 的历史 best-effort 行为、native void terminal destroy 的已披露限制，不是本 PR 新引入的 owner 修复失败。

## Strengths

- CHMT 存储由实际 Buffer owner 绑定稳定 device view；model/layout 替换、failed publication rollback 与 rejected owner 分开。native failed-free 路径保留尚未释放指针并阻止后续覆盖；已修复旧 owner finding 的调用链可见。
- CHMT 两个化学半步明确消费 Vold/Vnew；先捕获每半步 audit 再让下一半步覆盖 disposable 设备 audit，commit 统一累计。公开状态和累积 audit 并未依赖一个退出码或预置成功标志。
- cold1D 的 paired gas source 在实际 wrapper 内只执行一次，先冻结物理 mass/geometry；cold2D 保留 top-surface flux 子步法，其 gas ledger 来自实际 profile+carry 发布增量。profile floor 与 double carry 的角色清楚，未把 allowed CHT profile float 扩成其他物理 float。
- CHMT runtime checker 有独立 CSV 守恒、EOS、接触导热量、species/element、restart 和 H/2、dt/2 断言；gas pool fixture 有真实七矩、count、status/RNG 对照。它们的数值断言比单纯进程成功有用。

## Issues

### Critical (Must Fix)

未识别到 P0 级缺陷。

### Important (Should Fix)

### SR-1 — [P1] native refinement 未调用严格 chemistry trial 入口

**位置**：`common/operators/advanceGasChemistryKernel.cuh:140–144`，F014-405af05f5909 / FN014-477ee9937ec4；同文件 SHA256 `0b54549cdbcd52d2799cafa6c7484cde74cf7f256dad859ffa5b852463918a7a`。

`nativeRefinement` 只参与前置旧 status 检查，实际 reactor 无条件调用 public `advanceClosedReactor`。新加入的 `advanceClosedReactorTrial` 在 `common/chemistry/LocalStiffIntegrator.H:254–258` 明确写着 retry-capable owner 必须使用，但没有任何生产 callsite（仅定义和两个测试）。

**实际生产链**：`applications/gasUGKP/private_backend/GpuResidentStrict.cu:5430–5458` → `advanceGasRequestedInterval` → `common/GpuGasAdvance.cuh:760–774` 的 refining=true → 第一个 half launch `:577–578` / 第二个 `:688–689` → adapter 的 public reactor。`SharedGasTrialPolicy.cuh:281–298` 本来检查 native chemistry status，但 public reactor 在 `LocalStiffIntegrator.H:189–192` 耗尽 local maximumSteps 后切换到 terminal continuation，可能仍返回 Success，因此原本应触发的小步重试未发生。

**可触发条件**：全部 backing 有效、有限正质量/体积/温度、真实反应；单个化学半步需要超过 maximumSteps，但外层 requested interval 可用更小子步成功完成。此问题不是 NaN/Inf、不缺存储、不依赖故意越界。

**影响**：原来成功的 native refinement 采用新的 terminal remainder/Newton 路径、trial 分区和工作次数；当误差/非线性求解预算触发时，可提交与原成功方法不同的 species。即使某简单 case 最终误差很小，原成功次序/重试语义已经改变。

**明确 base 依据**：d611 `LocalStiffIntegrator.H:75` 在预算耗尽返回 StepLimit；base adapter `:156–162` 不发布失败结果；base `SharedGasTrialPolicy.cuh:212–227` 将 StepLimit 视为可重试；base `GpuGasAdvance.cuh:739–755` 失败后半步、成功后继续。要求恢复此既有成功方法，**不是新增整步拒绝/rollback/重试门禁**。public/nativeRefinement=false 的终端完成仍是授权行为；不要让其恢复旧数值拒绝。

**最小 real CUDA 复现建议（本审查未运行）**：按 `tests/chemistry/test_gas_rates.py:14–24` 的真实 Ns2 Model，molarMass 两个均 .028，linearCp 系数 [1000,0,0]，element [1,1]，basis [-1,1]，不可逆 S0→S1、A=2。nCells=1、rho=28、momentum=0、rhoSpecies=[28,0]、stageVolume=.001，rhoE 来自 T=700 的 mixtureEnergy。controls.maximumSteps=1，initialStep=maximumStep=1e-6，其余原有效默认。真实分配并把 thermo/mechanism 所有数组指针改为 device 地址；当前 launch `advanceGasChemistryKernel<<<1,32>>>(dState,2e-6,dVolume,true)`，base 使用其 revision-correct 三参数入口。捕获完整 bulk/species、ChemistryStatus 全字段、cellStatus、ChemistryAudit 全字段、周边 canary。预期 base StepLimit 且 species/audit 不发布，当前 terminal 可能 Success/acceptedSteps=2/time=2e-6 并发布。若运行原 requestedDt=4e-6 链，理想低-Co输入下预期 base trial序列 `[4e-6失败,2e-6成功,2e-6成功]`，当前可能 `[4e-6成功]`；须以 GPU 记录确认，不能把这里的静态预测当运行证据。

**已读真实 CUDA 证据**：主代理完成实际分配、device 指针重绑定、原生产 `<<<1,32>>>` kernel、synchronize、D2H 与 canary 检查。本席位只读核对 `inputs/chemistry_retry.cu`（SHA256 `9af888df37cff5a0c552863d8c3b8804003c0b7a9b0a416254be4bf2e83819ba`）、三版本 spec 与 logs、`inputs/check_chemistry_retry.py`，不自行重跑。日志相对本次 delivery 目录：

| 日志 | 实际关键输出 |
|---|---|
| `logs/CHEM-RETRY-base-step-limit.log` | code=10/StepLimit，cellStatus=8，species=[28,0]，audit 保留 acceptedSteps=7/integratedTime=99 sentinel，budget=[0,0] |
| `logs/CHEM-RETRY-pr-step-limit.log` | code=0/Success，cellStatus=0，attemptedSteps=2，audit acceptedSteps=2/integratedTime=2e-6，species=[27.999888000223997,0.00011199977600030318]，实际 audit/budget 发布 |
| `logs/CHEM-RETRY-patch-step-limit.log` | 全部输出 JSON 数值/整数与 base 相同 |
| success-control / ordinary 的 base、PR、patch 日志 | 完整日志字节相同；相应 dt=1e-6/maxSteps=1 与 dt=2e-6/maxSteps=100 均保留原成功输出 |
| public-terminal 的 PR、patch 日志 | 完整日志字节相同，Success、acceptedSteps=2、time=2e-6，终端完成保留 |
| `logs/CHECK-CHEM-RETRY-RED.log` / `logs/CHECK-CHEM-RETRY-P002.log` | 同一 checker 的四控制为 [false,true,true,true] → [true,true,true,true]；失败仅 StepLimit full-field publication；没有改断言 |

这些证明实际 density adapter/kernel 的 StepLimit 路由回归以及该窄路径诊断修复；JSON 精确数值比较/日志相同**不是原始 double bit-pattern 审计**。fixture 调用单个化学 kernel，不执行完整 requested-interval owner 循环，因此上文 `[4失败,2成功,2成功]` 的真实 trial 序列、两半步 owner restore/commit、所有更刚性机制仍需开发者定向验收。

**P002 补丁只读结论与限制**：`patches/P002-native-chemistry-trial-routing.patch`（SHA256 `e538f361f0ecb09f865f4c340e3ab44e8ee48b7cd3f68c2a10c7887f1d267772`；json 的 before 为固定源码 SHA，after 为 `2bf1c0399643fe0af50a08ca8d812470c41974427efea6765f9fb8df9a4dc438`）让 true 走已有 Trial，false 仍走 public，并把 native reactorAvailable=false 且 chemistry code!=InvalidModel 从 InvalidStorage 改成 ChemistryFailure。后一分类影响不止 StepLimit。确切未覆盖例子：`MixtureThermo.H:538` 在缺 `view.species` 时返回 thermo InvalidModel；`LocalStiffIntegrator.H:171–172` 包装成 chemistry TemperatureFailure、evaluated=false，因此该诊断补丁会把真实缺 thermo backing 归到 ChemistryFailure。正式开发者必须结合 thermoCode 和实际存储可用性审计，保留真实执行不可用边界；本报告**不批准 P002 为完整生产修复**，也不要求新增数值拒绝门禁。
**测试漏口**：`tests/chemistry/test_gas_stage_lineage.py:76–84` 的 retry-owner 分支直接测 Trial API，没有走 density adapter/真实 native CUDA launch，因而不能发现此未接线问题。仅测 successful reactor、public exhausted completion 或化学响应/守恒，也不足以判定原成功 trial lineage 保留。

**修复方向**：nativeRefinement=true 选择已经存在的 Trial 入口；false 保留当前 public 入口。正式修复仍由开发者完成，随后用同一 GPU probe、base、ordinary/public 控制和两半步重查。

### SR-2 — [P2] 更新 FieldSchema 后未同步独立 runtime checker

**位置**：`applications/CHMT/restart/Checkpoint.C:594` 与 `applications/CHMT/verification/coupled_runtime/check_case.py:19,52`。

PR 从 PhysicsConfig 和 ModelConfig 的 FieldSchema 尾删除三 fingerprint 项。对固定 source 字符串独立 FNV-1a 重算得到 `11438850135072464462`；d611 字符串为 `15664811910203657810`，仍硬编码在 checker:19。因此新 writer 的每个正确 checkpoint 都会被 checker:52 拒绝，后续实际守恒/restart/refinement 断言根本到不了。state 尾字段顺序并未由此改动。

**复现/影响**：新构建 CHMT 输出任一常规耦合 case，执行原 `check_case.py CASE`，失败为 field schema differs。主代理已报告真实 GPU full/split/resumed/refined thermal/single/diffusion+zeroD 均触发原 checker 失败；只在仓库外诊断适配该常量后，原数值断言通过。这是主代理报告，本审查没有独立打开其完整运行结果或重跑。

**修复方向/边界**：维护测试的当前 wire 支持常量与说明，保留旧失败及诊断版 provenance。不得为修复测试往生产读侧重新加 schema/metadata/precision 门禁，也不得改数值阈值。该问题不表示 GPU 计算或 checkpoint 的库存数值错误。

### Minor (Nice to Have)

未列纯样式或无实际影响的建议。

## 新增只读诊断：CUDA-DIFF-gasUGKP 首败来自过时 frozen 文本

主代理后续实际 GPU 日志 `review/pr14/reviewer-deliveries/20261013T182722Z-fast/logs/CUDA-DIFF-gasUGKP.log:1` 为 `FAIL bits moment/face primitive 0.0064999999999999997 != 1.625`（日志 SHA256 `93bbbab128bd20f47675a8218e9c6c9c252b8d6631fb52cccc5829fa2f95f488`）。本席位未运行 GPU，直接只读检查 generator、生成 CUDA、当前源码、真实 base 和历史。

**结论**：这一首败不是 PR14 生产回归，也不是 float/FP64 差异或 moment 的单位换算；它比较当前 wall-EOS 密度与遗漏固定温度处理的旧密度。同一行为已在 base d611 存在。它是既有 fixture 维护缺口，不增列为 PR 新引入的生产 finding。

**具体路径与字段**：`differential.py:39–46,61–73` 的 `primitives` 同时比较三个 solid moment 标量与六个 face 字段；错误标签把这些合称 moment/face primitive，未输出 i/j。按首个不同数值、输入和遍历顺序静态定位到 i=21、f=21、faceOwner=5、比较 j=3，即 face.rho；此定位是源码推导，不是新增 GPU index 日志。生成 `work/cuda-operators/DIFF-gasUGKP/differential.cu:528–536,832–844` 同样如此。

- Seed c=5 的 rho=1+.125*5=1.625，p=2+.25*5=3.25，Tgas=305；Rgas=1。f=21 为 boundary，kind=21%5=1/slip，TFix=1，wallT=500，normal=(1,0,0)。fixture 的 p/rho/R 与 Tgas 故意不一致，因此不能把中心 Tgas 当成其理想气体 EOS 温度。
- 当前 `common/operators/riemannFacePrimitiveForGradient.cuh:60–85` 投影法向速度后，还按 TFix 发布 T=500 与 rho=p/(R*T)=0.0065；与日志左值吻合。`frozen['gasUGKP:riemannFacePrimitiveForGradient']` 的 slip 分支（生成 CUDA `:176–187`）仅投影速度即 return，保留 centre.rho=1.625、centre.T=305；与右值吻合。两者 rho 字段都是气体密度；差异不来自 V、momRhoP、rhoSolid 或存储精度。
- d611 同一生产文件 `:76–84` 已有完全相同的固定温度及 EOS 公式。base→HEAD 对该函数的 diff 仅增加 nativeRefinement 参数并传给 moving-mesh/boundary helper，未更改该静止 SingleLegacy slip-wall 成功分支。

**冻结来源与 pin**：`baseline.json` 只有20个带 app/name 键的函数字符串，无源 commit、单位或模型/schema 描述；generator `:19,25–27` 原样导入文本并只重命名函数。git --follow 的唯一 file commit 是 `f03e21539a1df32f05a3cd3ee25afa95ed2d9f67`（2026-10-04，Unify remaining thermal operator bodies and retain numerical policies）。该文件在 d611 与 HEAD 都是 blob `e6bb980cd26e7acb1cf77c06b9e68a9e2d22b792`、SHA256 `42e2413e6447060b4e953cd8e6edf7cf4749d2d0f3187017cb3d94a33032017e`；gas face body 字符串 SHA256 `09ab820ab07fbac252381c077a24c4880b8886ddd9d043fa06e202c0527229eb`。随后 `c73591f1096f6245ac5cc96f0a1f04397499b1b0`（Preserve slip-wall thermal gradient primitive）向生产函数增加9行热边界处理而没有更新此 frozen 文本；此 commit 已先于 d611。base 函数文件 SHA256 `6cac757e84caaa19e8ae2bfcfb47a2ed3596ec93d7abce6ecac872bad6da5e78`，HEAD 为 `6108161aba86933385beed51e199c17d64db96420bfa4bb89ade49c3440ab780`。

**最小可解释适配建议（本席位未写）**：在仓库外诊断版 frozen gas face 函数的 kind1 速度投影之后、return 之前，移植 d611:76–84 已存在的 TFix/T/rho 分支，仅把 template GasState 的类型查询适配到该 concrete DeviceState；引用 base 及 c735 的 source pin。保留原 seed、同一512×9双精度 memcmp断言、失败原日志；不要清 TFix、修改 production、放宽容差或用 HEAD 整体重生成 expected。另加 f21 的独立物理检查 rho=3.25/500、T=500、normal velocity=0，以避免把当前输出本身当基线。诊断适配通过仍只关闭这个首败；其他 drag/pressure、periodic、mixture、移动网格以及 FSH/CHT 的 frozen 文本需各自按真实 base/依赖范围核查，不能据此宣称整套 differential 已证明正确。

当前 generator SHA256 `1a8afcdae6fd86be6879ee4e63f9ca22dabb452dc32835e8a376f2145600afa5`；生成 CUDA SHA256 `e30775a7ad70a8aebbf90ab0a6ad0a394867fa7e60dd13c945e1977b43de69f6`。本项仅追加只读证据和适配建议，未改仓库或运行 fixture。
## 新增真实 cold2D 失败：已定位的 producer 缺口（基线亦有）

**影响与归因范围**：实际正常物理输入会在第二步将 pT、64 个 float profile 节点、double carry 发布为 NaN，按生产影响属于 Important/P1。但同一 zero-effusivity producer 在 d611 已存在；目前不将此项新增为“原成功 FP64 路径回归”。本席位只读源码、原始分析 JSON、P006 诊断日志及三种 sanitizer 的 summary，未执行这些程序。该 NaN 来自真实算术消费，不是 MissingStorage；应修正 producer 接线，不能恢复旧 positivity/finite trap 或新增数值接纳门禁。

**可复现输入**：delivery `inputs/native/resident/CHT-coldWall2D-INPUT2`，原版 `work/install/ns2/bin/CHT`；1×1×10 网格、一个 d=.005 m 的粒子、pm=.00019054628487089897 kg、T=2373.15 K，位置 (.02,.02,5e-7)、速度 (0,0,-5)，dt=1e-6 s，共10步。物性和其它输入保留已固定例子；1D 控制只改变模型选择。两模型实际均进入 stuck_state=3、face=9，不是未撞壁的假控制。

| 原版时刻 | 2D 温度 / profile / carry | 1D 控制 |
|---|---|---|
| 1e-6 s | T=2373.1499520863576；64 节点均4191779.5；carry=1.486361866920305e-10 | 相同温度、8 节点4191779.5、相同 carry |
| 2e-6 s | T、64 节点、carry 全NaN；ring0 age=1e-6、frozenArea=1.4463267383218867e-7 | T=2373.1500218609517、8节点4191779.25、carry=.24245608314733094 |
| 1e-5 s | T、64 节点、carry仍全NaN；ring0 age=9e-6 | T=2373.149970086867、8节点4191779.25、carry=.1644389244694987 |

**首次 NaN 的具体调用链**：

- `applications/CHT/gpu/GpuResidentStrict.cu:3856–3873` 分配非空 per-face effusivity backing，并在 pending bit4 时 cudaMemset 为零；`:5903` configure 初始化，`:5944/5976` 将指针发布到 DeviceState。
- `applications/CHT/diluteUgkwpFoam.C:106–111` 默认 `gpuResidentSolidThermalCoupling=false`，实际输入没有打开它；`:669–686` 因此不创建 coupler/调用首次墙温生产。真实固体物性 producer 在 `applications/CHT/thermal/GpuSolidThermalCoupler.C:1608–1632`，按 rho(T)*Cp(T)*kappa(T) 开方，只更新 selected coupled particle-contact patches；`GpuResidentStrict.H:908–930` 也只上传所列 patches。
- `common/wall/GpuColdWall2DDevice.cuh:849–856` 以“数组指针非空”为选择条件，因此选中 face9 的0，不能到达 scalar sqrt fallback。现有 DeviceState 默认 `GpuResidentStrict.cu:302–304` 为1800/710/100。P006 原方程仅添加 printf，真实记录 `effArray=0`、`effScalar=11304.866208850062`、rho=1800/Cp=710/k=100；不要把 example 字典7900/500/15 当作 device 实际物性：当前 stuck-model configure API 不接收这三项，host 没有消费这些 key。
- `GpuColdWall2DDevice.cuh:245/248–260` 首个 wet ring 的 localAge=0、wallEffusivity=0，执行 sqrt(pi*0)/0 得 NaN；wetArea>0 进入真实 wallConductance 分支，NaN 沿 `:478–481` wallPower/nodePower/energy 传播。P006 首条直接打印 age=0、wetArea=1.4463267383218867e-7、wallG=-nan、wallT=473.14999999999998、gasG=.00069338576716127974、oldCarry=1.4863618669203049e-10。首次 scaled gas conductance 是有限非零，不支持把 particleHeatCoefficient 当作首个 NaN 来源；后续 gasG=0 是已污染状态之后的结果。
- `:542–557` paired mean correction 会把局部 NaN 通过 group sum 与统一 correction 传播到8×8全部节点、nextCarry和输出温度。几何公式 `GpuColdWall2DSolidification.H:45–68` 的体积/最大面积/物理质量在该输入都是正有限；未发现此次 NaN 由这些几何除法先产生。kernel `:699–709` 按8 lane一组、四组一warp32，particle0覆盖完整mask；未识别确定的 warp mask/branch 缺陷，不能从一个粒子推断所有lane形态已正确。

**为何 1D 有限不足以关闭壁面参数问题**：`GpuColdWall1DAdvance.inl:106–119` 按 ring 中点计算 firstWetAge。实际 duration=.006646106291838649、peakFraction=.2771202269110757，ring0 firstWetAge=5.848388097241029e-5 s；这十步仅接触9e-6 s，因此1D的 localAge0/localAge1均为0。`GpuWallInterfaceHeat.H:55–59` 在 deltaAge==0 时先返回0，还没消费 effusivity 的除法。该短1D控制证明 profile/carry/gas path 可运行，尚未激活同一墙传导除法。

**d611 证据边界**：base `applications/CHT/gpu/GpuResidentStrict.cu:6311–6316/6546–6549` 同样分配并把 effusivity 清零；base main `:138–143` 同样默认关闭 solid coupling。base `common/wall/GpuColdWall2DDevice.cuh:927–941` 在 wallEffusivity<=0 时 `asm("trap;")`，helper `:149` 也拒绝它。因此静态预计同一正常 disabled-solid 输入在 base 会 trap，而不是给出原成功有限基线。本席位没有运行该 base 输入；不能把静态预测标成已通过/已失败的真实基线执行。旧 gate 的移除不授权用新的数值 gate 遮住这个 producer 缺口。

**P007 隔离补丁只读审阅**：已读 `patches/P007-cold-wall-fallback-producer.patch/json` 和 diagnostic-P007 源码。只在 bit4 未完成初始化时，用现有 scalar rho*Cp*k 的 sqrt 填满新数组；cudaMemcpy H2D 成功后清 bit，保留0x800的 device指针发布工作。正常重复 configure 的 bit4已清，不重填，因此后续 coupled patch 覆盖不被重复 configure 撤销；首次复制失败仍保留初始化工作。补丁未添加 finite/positive/reject/clamp，原方程未改。实际 analysis 中2D十步均64节点有限，最终T=2373.137161817261、carry=.08661541087495861、frozenArea=0；1D INPUT2 与 P007 的全 analysis states（含field hashes）完全相同。

**尚不能签为正式修复**：上述只证明这次 zero producer 诊断和普通1D控制。还需正式开发者证明真实 coupled patches 的上传时序、混合coupled/uncoupled faces、重复configure保留已上传物性、actual copy失败恢复和owner发布；只在合法固定owner backing范围内测试。当前scalar在API中没有可变物性setter，因此将来若增加setter，必须同时定义per-face旧默认值如何更新，不能由该补丁推断。P007也未实现字典7900/500/15；是否应由host消费是另一个物性配置需求。新建全face `std::vector` 的真实host分配失败需要符合owner的错误处理。host sqrt与原device fallback的位级比较、完整墙ledger/gas预算守恒及非均匀多粒子组尚未由本席位关闭。不可把 finite 输出或sanitizer summary0本身等同于完整修复通过。

已读原 cold2D `SAN-NATIVE-CHT-coldWall2D-INPUT2-{memcheck,initcheck,synccheck}.log` 均 `ERROR SUMMARY: 0 errors`，P006仍在首个wet stage真实读出0/0；这与纯数值路径相符，但不证明所有内存/同步路径安全。关键SHA256：生产cold2D=`903b5b615caa4f7aa432126c04a07c054335f04038bf941362a3078e10fe0a61`，生产CHT resident=`3131e68bb2220f415f3ef83133f358c58cb6741f7cfcd3eeebaf5054b8c65284`，P006 log=`5dfd3930fd4a0b87a77af4a9a92ea392ccc59bc3dca0a6642e2d8cfad6595730`，P007 patch=`35983e87accf7a2bf6b5cb4baf7c9adc097579d916d03c689aac6e5af29f3dda`，P007 resident=`0ef0585884b0490380ed66456568e70b739ca1047d6113a0f9ed74af0df39604`。本次读出的 resident-analysis JSON SHA256=`5f269be438fa93c9b9d4837b7cd924ad2de99c451ac2f8fa8d3d39ec3a5698b3`；该分析持续追加案例，引用的是这次读取内容。

## 新增 CHMT FP64 微差：原因与批准范围尚待解释

主代理报告同一 native thermal 两窗口 base/PR均50microsteps、CSV行数相同，mass/energy/temperature/pressure/material相同，但gas momentum绝对差5.42e-20、相对2.66e-15，exchange heat绝对差1.87e-16、相对1.25e-13；原checker容差通过。该数据目前来自主代理运行报告，本席位未独立解析这项完整CSV。**不是所有字段逐位相同，也尚未证明符合“原成功FP64公式/次序”要求**；小于容差不能直接豁免该要求。

静态优先追查 `Backend.cu:427–442` 的剩余时段/Co选dt、`:53/251` 的阶段时间与 `GasWindowProgram.H:11` 插值。旧base直接用 absolute knot−time和 `(time−a.time)/(b.time−a.time)`；当前 `WindowTime.H:8–15/31–45` 使用long-double compensated local time，只有本地差与binary64绝对差恰等时才走原表达式，否则以local差计算比值再转double。相同microstep数量不保证末段dt、stage stamp、alpha逐位一致。低位差可能再经 `Backend.cu:283` flux packet及`:300–305` ledger积分累加表现出来；这里的原packet乘加公式/顺序未见改写。这是源码支持的追查候选，**尚未锁定实测首个分歧或排除编译/FMA/reduction影响**。最小复核是按微步记录 hex dt/stage stamp/alpha/原packet，找到第一项不同后解释其是否属于已批准的补偿时钟边界，不新增生产接纳阈值，不认定mergepass。

## 新增 Ns2 化学算子计时：解释成本方向，限制外推

主代理实际GPU计时报告：同sm89、O2、fmad=false，dt=1e-6、maximumSteps=100、accepted1，3warm/7repeat；1cell base2.731ms vsPR.833ms，32768cell base34.726ms（28.17–37.22）vsPR11.290ms（11.26–11.31）。所有32kcell完整输出FNV-bit-hash一致，固定allocation19/8MiB相同。本席位没有运行计时，也尚未读最终cuobjdump/clock记录，以上明确是主代理的实测陈述，仅限该Ns2化学kernel、成功输入、flags和计时方法。

静态成本方向有依据：base `LocalStiffIntegrator.H:64` 每次advance先进行完整mechanism view验证；base `:43–44` 在full/half/end每次sdirkStep内构造context并prepare正交基/leftInverse。当前 `:175–183` 在推进器外准备，`:211` 的middleContext复用leftInverse；当前strict implicitStage复用evaluate已求得masses/T，线搜索用evaluateState，减少base重复state/热反演。局部数组及device leaf/noinline重排还可能改变寄存器、stack/local memory与occupancy，需真实编译资源数据确认。**源码减少重复调用不等于证明提速倍数或具体主因**；不能把这项计时外推整solver、Ns10、更复杂mechanism、内层失败/terminal重试、冷壁或IO，也不能替代SR-1真实native trial lineage。固定allocation未变不等于所有device局部storage成本未变。

## Recommendations — 当前计划的可用证据与缺口

1. **chemical lineage**：优先补 SR-1 实际 production adapter/native path probe；输出两半步 audit、trial sequence、rollback/commit 状态，和普通成功/public completion 控制。CHMT publicly terminal stage 测试不能代替 gasUGKP 的 native refinement。
2. **冷壁差分脚本须限定范围**：`tests/fixtures/operator_unification/differential.py:126–151` 的 scalar reference 仍含 gas source，CHT PCR call 仅进墙面求解，没有 paired producer；`:141–143` 不复制 carry，`:167–188` 能量也不计 carry；pm 继承压力粒子 .5 而给 cold physicalMass=3.2e-9。`:193–197` 还要求 NaN/负 dt 拒绝并回滚。这些旧断言不适配当前授权路径；保留失败、独立补当前 pipeline，不用它们推动生产重新拒绝数值。
3. **可用 cold 最小入口**：移植 `tests/gas_transport/test_cold_wall_thermal_carry.py:215–273` 的 mixed_cell（该现有文件是 HOST adapter，不能当 GPU 证据）到真实 CHT DeviceState/真实 CUDA。三粒子同 cell、pStatus全1、ordinary/1D/2D，先生产 `correctPoissonThermalizedMobileParticlesKernel<true>` 产生候选，空 selected-stuck consumer，accept；1D wrapper 在其内部由 lane0 发布 paired heat；ordinary 当前 publisher/relaxation；2D wrapper 的 top flux；settle。native 顺序见 CHT GpuResidentStrict.cu:4726/4744/4764/4774/4841/4859。相应 scope 是 thermal-only；尚未覆盖真正 selected-stuck collision/contact 转移。
4. **cold 全输出**：profile8/64、double carry、pT、phase/ring/frozen/history、机械状态/旧速度、wall reflected/deposited、candidate A/Q/C、接受 scale、实际 particleHeatBudget、gas conserved/EOS。库存使用 `sum(pm*(mean(floatProfile)+doubleCarry)) + rhoE*epsGPrev*V + sum(wallLedger)`；pm 用真实 multiplicity×物理质量。分别测热/冷、floor、microheat、1D no-duplicate、2D top nonuniform、8/16/24/32/更多完整组，不只看均温或进程退出。
5. **公开 CSV 的限制**：CoupledOutput.H:21–34/54–60 不输出 chemistry/SST audit、wall diagnostics/time，exchange 也不输出 momentum/species。check_case 可证明其实际读取的能量/质量/EOS/packet energy，却不能关闭两半步诊断累计、SST 六通道或全部终端 packet 字段。这些需 native D2H/probe 或 checkpoint 全字段 decoder。
6. **脚本是否真的运行**：gas_auto_pool.py:53–55 只编译并打印 executable；必须显式执行其 .cu.in:80–107 的七矩/status/RNG 断言。pressure fixture 的 COMPILE_ONLY 环境可跳过运行，应记录实际 RUN。direct_dispatch.cu:99–108 的物理累加是 accumulateMock，只证明真实 persistent dispatch 的调度，不证明真实物理 pool。collision_behavior 的 sampling 数值断言范围较窄，不能由其 RNG hash 推断所有物理输出已检查。
7. **baseline 与精度**：differential 的 baseline.json 是内嵌函数文本，没有可直接当作 d611 整树的 identity；cold32/cold64 比较也不能推广成当前 combined pipeline。为原成功 bitwise/调用次序比较单独 pin 真 d611 源码与匹配依赖/flags/input；守恒阈值与 Cantera accuracy/control 可作为独立物理检查，但不能代替成功 lineage。这里不建议重跑已由准确依赖证据关闭的无关类。

## Declined to judge

- 整个 426-file PR 的逐行正确性：本次只读席位按高风险调用链分批，未逐行读全部文件；该缺口需主代理按 33-topic 矩阵披露。
- GPU correctness、sanitizer、同步/FMA/atomic order 与性能：本席位没有执行授权；仅主代理的固定 source 真实运行记录可作证。
- 故意超出 backing 的 metadata/address、mapper operator[]、overflowing unsafe sizes：按用户禁止故意越界，不执行也不宣称安全；已披露承受的行为不作为新 gate 建议。
- NaN/Inf/负值的有限或物理解质量：允许实际 IEEE 终端输出；只核是否真实表达式/字段/消费者到达，不为病态值新增 finite/positive/convergence 接纳阈值。
- 有效历史 Co 复用的精确即时一致性：用户明确允许效率近似；除 stale/missing backing 外不作为缺陷。
- 冷启动 CHMT Js 初始化与旧冷缓存 FP32 消除造成的授权物理差异：用户批准；不给原 FP64 普通成功差异泛化豁免。
- Carry 在 activation/release 时保留 J/kg、ordinary heat/radiation 严格出版 API 的既有边界：基线已有且旧专项报告明确覆盖，与本次墙面 terminal scope 分开，未识别新改变。
- 全部历史 restart format/tool 和跨 endian 主机运行：只审 current codec/checkpoint 的直接范围，无相应运行环境或完整输入。
- README、正式生产修复、提交/推送/合并/GitHub 评论：只读分工禁止，保持未做。

## Assessment

**Ready to merge? No.**

**Reasoning:** 固定源码存在高可信度的 native chemistry trial 入口未接线问题，违反原成功重试次序保留；独立 runtime checker 也需要随 FieldSchema 维护。当前定向源码审查与实际 kernel 的 StepLimit 红/绿日志足以确认第一项；P002 的更广 unavailable 分类与真实 requested-interval trial 序列尚未关闭。第二项由当前 wire 字符串和 checker 常量直接确认。实际cold2D还揭示基线已有的zero-effusivity producer缺口；P007只关闭本次诊断，CHMT成功FP64微差仍待逐步解释，特定Ns2化学kernel计时不能外推整体。上述不能替代仍未关闭的真实GPU consumer/concurrency/performance验收。正式修复及固定版本反证/重查完成后再作合并判断。




