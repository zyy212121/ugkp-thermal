# PR14 第四轮 FAST 开发交付草稿

PR14_REVIEW_ONLY。此目录是尚未发布的开发交付草稿，不是整体 FAST 通过结论。化学负例断言已补入正式测试源码，低设备优化的完整原生 CUDA 编译和链接通过；原 O3 构建仍受阻，修订后正式夹具尚未在 GPU 上运行。重启只读审查建议保持成熟数值路径不变，同时保留严格 full-versus-restart 位差。蒸发的 signed open-edge 窄修复已应用并完成 native CPU 组件验证，原完整 GPU 终点仍待验证；用户已取消 wet 单 scalar 提案，改为删除无物理消费者的纯支撑反力记账及专用 force 构造，并排查同类冗余。用户随后要求彻底删除所有仅供审计、对守恒量计算无意义的生产开销；CHMT SST审计与对外chemistryAudit链正在删除，不保留生产audit开关。实现及验证证据待补。发布身份和剩余证据必须在发布前补齐。

## 当前身份和发布状态

- 仓库与 PR：`zyy212121/ugkp-thermal`，PR #14，目标分支 `perf/chmt-first-round-20261013`。
- 当前证据基点：`0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668`；第三轮目录 `review/pr14/reviewer-deliveries/20261014T120103Z-fast/`。
- 第三轮真正被测的正式源码：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`，tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`。
- 本稿尚无最终 source commit/tree、远端发布证明或行政注册。`SOURCE.json` 的工作树文件 SHA256 只固定当前草稿，不冒充已提交版本。
- 未修改 README 或 Conversation，未合并，未启动 PRODUCTION。本目录尚未复制进仓库。

## 化学负例断言修复

对应 `TEST-R3-NEGATIVE-SEMANTICS`，相关 R014 topics 为 010、012、014、026。仅修改：

1. `tests/fixtures/chemistry_trial/chemistry_trial.cu.in`
2. `tests/test_chemistry_trial_cuda_fixture.py`

正式夹具新增第三轮 S301/S302 已在真实 GPU 上通过的独立负例 transaction 见证。16 种原模型、被消费 backing、元数据和 maximumSteps/maximumRejectedSteps/Newton 组合均保留；每组独立准备 accepted-state reference，调用真实 `SharedGasTrialPolicy::begin`，运行原生 chemistry kernel，检查 native validation 后执行真实 rollback。

新增检查覆盖两 cell 的六个 `ChemistryStatus` 成员及准确 cell-status 分类，所有 transaction cell/face 数组、全部 `ChemistryAudit` 语义成员、transport/initialTransport/chemistrySpecies budget、faceStatus、host ledger、rollback 后完整恢复及 retry/terminal 标志。kind 10 断言 StepLimit 同时保留 reaction 0；kind 15 的零区间和 thermo 元数据失败不被误写成未消费 basis 的存储错误。kind 7/15 的原 native 失败先记录，随后 public replay 使用独立标记，两 cell 的 replay 审计均检查。

`compareBudget` 还直接核对 identity、path、deviceBytes、hostReadBytes，补足空 ledger 不通过输出记录间接覆盖这些字段的范围。字节计数不依赖 accepted 子步分段：本 fixture 的 owner/reference 各 prepare 一次、commitInterval 一次，共读 256 bytes；direct-kernel 对照为 0，负例 transaction/reference 各 prepare 一次为 80 bytes。trial begin/commit/rollback 不增加这些 host-read 计数，deviceBytes 在手工分配 fixture 中保持默认 0。详细调用点和推导见 `evidence/chemistry_fixture/coverage-preservation.json`。

fixture 输入初始化、原 `ownerTests` 函数和 16 组合配置区块与基点逐字节一致；完整失败状态的期望及比较区块与 S302 逐字节一致。生产 chemistry、TrialPolicy、Co、时钟、阈值、精度和项目默认编译参数均未因本项改变。未增加生产拒绝、rollback 或停算条件。

## 实际验证及其边界

| 检查 | 实际结果 | 可证明的范围 |
|---|---|---|
| 新生成契约测试的 red | 1 failed，3 passed | 正式夹具缺少完整 transaction 检查时，新测试会失败 |
| 最终定向 pytest | 4 passed，2 skipped | 生成及支持契约；默认 PATH 无 nvcc，两个 CUDA pytest 用例跳过 |
| 原 O3 完整 CUDA 编译 | exit 137，Killed，195 s | 未得到成功编译或链接，原日志保留 |
| 同命令 O3 重试 | exit 137，188.02 s，最大子进程 RSS 2,088,004 KiB | 未通过；不继续重复重编 |
| 仅 host `-O0` 检查 | exit 137，189.78 s，RSS 2,087,956 KiB | device 优化仍开，未通过 |
| 低设备优化完整 CUDA 编译和链接 | exit 0，19.84 s，RSS 792,760 KiB | 完整正式 fixture 直接包含 unchanged production owner 的原生编译/link；ELF 有 `.nv_fatbin` |
| 新正式源码 GPU 数值、WSL 和 sanitizer | NOT_RUN | 必须由审查端取得，不用 CPU 仿 CUDA 替代 |

成功的诊断命令只在独立证据脚本中增加 `-O0 --Ofast-compile=max -Xptxas=-O0`，仍保留 `sm_89` 和 `--fmad=true`。原 generator/项目默认 O3 命令没有修改。此成功不能替代正式 O3 codegen、运行性能或数值验收。接近 2 GiB 的两次记录使资源上限成为合理怀疑，但缺少 OOM 计数，不能断言 kill 的确切机制。

四次尝试的完整命令、UTC、退出码、日志和资源数据均在 `evidence/chemistry_fixture/compile_receipt.json` 及其引用文件。成功日志保留 30 条既有 #20013-D 和 2 条 #177-D warning，没有添加 suppression 或 relaxed-constexpr。9 个明确捕获的关键源文件全部重新校验一致；这份清单不是整个传递 include 闭包的替代品。源补丁为 `evidence/chemistry_fixture/chemistry-fixture.patch`。

## 重启只读审查

`evidence/restart_readonly_review/REVIEW.md`、`metrics.json` 和复算脚本只读取固定源码及既有 GPU 日志，辅以离线 binary64 算术。没有新的 GPU 计算，也没有源码修复。

thermal window2 的 400 条 post-currentCo 单元配对记录显示最大温差 `1.1368683772161603e-13 K`，约 600 K 处 1 ULP；最大压力差 `2.9103830456733704e-11 Pa`，保守质量、总能量和体积差为零；50 条 selected-dt 完全相同。现有热反演初值与停止条件允许相邻浮点温度返回，证据不足以要求清零缓存、统一 cold seed、增加无意义迭代、改 checkpoint 或改写成熟 FP64 热恢复。

这不关闭 `OBS-R2-RESTART`。严格 full-versus-restart 位比较仍失败；独立 wet 用例保留 27 个终态语义差异，不能概括成“所有字段只有 1 ULP”，也不能用 thermal 的 seed 观察作 wet 的排他因果证明。原 numerical-continuation 容差检查与严格 byte equality 是不同标准，本文不改变任何一个标准。

三项关键热恢复源码在已核实的 802、4b5、9d5 版本间 blob 相同，802/formal 同 checkpoint 的 10 文件逐字节相同。这支持“非最近两轮修复新增”，不足以推广到整个 PR 之前的最早 d611 基线。保持未改源码不需要再跑一次长 GPU 回归；若最终其他修复不触及该路径，沿用固定证据即可。若另行要求捕获确切 Newton 返回分支，仅补一次隔离的首受影响真实 GPU 只读观察，不能以此替代 wet 自身的见证。严格位一致是否仍是整体验收必要条件由用户决定；本稿没有更改或豁免。

## 蒸发窄修复和仍开放的原终点

### 实际修复与 native 证据

`OBS-R3-EVAP-TINY-LOCAL-STEPS` 保持开放。第三轮正式原输入 900 s 仅提交到 `0.00025 / 0.001 s`；独立 300 s 只读观察见约 `1.13686e-16 s` 的 tiny local dt 和持续微小进展。原 SIGFPE、原 D204 300 s timeout、正式 900 s 阻断均保留。

本轮已在 `applications/CHMT/film/CpuFilmDriver.C` 删除 ordinary transport 对 finite negative open-edge lengthRate 的拒绝条件，保留原 owner reconstruction/extrapolation 算术；对应回归位于 `applications/CHMT/devtools/multirate/test_cpu_film.cpp`。这是明确的普通路径行为扩展：从 outflow-only 变为 signed owner-extrapolated 开边通量，不能称原普通边界模型早已声明支持 inflow。nonfinite、实际 donor/species/storage 检查及物理 step estimator 保留，没有新增阈值、kernel 或遍历，没有移动 terminal owner 的一般完成边界。

原生 production CpuFilmDriver 的独立检查显示：17 个原共同成功输出的全部序列化字段/binary64 位完全不变；8 个新 ordinary 成功候选与原 terminal 在同一 dt 的完整输出逐位相同。原严格 C++14/O2 回归完成 red/green，加入 signed inflow 和两 ULP pressure 差控制；原 `check_cpu_film.sh` 的 film、curved-film、phase-lifecycle 三组件在应用后的共享树上全部通过，exit0、46.36s、最大子进程RSS318688KiB，输入前后hash一致。证据位于 `evidence/evaporation_native/`，这些是真实 CPU 生产算子的检查，不是 CPU 仿 CUDA，也不是完整 GPU owner 运行。

结合既有 D302 时序，已用精确捕获时长重现 4,276 个局部 dt 的30次二分关系；native film 算子在 checkpoint-derived pressure 下复现与 dt 无关的 finite negative open-edge 拒绝，均压隔离控制通过。这里仍缺原完整失败 owner 的全部 gasTrace/gradient/traction、geometryEndpoint、history/forcing等字段，因此没有完成原真实失败历史的逐位 owner replay，也没有直接捕获原首次拒绝的完整栈。`native_replay_scope.json` 明列缺项，不能以窄重放证明冒充端到端根因及终点验收。

下一步需对最终正式源码完成原 flags 的受影响 native host 编译/最终应用链接，再在原 GPU short case 上补受影响控制和原完整 `0.001 s` 蒸发终点。原输入、终点和标准不变；保留原阻断，不通过无限预算、缩终点、关闭蒸发或新门禁换绿。当前只可称窄机制及窄修复已获得 native 证据，不能称整个 `OBS-R3-EVAP-TINY-LOCAL-STEPS` 已关闭。

## 最新决定 删除纯支撑记账并核实零物理影响

用户消息 `Sentinel_bed493fb3af4819191893c13d013f892` 明确要求“直接删除，并且举一反三的检查一下还有没有这种对计算结果不产生任何影响的多余计算？”。此前单 scalar EOS 压力时间积分提案已取消；它及原独立设计审查只作为历史快照保存在 `evidence/wet_design_superseded/`，不再待审批，也不得继续实施。

当前已授权工作是删除纯支撑反力记账及仅为它构造的 force，逐个证明这些量没有物理消费者。不修改真实 shear、压力功、热量、相变、状态推进或用于实际求解的压力；不新增积分字段、遍历、kernel 或拒绝边界。16:44 UTC最新指令进一步要求：对仅供审计/检查、对守恒量计算没有意义的开销“一概删除”。短暂考虑的 `CHMT_ENABLE_SST_AUDIT` 方案已取消，不留可选生产audit模式。CHMT GasSstAudit 的13数组/6D2H整链及CHMT对外chemistryAudit buffer、清零、两half的D2H和host累计正在删除；积分器内部真正影响求解/step决策的量保留，真实rhoK/omega/sourceNumber/Co与物理推进保持。必要测试核算放在tests独立代码，不能变成生产功能。第二遍继续查shared-gas opt-in budget和radiation等候选；最终删除范围须有零物理消费者证据，未证实项不能先报已删。

支撑、SST及CHMT对外化学审计删除源码正在实施，精确路径、最终 patch/hash、零消费者证明、原生红绿、最终单一生产构建、既有 FP64 物理解逐位不变及旧 checkpoint 兼容证明均标为 PENDING，不能借先前设计审查或旧蒸发三组件通过宣称新删除已经验证。最终证据到齐后再更新 `SOURCE.json`、完整差分与依赖复用范围。

旧 `OBS-R3-WET-P-PHYSICAL` 的完整 gas+reduced-film+support 账残差（约 `−9.01619622716089e-5`、`−1.57820679374845e-4 kg·m/s`）保留原历史 FAIL 记录，但这套纯记账全系统指标不再作为此次实际计算修复目标。删掉字段或残差列后不再出现残差，不构成守恒通过，也不能写成修复了全系统物理动量。真实气相动量、质量、能量、真实 flux/source/库存及其他原未覆盖范围仍需独立验证；原严格 restart 位差不会因此自动关闭。

本次删除的独立审查重点是：所有生产者到消费者的路径均无物理反馈；只有明确诊断/兼容输出改变；旧 checkpoint 所保留兼容字段读取后不参与计算；物理浮点算序和结果不变；没有新 pass/kernel/下载/额外积分来替代被删除开销。完整验收要求见 `REVIEW_REQUEST.md`。

## 最小新增审查

当前新增范围包括本次正式 chemistry fixture、已应用的 signed open-edge 修复，以及正在实施的纯支撑记账、SST及CHMT对外化学审计彻底删除；删除项尚未交付稳定源码和证据。化学部分为：原 O3 fresh build、真实 GPU 全 fixture 和新增负例 transaction 的必要内存检查。原四类 owner/direct 控制随同一 fixture 一次运行保留，不为其另开重复 campaign。S302/S305 及第三轮已通过的 Ns10 full initcheck、active cold groups、合法双向 phase、host mapper/wire 等按固定 source/dependency 和原限定范围沿用。

蒸发部分只复验其实际受影响链条、合法控制及原短输入完整终点；审计删除部分待稳定差分后验证零物理反馈、FP64等价、最终生产构建与兼容。不要再排期单scalar实现或用纯支撑账全系统残差作为其修复目标。原 14 个 gap ID 和严格红项完整列在 `GAP_RECHECK_PLAN.md` 与 `evidence/reused-evidence-index.json`；未受影响的开放项保留状态，不借本次局部交付扩大为重新跑所有缺口。完整 reviewer prompt 见 `REVIEW_REQUEST.md`。
