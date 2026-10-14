# PR14 第四轮 FAST 增量审查申请草稿

PR14_REVIEW_ONLY。你是独立 reviewer。此稿待开发者补齐最终 commit/tree、发布证明、蒸发最终编译/发布证据与生产审计彻底删除的稳定源码/验证后才形成最终运行申请；当前不要把工作树 SHA 或待补章节当作已发布修复。任务只做 FAST，不启动 PRODUCTION，不合并 PR。

## 版本和工作边界

- 仓库：https://github.com/zyy212121/ugkp-thermal
- PR：https://github.com/zyy212121/ugkp-thermal/pull/14
- 目标分支：`perf/chmt-first-round-20261013`
- 最终正式 source commit/tree：待发布者填写 `SOURCE.json` 的 final 字段，并给出实际远端校验；当前为 null。
- 本轮正式开发交付目录：待发布者选择新的 `review/pr14/developer-deliveries/<UTC>-fast/`，本稿目前只位于库外草稿目录。
- 第三轮固定证据：提交 `0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668` 下的 `review/pr14/reviewer-deliveries/20261014T120103Z-fast/`。
- 该轮真正被测源码：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`，tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`。
- 既有 test 区：`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`。只在当前任务明确授权的隔离 test 副本操作，核实安装输出与符号链接仍在 test 区；不覆盖正式源码、正式安装、生产结果或原 checkpoint。

遵守 `review/pr14/PROTOCOL.md`。不得修改 README、PR Conversation 或描述，不 force push，不合并，不新增数值、精度、版本、契约或预检拒绝/回滚/停算边界。真实被消费资源缺失及 IO/CUDA 错误沿原错误通道区分；观察器和外部有限 timeout 不写进生产求解器。

## 先作依赖化增量计划

先读本轮 `REPORT.md`、`SOURCE.json`、`VALIDATION.json`、`EVIDENCE_REUSE.md`、`GAP_RECHECK_PLAN.md`，以及第三轮 `findings.json`、`results.json` 和 `coverage.json`。按需打开固定原日志，不复制或重跑整套旧证据。

在新 `TEST_PLAN.md` 中列出最终源差分、source/dependency/input/binary hashes、当前目标 R014/topic/file/function IDs、拟复用 acceptance unit 及原局限。只重跑真实受影响链条或新增正式断言；不能把尚未关闭但本次无相关改动的缺口自动变成全套新 campaign。若最终蒸发/纯支撑记账/SST与CHMT对外化学审计删除触及共用依赖，解释具体 producer→storage/transfer→consumer→commit 关系后扩展必要最小复验。

开发端没有本轮 GPU 数值结果。其低设备优化原生编译/link 通过，不等于原 O3、runtime 或 sanitizer 通过。两个 O3 exit137 和 host-only O0 exit137 必须保留；不要用低优化 binary 代替正式构建验收。

## 必做的正式化学增量

目标为 `TEST-R3-NEGATIVE-SEMANTICS`，R014-010/012/014/026，文件 `tests/fixtures/chemistry_trial/chemistry_trial.cu.in` 及对应 driver 测试。第三轮正式 owner/成功控制已通过，完整负例证明当时仅存在于 S301/S302 支持版。本轮正式化这些断言，不能拿历史支持版给新提交代验。

使用本轮最终 checkout 自带 generator 原 O3 参数 fresh compile/link，不叠 S301/S302，不替换 kernel/owner，不用 CPU CUDA shim。以下是拟执行模板，不是已执行记录；请记录实际绝对路径、机器、版本和命令：

```sh
python3 tests/fixtures/chemistry_trial/prepare_chemistry_trial.py \
  "$CHECKOUT" "$NEW_REVIEW/fixtures/chemistry-formal" \
  --nvcc "$NVCC" --arch "$CUDA_ARCH"
timeout 300s "$NEW_REVIEW/fixtures/chemistry-formal/chemistry_trial"
```

生成器须仍使用 `-O3` 与 `--fmad=true`，实际架构单独记录；不要添加开发者诊断用的 `--Ofast-compile=max` 或 ptxas O0 来宣称正式通过。默认 proposed 外部预算为 build 600 s、数值 fixture 300 s；预算是观察窗口，不是完成保证。若资源不足，保存退出码/信号、可用内存及诊断，并报告 BLOCKED，不反复 OOM 或偷偷缩减源码。

### 负例与事务

核实全部 16 原组合、所有原输入/阈值/interval/maximumSteps/rejection/Newton 控制保持不变，执行真实 host/device owner 分离、显式 upload 与 sync、native kernel、policy begin/validate/rollback。

每组在 reset/public replay/rollback 覆盖失败状态之前记录两 cell 的 code、thermoCode、reaction、attemptedSteps、rejectedSteps、nonlinearIterations，且按准确期望比较。必须有 32 条对应 kind 0–15 × cell 0–1 的完整状态行。

- kind 10：StepLimit 保留 reaction 0，实际 unavailable 分类与 retry/terminal 保持正确。
- kind 15：原零区间、thermo 元数据失败及未被消费 basis 语义不变。
- kind 7/15：原 native 失败记录与后续 public replay 分列，不能拿 replay 后的 status 冒充失败诊断。
- 所有 cell/face transaction 数组、accepted audit 全语义、transport/initialTransport/chemistrySpecies、faceStatus、host ledger/records 在失败前后及 rollback 后按原位级/语义规则比较。不得用结构 padding 比较取代语义，也不得删去不相等字段换绿。
- retry/terminal 决策在 rollback 后保留，正常路径不 poison owner。输出 ledger 比较只排除 nondeterministic host wall-clock seconds；新增 byte counters 的 partition-independent 推导见 coverage-preservation 证据。

核对 16 条 `PASS failed-cell kind=`、16 条 `PASS failed-transaction kind=`、32 条 `FAIL_STATUS` 和 kind 7/15 的两个独立 `PASS public-replay` 标记，同时审阅实际值、断言与完整原始日志。标记计数本身不是数值验收。

### 原成功与 owner 控制

同一次正式 fixture 运行必须保留四类原控制：4e-6 真 Co/refinement 的精确 accepted-duration replay、maximumAttempts=1 的 public terminal 完整区间、1e-6 real-Co requested-owner 成功和1e-6 direct native/public 成功。输入 targetMaxCo=0.5，起点2，原终点/容差不变；不要硬编码“两段 accepted”或更改 CFL 来换绿。四类控制已有历史通过，但本轮增加的共享 comparator 仍由这次完整 fixture 运行核实，不额外重跑大规模应用。

原 rejected-entry chronology 的限定证明可复用 S305，前提是最终 owner/dependency 未变。不要把 S304 的 stale currentDuration 当作真正 rejected dt；旧部分观察与修正证明均保留。只有相关依赖改变时才做新的只读 chronology 观察。

### 必要内存检查与范围

对同一原 O3 binary 做一次新增负例 transaction/正常销毁范围的真实 `compute-sanitizer --tool memcheck --error-exitcode 99`，默认外部预算300 s，保留完整 summary。若实际成本需调整，先记录有限预算与理由，不能自动转成长算。initcheck/synccheck 仅在源差分与既有证据表明存在新的未初始化/同步缺口时追加，并明确范围；不因此重跑已经通过的 Ns10 full 原输入 sanitizer campaign。

fixture check()/cudaCheck() 会结束进程，成功不能被描述为一般 allocation/free/sync 故障恢复或所有并发安全已通过。任何新增支持补丁都须保留正式原失败、patch/hash 和修后结果，不能静默修改后再宣布提交原版通过。

## 重启项的当前处置

复核开发端只读报告和固定引用即可，不要求仅为“保持源码不改”再跑完整重启用例。

`OBS-R2-RESTART` 的严格 full-versus-restart FAIL 继续开放。thermal 首个温差约600 K处1 ULP，原 dt 序列相同；wet另有27个终态语义差异，二者不能混作同一排他因果证明。现有证据不足以要求清零 Tgas、统一 seed、修改 Newton 停止条件、增加迭代、改 checkpoint 或补偿时钟。也不能把 numerical-continuation 容差检查替换成严格位比较的通过标准。

若最终其他修复不触及恢复路径，沿用 `R3-thermal-restart-first-currentCo-cause`、`R3-wet-restart-strict-bitparity` 及旧802/formal同checkpoint证据，保留原 verdict。只有任务另行要求排他根因，或新差分实际影响该链条时，才在原短输入首受影响位置增加只读真实 GPU 迭代/残差/返回分支观察，并证明 observer 未改变原输出。该观察也不自动关闭严格位差。

## signed open-edge 修复的正式 GPU 增量

当前已应用的窄差分仅是 `CpuFilmDriver.C` 对有限负 open-edge rate 去掉 ordinary 拒绝，沿用原 owner extrapolation。对应 `OBS-R3-EVAP-TINY-LOCAL-STEPS`，R014-003/006/013/014/020/021/026/029等原映射按PROTOCOL核对。开发端17个原成功输出逐位相同、8个新增ordinary等于原terminal同dt、原三组件CPU脚本通过，均不能替代真实 interval owner/GPU 运行。

1. 独立核对完整源diff，确认普通 open-edge 行为扩展明确，原 finite/donor/species/实际资源检查及物理 step limits 保留。没有新增次数/数值/精度限制，没有将所有 numericalFailure 提前转入 terminal，没有仅扩大 dt 而复用 tiny candidate。
2. 对最终 source 作受影响原生 host 编译与最终 CHMT 应用链接。若复用未变 CUDA Backend 对象，须核对对象及完整源/依赖身份，明确这是增量链接；host helper 编译不等于完整应用链接。旧二进制不能冒充本轮正式构建。
3. 使用第三轮原 evaporation 输入、原终点 `0.001 s` 和原比较规则运行正式版本；保留原900s仅到0.00025的BLOCKED。新尝试建议先记录900s的有限外部预算，单GPU顺序执行，不自动加到无限长，不缩输入或终点换绿。如果仍未达到终点，记录实际accepted/global/local时间、dt、进展、错误和资源，保留BLOCKED/INCONCLUSIVE。
4. 记录实际 open-edge signed lengthRate、物理 dt estimator、native ordinary/terminal/rollback 调用、真实history/cursor/packet/geometry consumption以及最终M/E/species与force账。新增成功必须是原 interval 的重新执行，不能仅凭最终有限或日志增长判定完成。
5. 原 wet/noMelting短输入及普通合法outflow、signed inflow、two-ULP pressure控制按实际受影响链条定向核实；保留热量、力、species、geometry/ledger。双向phase控制只在最终diff确实进入其消费者时追加，不重跑不相关全套。原纯支撑记账全系统P诊断保留历史FAIL，但不再作为当前计算修复目标；真实gas P/M/E与严格restart缺口分别保留，不因删除诊断或本项完成而自动关闭。
6. `native_replay_scope.json` 已说明旧D302归档不足以重构完整历史owner。若要确认原首次失败分支，使用隔离只读observer或完整原history capture，证明观察未改原输出；不要将checkpoint-derived端点/forcing控制称作精确历史重放，也不为复现旧已知危险OOB重新运行它。

正式GPU原终点完成前，蒸发问题仍开放。此前所有SIGFPE、300s/900s timeout、tiny-local-step原始证据原样保留。

## 纯支撑反力记账删除与同类冗余审查

用户已取消单scalar压力时间积分提案，明确选择删除对计算结果没有影响的支撑反力记账及专用force构造，并检查同类冗余。不要实施旧 scalar、56/64B或三向量设计，也不要加一种新积分、求积规则或审计补偿替换旧开销。16:44 UTC用户进一步要求将所有只供审计/检查、对守恒量计算无意义的开销一概删除。`CHMT_ENABLE_SST_AUDIT` 方案已取消，生产不留可选audit模式。当前正删除CHMT SST审计13数组/6D2H及CHMT对外chemistryAudit buffer/清零/两half D2H/host累计；真实rhoK/omega/sourceNumber/Co和积分器内部影响求解的量保留。测试核算仅置于tests独立代码。上述实现、原生红绿、最终生产构建及第二遍查漏证据仍为 PENDING，最终计划必须等待实际冻结差分；下列要求不是已通过记录。

1. **零物理消费者。** 对每个删除量列出 producer→storage/copy→consumer→export/codec 路径，覆盖 ordinary/terminal/retry、phase和相关模型分支。证明它不进入状态更新、profile/shear、压力功、热/相变、gas/material/film forcing、geometry、步长/重试/拒绝决定。名称叫 audit/budget 不能代替数据流证明。若发现真实消费者，立即从可删集合剔除并报告，不以改变物理闭合方式完成删除。
2. **物理算序和结果。** 用真实生产算子/owner定向红绿与旧版对照，确认所有既有物理 FP64 输出逐位不变，保留原样本及输入、source/binary hash。允许的不同只能是明确取消的纯记账量/对应诊断输出；不能对整checkpoint无差别比较后掩盖变化，也不能删去真正物理字段或宽化原容差。shear、pressure-work、热量、相变、mass/energy/species、geometry和实际gas动量必须逐项保留。静态无消费者证明与运行证据分别报告，不能用公式surrogate或CPU仿CUDA代替native/GPU路径。
3. **兼容。** 旧checkpoint若需保留兼容字段，核实读取/保存结构与原格式可用，兼容值不参与实际计算。使用实际旧checkpoint和原native codec/owner测试；分别说明保留字段、忽略值、默认值及round-trip/续算范围。不得仅因删除诊断令解析失败而增加新格式/版本拒绝边界，也不得宣称尚未执行的完整GPU重启已通过。严格原full/restart位差仍单列。
4. **工作量确实减少。** 证明相关专用force构造、累加、复制/搬运和仅为该账的consumer已实际删除，没有留下死计算。确认没有新增pass/kernel/下载、压力积分或额外遍历；记录实际diff和必要计数。静态少运算不是整机提速证据；没有匹配实测就不声称运行加速。类似冗余审查的每项必须标记确定可删、待证或真正物理消费者，并明确实际实施状态，不把只读发现写成已优化。
5. **受影响最小真实执行。** 按最终diff确定必要host组件/最终应用链接及短GPU控制，沿用未变依赖的旧证据，不重跑全套或生产长算。原蒸发.001终点仍是独立缺口，支撑账删除不能代替该终点。若原wet短case用于物理FP64不变验证，比较真实gas/film/material状态和真实flux/热量；不要把已取消支撑账的总残差当本轮修复目标。
6. **守恒与历史边界。** 原完整gas+reduced-film+support诊断FAIL保留为历史事实；用户已撤销其作为实际计算修复目标。这是范围变更，不是守恒PASS、修复证明或物理验收豁免。真实gas动量/质量/能量的原验证与剩余缺口必须保留；删列、置零或不再输出残差不能用来报守恒。原gross库存、实际资源故障、严格restart等不相关开放项不受此变更关闭。

### SST 与对外化学审计整链删除的额外核验

- CHMT GasSstAudit的13数组、6D2H及其专用分配、清零、填充、reduction、复制与host累计应在生产路径整体消失。CHMT对外chemistryAudit buffer、清零、两个half的D2H与host累计同样核实。不能仅关最终输出、留死计算，不能留下 `CHMT_ENABLE_SST_AUDIT` 或另一生产audit开关作为替代实现。
- 列明保留的真实物理消费者：rhoK、omega、sourceNumber、Co，以及化学积分器内影响求解、误差控制、step接受/拒绝或状态更新的量。不得将“audit”名字当成删除依据；若量同时参与求解必须保留并报告其消费链。真实gas momentum/mass/energy和物理FP64算序不因删除统计而改写。
- 对最终单一生产配置完成必要native编译/链接，保存flags和binary/source/dependency身份；没有新GPU运行就标NOT_RUN。没有default/diagnostic双生产模式验收要求。针对真实生产算子/owner作物理解等价检查，必要计算核算只在独立tests中实现，不把审计功能重新带回生产运行链。
- 如为旧checkpoint兼容保留历史槽，证明槽值无求解反馈，外部检查不把废弃/未采集值解释为实际零生产耗散或守恒通过。使用真实旧格式native codec/owner定向验证；未执行的GPU续算单列，不能借删除诊断关闭原strict restart。
- 第二遍查漏覆盖shared-gas opt-in budget、radiation等仅审计候选，逐项列出已证实无物理消费者且实际删除、待证、或有真实消费者必须保留。当前尚无最终报告，不先报全仓消除冗余。没有匹配实测不声称整体提速；没有新增pass/kernel/复制来替代已删工作。
- 原gross审计缺口与真实gas P/M/E验证不因审计设施删除而通过。后续若需补证明，应在独立测试/只读外部见证中取得实际证据，不把可选生产audit功能作为本轮设计重新引入。

最终交付须给删除清单、保留的真实物理消费者清单、兼容字段说明、原生红绿/FP64对比完整结果和未执行项。若任一项只有待证结论，保留PENDING，不为凑齐PASS改物理计算或制造新条件。`whole_FAST_acceptance` 与 `merge_ready` 仍为false。

## 证据复用与保留

逐项沿用第三轮 acceptance-unit 原范围及固定 commit/path/SHA256；`evidence/reused-evidence-index.json` 保留全部14个历史gap ID及附加开放范围。

不受最终依赖变化影响时，直接复用 Ns10 full original initcheck、active CHT/FSH16完整组、合法双向相变、host codec/mapper/wire、SST/ALE/Co ownership和原profiling等限定证据。不要重复全套测试，不把historical PASS计作本轮新GPU运行。未执行的 gross inventories、GPU empty/multi mapper、材料/几何、其他资源/并发范围仍保持开放；本稿不自动追加这些无关工作。

所有旧 FAIL/BLOCKED/INCONCLUSIVE 原记录保留，特别是严格累计P、原完整wet支撑账诊断、full/restart位差和蒸发未到原终点。原纯支撑账全系统诊断已退出此次计算修复目标，但历史FAIL不改成PASS；真实gas P/M/E范围不缩减。第三轮整体FAST本就未通过，本轮局部夹具修复不能改写该事实。

## 统一交付

新建 `review/pr14/reviewer-deliveries/<新UTC>-fast/`，不覆盖旧目录。交付 `REPORT.md`、`TEST_PLAN.md`、`SOURCE.json`、`coverage.json`、`results.json`、`findings.json`、实际命令、环境/版本、raw logs、必要输入/逐点数据、支持或只读诊断patch和SHA256SUMS。

每项记录 source commit/tree/blob/dependency/input/binary hashes、UTC起止、工作目录、精确命令、GPU/driver/CUDA/OpenFOAM/flags、外部timeout、退出码/信号、执行类型、预期与实际逐字段结果和局限。正式版、支持版、诊断版、历史沿用和NOT_RUN分列；一个raw artifact只存一次，跨topic引用相同evidence ID。

按PROTOCOL追加对应responses/manifest/owned_paths并执行行政验证；这些是审查记录，不是生产运行条件。只有本次任务另有明确发布授权时才正常推送同一PR分支，推送前核对并保留他人提交，不force push。最后报告固定审查提交、远端核验、新目录、限定关闭范围和全部剩余缺口；不得宣称merge-ready或执行合并。
