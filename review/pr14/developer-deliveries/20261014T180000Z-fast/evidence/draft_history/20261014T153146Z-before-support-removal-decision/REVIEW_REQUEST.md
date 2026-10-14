# PR14 第四轮 FAST 增量审查申请草稿

PR14_REVIEW_ONLY。你是独立 reviewer。此稿待开发者补齐最终 commit/tree、发布证明、蒸发最终编译/发布证据与 wet 决定后才形成最终运行申请；当前不要把工作树 SHA 或待补章节当作已发布修复。任务只做 FAST，不启动 PRODUCTION，不合并 PR。

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

在新 `TEST_PLAN.md` 中列出最终源差分、source/dependency/input/binary hashes、当前目标 R014/topic/file/function IDs、拟复用 acceptance unit 及原局限。只重跑真实受影响链条或新增正式断言；不能把尚未关闭但本次无相关改动的缺口自动变成全套新 campaign。若最终蒸发/wet 修改触及共用依赖，解释具体 producer→storage/transfer→consumer→commit 关系后扩展必要最小复验。

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
5. 原 wet/noMelting短输入及普通合法outflow、signed inflow、two-ULP pressure控制按实际受影响链条定向核实；保留热量、力、species、geometry/ledger。双向phase控制只在最终diff确实进入其消费者时追加，不重跑不相关全套。完整wet物理P和严格restart位差按原标准分别报告，不因为本项完成而自动关闭。
6. `native_replay_scope.json` 已说明旧D302归档不足以重构完整历史owner。若要确认原首次失败分支，使用隔离只读observer或完整原history capture，证明观察未改原输出；不要将checkpoint-derived端点/forcing控制称作精确历史重放，也不为复现旧已知危险OOB重新运行它。

正式GPU原终点完成前，蒸发问题仍开放。此前所有SIGFPE、300s/900s timeout、tiny-local-step原始证据原样保留。

## wet 单 scalar 接口提案待审批

最新提案仅是每 face/history 累积一个 `Σ ledgerDt * p_EOS` scalar，材料步用原 geometry 替换 `p_first * dt`；现有 shear/Gδ/phase 热量不动，不增加 α 规则。该提案已提交用户审批但未获回复；最终设计已通过独立只读数学和数据路径审查；旧56/64B提案不是当前范围。

当前不得实施、运行或将其列为正式修复。最终设计与独立复核见 `evidence/wet_design_pending/`，设计审查通过不等于接口批准或实现验证。发布前需明确用户决定、数学审查结果、精确源差分和适用几何/压力/功语义；若获批实施，再基于真实依赖补最小实际stage/cursor/remainder/retry/commit及原wet控制，而非预先宣称整个moving/phase机械闭合。原实际gas M/E/vector-P限定通过与完整gas+reduced-film P红项分别保留，不用局部账或虚构反力替代。`whole_FAST_acceptance` 与 `merge_ready` 均为false。

## 证据复用与保留

逐项沿用第三轮 acceptance-unit 原范围及固定 commit/path/SHA256；`evidence/reused-evidence-index.json` 保留全部14个历史gap ID及附加开放范围。

不受最终依赖变化影响时，直接复用 Ns10 full original initcheck、active CHT/FSH16完整组、合法双向相变、host codec/mapper/wire、SST/ALE/Co ownership和原profiling等限定证据。不要重复全套测试，不把historical PASS计作本轮新GPU运行。未执行的 gross inventories、GPU empty/multi mapper、材料/几何、其他资源/并发范围仍保持开放；本稿不自动追加这些无关工作。

所有旧 FAIL/BLOCKED/INCONCLUSIVE 原记录保留，特别是严格累计P、完整wet物理P、full/restart位差和蒸发未到原终点。第三轮整体FAST本就未通过，本轮局部夹具修复不能改写该事实。

## 统一交付

新建 `review/pr14/reviewer-deliveries/<新UTC>-fast/`，不覆盖旧目录。交付 `REPORT.md`、`TEST_PLAN.md`、`SOURCE.json`、`coverage.json`、`results.json`、`findings.json`、实际命令、环境/版本、raw logs、必要输入/逐点数据、支持或只读诊断patch和SHA256SUMS。

每项记录 source commit/tree/blob/dependency/input/binary hashes、UTC起止、工作目录、精确命令、GPU/driver/CUDA/OpenFOAM/flags、外部timeout、退出码/信号、执行类型、预期与实际逐字段结果和局限。正式版、支持版、诊断版、历史沿用和NOT_RUN分列；一个raw artifact只存一次，跨topic引用相同evidence ID。

按PROTOCOL追加对应responses/manifest/owned_paths并执行行政验证；这些是审查记录，不是生产运行条件。只有本次任务另有明确发布授权时才正常推送同一PR分支，推送前核对并保留他人提交，不force push。最后报告固定审查提交、远端核验、新目录、限定关闭范围和全部剩余缺口；不得宣称merge-ready或执行合并。
