# PR #14 第三轮 FAST：正式修复与剩余缺口复验申请

PR14_REVIEW_ONLY。本文件可以直接交给一个没有前文的新审查 agent。你是独立审查者，负责客观源码审查、真实 GPU 测试和完整证据交付。本轮只做 FAST，不启动 PRODUCTION，不合并 PR。

## 1. 仓库、版本与授权边界

- 仓库：https://github.com/zyy212121/ugkp-thermal
- PR：https://github.com/zyy212121/ugkp-thermal/pull/14
- 分支：`perf/chmt-first-round-20261013`
- 本轮开发交付：`review/pr14/developer-deliveries/20261014T112400Z-fast/`。正式修复 commit/tree/blob 和编译范围以其中 `SOURCE.json`、`VALIDATION.json` 为准。
- 本轮正式修复 commit：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`，tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`。已核对五个 source/test 文件的 blob/SHA256 与本地编译源一致；本地/远端完整 tree 相同。固定链接：[正式修复提交](https://github.com/zyy212121/ugkp-thermal/commit/9d5be6eff69dbced2fc959d422d3ba419a1bc39e)。不要把后续记录提交混作新的生产改动。
- 上轮真实被测正式源码：`4b5c480b572f234ddecf5e89164fe59b0d2e9735`，tree `0fa2562f4c83394cdf94e9650de3b22c4fb4bb15`。
- 上轮不可变证据提交：`f5e6c5e2cbde2693f5fa811d874c200b07bbbc66`，目录 `review/pr14/reviewer-deliveries/20261014T072053Z-fast/`。完整链接：https://github.com/zyy212121/ugkp-thermal/tree/f5e6c5e2cbde2693f5fa811d874c200b07bbbc66/review/pr14/reviewer-deliveries/20261014T072053Z-fast
- 用户本地 test 开发区：`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`。用户授权在该 test 区完整重构、覆盖、更新源码，重新构建、安装测试产物和调整测试支持；旧原始失败证据须先保留。正式仓库、正式安装、生产结果及原始 checkpoint 不在覆盖权限内。先确认实际路径、安装输出和符号链接均留在 test 区，不把 test 构建安装到正式路径。
- 为越过测试阻断，可以在隔离 test 副本做最小诊断或夹具支持修复；必须保留原失败、补丁、源/对象/二进制哈希和修后结果。诊断版通过不等于正式提交通过；正式生产修复由开发者承担。
- 允许将完整新审查目录及诊断补丁正常推送到同一 PR 分支。不得 force push、合并、写 PR Conversation 评论或修改描述；不得新增或修改 README。
- 不得新增使原来可继续的计算被拒绝、回滚或停算的数值、精度、版本、契约或预检门禁。不要用新门禁修测试。真实 CUDA/内存/IO 和实际被消费资源缺失仍按原错误通道区分；测试断言和外部超时仅是观察工具。

## 2. 先建立增量测试计划

先读 `review/pr14/PROTOCOL.md`、本轮 `REPORT.md`、`SOURCE.json`、`VALIDATION.json`、`EVIDENCE_REUSE.md`、`GAP_RECHECK_PLAN.md`，以及上轮 `REPORT.md`、`findings.json`、`coverage.json`、`results.json`。仅按需要打开其中引用的原日志、输入、逐点数据和补丁，不复制整套旧证据。

核对本轮真实差分及 producer → storage/transfer → consumer → commit 调用链。先在新 `TEST_PLAN.md` 中列出本轮 source/dependency 哈希、沿用的验收单元 ID、必须重跑的原红项和剩余缺口。已证明且不受依赖变化影响的项直接复用，仍标记其原来源与原适用范围；不重新跑全套，不把同一证据在多 topic 重复计数。若实际差分触及更大依赖，明确写出因果关系并扩大必要复验，不能仅按文件名推定无影响。

开发者本轮只提供源码检查、真实编译/链接及明确标记的 host 回归。没有本轮 GPU 数值验收。CPU mock、源码字符串检查、生成成功、编译/链接成功均不能代替你的真实 GPU 结果。

## 3. 首要证明 A：GPU-F006 原红输入 → 正式版本

目标文件 `applications/CHMT/materials/CpuMaterialDriver.C`，既有 file ID `F014-004c176a472f`；沿用 GPU-F006 和原 R014 topic 映射。上轮合法 wet / ThicknessAveraged / noMelting 输入，`phaseCondensed=-1`、零相变质量的 SolidFilm 热包仍进入 `condensed[-1].rho` 消费，heat-only 和 evaporation 原版均留下 SIGFPE。静态非法消费已确认，精确信号 PC 尚未由栈证明；不要为了取数再次运行已知 OOB 旧版。

本轮修复意图是零相变质量不消费未选择的相变材料，仅对应体积贡献为零；原热量、力、动量交换及所有非零相变表达式和消费者必须继续保留。独立检查修复没有把整个 phase packet、film forcing、热通量或压力/力链一并跳过。

1. 从上轮不可变输入恢复 `EXTRA-wet-film-S209` 的同一物理问题，在本轮正式版本上完成原完整短区间，不叠加 D204，不改终点、物性、精度、阈值或 noMelting 设置换绿。
2. 按原标准记录全部 accepted timeline、gas/solid/film 状态、实际热包和力、几何/体积、global 与逐 recipient 的 M/E/P 全库存。原 D204 只有 M/E 限定通过；其 P 严格红仍须解释和单独保留。
3. 对同输入作正式版本 wet restart/continue 的受影响核对，保存真实 checkpoint、实际 consumer 状态和首次差异。不能用“最终有限”或“进程 exit 0”代替完整收支，也不能把原 full-vs-restart 严格差自动关闭。
4. 原 evaporation 物理输入也用正式版本补测。保留原版 SIGFPE 和 D204 原 300 s timeout。按第 6 节计划增大外部观察预算，不缩减输入，不改物理问题。仍未完成就保留 BLOCKED/INCONCLUSIVE，不能记 PASS。

## 4. 首要证明 B：合法非零相变成功控制

使用有效选择的相变材料和真实非零相变质量的小型合法输入，覆盖实际被改动分支；不得通过非法 index 试验制造风险，也不得靠新增 index 门禁绕开。

- 用明确有效的 melting/solidification 控制或已存在等价输入，记录所选材料、方向、phase mass、密度、porosity、recession/体积、film forcing、热量/力、交换包与最终库存。
- 正式修复前可安全运行的合法控制，与修复后用相同输入、步数、flags 比较；沿用原比较规则，保留位级对照或逐字段误差及依据。
- 证明非零分支原表达式和原成功结果没有被改变；零质量分支仍传递热量和力。若一项输入只覆盖一个相变方向，明确范围；按实际分支需要补另一个方向，不宣称全部材料模型已通过。
- 如需只读 observer，在隔离副本保留 patch/source/binary hashes，并证明 observer 未改变原输出；观察 I/O 不参与性能测量。

## 5. 首要证明 C：正式 chemistry 夹具的完整语义

沿用 `TEST-R2-S1`、`TEST-R2-S2`，目标 `tests/fixtures/chemistry_trial/chemistry_trial.cu.in`。上轮生产 owner 的限定证据 S208 有效，但提交夹具本身仍有 managed host/device 所有权问题和硬编码“两段 accepted”问题；不能借 S208 宣称提交夹具已通过本轮 GPU 验收。

用本轮正式 fixture 生成、编译并在真实 CUDA 设备运行，不叠 S202/S203/S208。读取实际 runner 的参数；建议入口为 `python3 tests/fixtures/chemistry_trial/prepare_chemistry_trial.py <本轮 test checkout> <新目录下 fixture 构建目录> --nvcc <实际 nvcc> --arch <实际架构>`，随后在记录好的外部预算内执行生成 binary。此处是拟执行命令，实际完整命令必须另行记录。

必须交付：
1. 16 组合失败矩阵，包含实际缺失 thermo/mechanism/reactants/volume 等被消费 backing、元数据组合、数值/StepLimit 及 maximumSteps 交互；原失败 state/audit/budget 不泄漏。非法供给的危险 OOB 不在测试范围。
2. 原 requested interval 的真实 owner：首次 StepLimit、缩步、rollback、最终精确覆盖整个原请求区间；按真实 Co/CFL 和 accepted timeline，通过 native=true 原路径重放实际成功分段；public-false 对照另见下一项，不强制 2 段，也不把“不是 2 段”改成“任意成功即可”。默认 targetMaxCo 的真实值为 0.5；记录本次实际控制和各 accepted dt。
3. 原 1e-6 requested-owner 成功控制、原 1e-6 direct chemistry 成功控制、public false 对照，以及内部 effort 耗尽后 terminal 完成；首段与 terminal 的真实调用链、完整请求结束时间和 begin/commit/rollback/terminal 次数均需记录。
4. 按字段而非结构 padding 比较所有 cell/face transaction state、ChemistryAudit/ChemistryStatus 语义成员、全部 budget arrays 以及 host ledger/records（仅排除 host wall-clock seconds）、所有物种和状态标志，保留 accepted trace；完整语义审计不应退回只比较个别 counters。检查 host owner/device state 分离、显式同步、上传/下载时点及正常销毁所有权，WSL 异步 host 访问不应靠偶然同步掩盖。夹具 check()/cudaCheck() 失败时会结束进程，本项不宣称一般 allocation/free/sync 错误恢复通过。

核对 16 条 `PASS failed-cell kind=` 与四类 owner/terminal/requested-success/direct-success 标记，同时检查真实数值与 trace；标记字符串本身不构成验收。保存生成的 command.json、build.log、run.log。

原 S208 是保留的历史生产证明；本轮正式夹具的 GPU 执行是新增证明。若 fixture 或 observer 再需修复，原失败、新支持 patch 和修后结果分列，不能算提交原版 PASS。

## 6. 剩余 FAST 缺口与预算

按 `GAP_RECHECK_PLAN.md` 的依赖顺序补齐尚未关闭项；先完成以上三条证明，复用已经成立的独立证据，不把缺口又扩成整套生产长算。重点包括真正 active 的完整 cold 组、full-vs-restart 首差、完整动量/阶段库存、cold-start Js/history-Co 新输入、empty/multiple-selected mapper、pore/melting、codec 之外 host sanitizer 及尚未覆盖的实际资源恢复。

默认每个小型运行外部预算 300 s。原 `CHMT-chemistry-full-initcheck` 和 `EXTRA-wet-evaporation-S209-D204` 的两个 300 s timeout 均须永久保留；不得覆盖为新结果。针对这两个已知缺口，在新 TEST_PLAN 先列原输入 SHA256、原命令/完整终点、未改比较标准、增加预算的理由、机器资源和明确有限预算，再执行扩大预算尝试。建议 Ns10 full initcheck 采用 1800 s、同原 evaporation 输入采用 900 s 的独立 FAST 观察窗口，单 GPU 顺序执行；这不是完成保证。如仍不足，保留进度和阻断，报告下一步所需预算，不自动转入长算。预算仅限制外部测试进程，不写进求解器。

Ns10 full initcheck 必须取得原完整 reacting 输入的最终 sanitizer 汇总与原结果检查；Ns10 control 的 initcheck 0 不能代替。evap 必须分辨推进慢、缺乏进展、异常或真实资源问题，保留时序证据；不得通过缩短终点、降低网格/步数或关闭蒸发换绿。

## 7. 一次性完整交付到同一 PR

新建一个不覆盖旧记录的目录：`review/pr14/reviewer-deliveries/<本次 UTC 时间>-fast/`。本轮所有原始日志、命令、环境、源/对象/二进制哈希、输入/逐点数据、补丁和报告统一归入这一新目录；需要旧证据时只引用固定提交、路径与 SHA256，不复制大量旧日志。

至少包含：`REPORT.md`、`TEST_PLAN.md`、`SOURCE.json`、`coverage.json`、`results.json`、`findings.json`、`commands.txt`、环境/工具链记录、原日志及必要输入/数据、支持与诊断 patches、`SHA256SUMS`。每个实际结果须有 UTC 起止、工作目录、精确命令、退出码/信号、外部 timeout、执行类型、GPU/driver/CUDA/OpenFOAM/flags、source/dependency/input/binary hashes、逐字段预期/实际与限定结论。

原版、正式修复版、支持修复版、诊断版和未执行项分列。保留 FAIL/BLOCKED/INCONCLUSIVE；单条日志只存一次，跨 topic 复用其 evidence ID。按 PROTOCOL 追加相关 `responses/R014-*.json`、`manifest.evidence` 和 `owned_paths`，运行行政验证后正常推送同一 PR。推送前 fetch 并核对远端，保留并整合他人提交，不 force push。

最后返回固定审查提交、远端验证结果、新目录链接、已关闭的精确范围、完整缺口和是否满足 FAST 验收。不要把源码检查、编译或诊断通过写成整 PR PASS；本任务没有合并权限。
