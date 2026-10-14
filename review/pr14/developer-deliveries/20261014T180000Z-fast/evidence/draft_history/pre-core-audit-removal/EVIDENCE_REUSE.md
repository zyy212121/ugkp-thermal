# FAST4 证据复用与边界

固定第三轮证据提交 `0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668`；真正被测源码 `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`。本表是增量计划，不把历史 PASS 算成新运行。每个固定引用的 SHA256/Git blob 及原记录保存在 `evidence/reused-evidence-index.json`。

只有最终相关 source/dependency、input、运行语义未变，才沿用原限定范围。原已验证增量为两个化学测试文件和 CpuFilmDriver/对应测试；用户随后授权的纯支撑记账、SST与CHMT对外化学审计彻底删除正在修改更多共享文件，最终差分和依赖身份尚未冻结。最终提交若扩大差分，应基于具体消费者重新判断，不自动跑全套。

| Acceptance unit | 原 verdict | 本轮处置 |
|---|---|---|
| R3-formal-chemistry-fixture | PASS | HISTORICAL_SUPPORT_PLUS_NEW_FORMAL_FIXTURE_REQUIRED；New submitted assertions and shared comparators require original-O3 build and complete real-GPU fixture; S302 cannot substitute. |
| R3-chemistry-full-failed-transactions | PASS | HISTORICAL_SUPPORT_PLUS_NEW_FORMAL_FIXTURE_REQUIRED；New submitted assertions and shared comparators require original-O3 build and complete real-GPU fixture; S302 cannot substitute. |
| R3-chemistry-actual-owner-entry | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain S305 true native-entry chronology; do not use stale S304 currentDuration. Final unchanged owner dependency must be checked. |
| R3-wet-formal-zero-phase-ME | PASS | RETAIN_HISTORICAL_PASS_RECHECK_AFFECTED_FILM_CONTROL；Ordinary open-edge production consumer changed; run the affected original short wet control without claiming full P closed. |
| R3-wet-actual-final-flux-gas-MEP | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-wet-full-physical-P | FAIL | RETAIN_HISTORICAL_FAIL_RETIRE_ACCOUNTING_ONLY_REPAIR_TARGET；User cancelled pressure-integral proposal and authorized deletion of physically unused support accounting. No conservation PASS follows; genuine gas P/M/E gaps remain. |
| R3-wet-restart-strict-bitparity | FAIL | KEEP_FAIL_NO_UNNECESSARY_RESTART_SOURCE_CHANGE；Strict equality remains open; later changes only trigger dependency-relevant checks. |
| R3-wet-evaporation-original-900 | BLOCKED | KEEP_BLOCKED_AND_RECHECK_CHANGED_NATIVE_OWNER_CHAIN；Signed open-edge change has native CPU evidence; original GPU endpoint .001 and full application link remain outstanding. |
| R3-nonzero-phase-both-directions | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain successful original control proof; repeat only exact legal controls whose consumer is demonstrably changed by final diff. |
| R3-Ns10-full-original-initcheck | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-active-cold-fullgroups-CHT-FSH | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-thermal-restart-first-currentCo-cause | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-cold-start-Js-ConstantTransport | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-host-codec-mapper-wire-sanitizers | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-prior-strict-P-arithmetic-reanalysis | PASS | REUSE_EXACT_HISTORICAL_SCOPE_IF_FINAL_DEPENDENCIES_UNCHANGED；Retain original limitations; no new execution counted. |
| R3-gross-SST-resident-complete-budgets | NOT_RUN | KEEP_NOT_RUN_NO_AUTOMATIC_UNRELATED_CAMPAIGN；Retain original limitations; no new execution counted. |
| R3-GPU-empty-multi-selected-mapper | NOT_RUN | KEEP_NOT_RUN_NO_AUTOMATIC_UNRELATED_CAMPAIGN；Retain original limitations; no new execution counted. |
| R3-pore-material-to-geometry | NOT_RUN | KEEP_NOT_RUN_NO_AUTOMATIC_UNRELATED_CAMPAIGN；Retain original limitations; no new execution counted. |
| R3-allocation-free-sync-host-concurrency | NOT_RUN | KEEP_NOT_RUN_NO_AUTOMATIC_UNRELATED_CAMPAIGN；Retain original limitations; no new execution counted. |

## 原支持与正式运行的区别

- S302 完整16负例/32状态支持版真实 GPU 已通过，保留为正式化断言的依据；修订正式源码必须 fresh O3 构建并执行，不能覆盖历史日志或拿支持版代验。
- S305 是真实 native entry 的 rejected-duration 时序；S304 stale currentDuration 保留为旧部分观察，不作正确拒绝时长引用。
- 新 comparator 与新增transaction范围随同一次完整 fixture 运行检验；四种 owner/direct 成功控制保留。
- 蒸发 host native 的17+8等价比较与三组件通过，不是全 GPU interval owner 或原 .001 终点证明。
- thermal/wet严格restart、旧累计P、完整wet物理P和旧蒸发SIGFPE/timeout原verdict保留。没有整体FAST通过或merge-ready。

## 复制证据的可移植性

复制件保留原始命令、绝对路径、历史中间receipt，不改写原事实。`compile_receipt.json` 是化学最终汇总，`attempts-so-far.json` 仅是先前三次失败快照；`chemistry-evaporation-pre-accounting-removal.patch` 仅记录后来支撑删除开始前的四文件差分，最终完整patch待补；蒸发 `FROZEN_CANDIDATE_RECEIPT.json` 先于应用，应用后的事实以 `shared_application_receipt.json` 和三组件receipt为准。各原始子目录的旧SHA清单可能包含原目录外依赖，整体复制集以本稿根 `SHA256SUMS` 为准；原引用仍可按固定Git或来源目录校验。

wet原单scalar设计与审查已移至 `evidence/wet_design_superseded/`，仅是已取消历史方案，不能继续实施。当前是已授权的零物理消费者生产审计彻底删除；CHMT_ENABLE_SST_AUDIT方案已取消，不留生产audit模式。源码、最终生产构建及原生验证证据尚待交付。`evidence/CURRENT_SCOPE.json` 固定此次范围变更；旧草稿身份/校验保存在draft_history，仅用于追溯，不是当前源码收据。
