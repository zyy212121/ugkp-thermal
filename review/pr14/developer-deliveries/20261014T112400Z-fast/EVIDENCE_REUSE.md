# FAST3 证据复用与依赖表

PR14_REVIEW_ONLY。历史证据固定在 `f5e6c5e2cbde2693f5fa811d874c200b07bbbc66` 的 `review/pr14/reviewer-deliveries/20261014T072053Z-fast/`；逐项原 verdict、test IDs、scope 和 limitations 以该目录 `results.json` 的 `acceptance_units` 为准。下面是复验决策，不是新 GPU 结果。执行前须以本轮最终 SOURCE 的真实 diff/dependency hashes 核实条件。

| 既有验收单元 | FAST3 处置 | 依赖与保留限制 |
|---|---|---|
| R2-A-owner-full-semantic | 复用其生产链限定证据；重跑修复后的提交夹具 | 生产 chemistry/Trial/owner 未变；S208 是测试支持版，不能给本轮正式 fixture 代验。新 fixture 须 16 组合、成功、terminal、全语义与 accepted trace。 |
| R2-A-submitted-fixture | 原 FAIL 永久保留；本轮正式 fixture 定向复验 | TEST-R2-S1/S2 修复触及 host/device 所有权、Co、对照重放与语义比较。 |
| R2-F-wet-formal；R2-F-wet-heat-diagnostic；R2-F-wet-evaporation | 原红、诊断限定通过和 300 s 阻断均保留；正式修复 fresh 复验 | CpuMaterialDriver phase-volume consumer 改变；同原 wet/evap 输入，不叠 D204。另补有效非零相变和 wet restart/continue 控制。诊断 M/E 不覆盖 P。 |
| R2-F-dry-ablation-scoped | 复用未变 gas/face 通量的限定证据；检查改动分支依赖 | 原 dry M/E/几何通过不证明新 wet 分支；新的有效非零 phase 控制覆盖改动成功路径。其严格 P 账另见 GAP 计划。若原 case 也实际进入改动分支，按消费关系补该条，不重跑整套。 |
| R2-A-Ns10-native；R2-C-Ns10-control-initcheck | 复用原全输入 ordinary 数值/字节及 control sanitizer 证据 | 没有新 chemical kernel 改动。与仍 BLOCKED 的 R2-C-Ns10-full-initcheck 分开；后者按原输入追加有限外部预算。 |
| R2-B-CHT-original-inputs；R2-B-FSH-double-restart；R2-B-init-config-copy | 复用限定通过 | CHT/FSH cold 初始化、profile/carry、owner/config/copy-retry 未变。不能覆盖新的 actual active 多完整组及完整 paired gas 账。 |
| R2-B-16-activegroups | 原严格 FAIL 与输入保留；补真正 active witness | S214 mobile/no-contact 的 inactive cache 与 active profile 语义不同；不是已确认生产缺陷，详见 GAP。 |
| R2-C-tail-whole-output；R2-C-cache-freshdata | 复用 Ns2 尾部/原输出、Ns2/Ns10 fresh data/operator 证据 | Backend/Co allocator/kernels 未变；未证明同 Backend 多 host 线程并发。新的其他共享依赖变化会使对应复用失效。 |
| R2-C-restart-vs802；R2-D-clock-lineage | 复用 cross-version restart 字节相同与已定位 clock 链 | 补偿时钟已获保留，生产公式未变。R2-C-restart-full-bitparity 的原 FAIL 仍开；只定位现有小输入的 full/restart 首差。 |
| R2-F-SST22-resource；R2-F-SST-RK-net-budget | 复用 owner/资源、net K/omega、实际阶段权重和 M/E 限定证据 | gross SST、完整 P、其它资源失败/并发仍开放。先复核现有真实 flux/packet 数据的算术归因，不重跑完整 SST 套件。 |
| R2-F-ALE-chemistry-volumes | 复用 | 原 100/700 microsteps 全阶段 GCL 和 1400 实际化学 volume 参数证据不因夹具维护失效。没有新改动依赖则无需重跑。 |
| R2-F-radiation-singlemapper | 复用 enabled/disabled、single/mixed、完整固体边界账限定证据 | empty/multiple-selected mapper、完整 gas/particle 库存仍未覆盖。不要把部分账扩大为三应用完整账。 |
| R2-F-host-codec-sanitizer | 复用指定 codec 路径 | 不能代替 wire/mapper/resource-owner 的 host sanitizer，也不是 GPU sanitizer。 |
| R2-performance-resource | 保留原实测结论，无独立依赖变化时不重做 profiling/计时 | allocation 339→140、stable 198 ranges 零重复申请是原证据；kernel 3000 未变，原大离散计时不支持稳定整求解器加速。不得把结论延伸到新增 wet workload。 |
| R2-NOT-RUN-cold-start-Js | 仍未关闭，补首步 oracle 和新输入 | 历史 Co 缓存 fresh-data 试验不等于 cold-start Js 的物理/调度证明。 |

第二轮从第一轮继承的 core/pressure/collision/pool 等单位，继续沿 `metadata/reused-evidence.json` 引用原不可变来源和原限定范围。不要重新上传这些字节，不将其统计为 FAST3 新 run。一个 raw artifact 可服务多个 topic，但只能按实际验收单元计算一次。

完整状态保留在原 coverage 的 14 个 gap ID；本轮复验只追加状态与真实证据，不覆盖旧结果。若最终代码差分超出本文假定的 CpuMaterialDriver 与 chemistry test support，审查者应在执行前更新依赖表并解释所需扩展。
