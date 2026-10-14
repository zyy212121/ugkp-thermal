# PR14 FAST4 开发交付

PR14_REVIEW_ONLY。本交付固定生产源码 `47034e4e0380fa5b84165b49207a95a9b187a52a`、tree `c932823a00b3ed2f647e22de5002f9b913962fee`，目录 `review/pr14/developer-deliveries/20261014T180000Z-fast`。139项变更的本地SHA/计算Git blob已独立核对publisher远端核实收据；交付证据远端上传尚未核实，不是整体 FAST 通过。最新用户要求删除所有仅供审计、对实际守恒量计算无意义的生产开销；已取消 wet 单 scalar 和可选 SST audit 模式。必要独立测量放在测试中。原物理输入、FP64、时间终点与验收容差保持。

## 当前结论

- 化学正式 fixture 已补全原16负例的真实 transaction/rollback/status验证，随后适配生产审计删除。五项实际 ChemistryStatus、cell/face、原真实Co和独立accepted-duration重放均保留。最终 C++14 兼容修正后的完整 native CUDA 低设备优化编译/link通过，288实际依赖before/after相同；不能替代原O3或GPU运行。
- signed open-edge 蒸发窄修复保留。17原成功逐位不变、8新增普通路径与旧terminal同dt逐位相等、native film/curved/lifecycle验证通过。原完整GPU蒸发0.001终点仍开放，原完整失败owner历史并未完整重放。
- 纯支撑反力/force、shared gas预算、SST生产审计链、CHMT外部化学审计与核心化学纯输出累计、CPU材料/film纯报告、outer Budget生产链已删除。真实求解控制、rhoK/rhoOmega、Co、species/rhoE、物理界面packet与heat/phase/pressurework路径保留。删除账不构成守恒通过。
- V1 wire最终保持25+2Ns及原species偏移；9旧audit位置写零reserved，configuration reservedAudit零初始化。此前16+2Ns方案已撤回，不能据旧描述验收。没有新增版本门禁。
- strict thermal/wet full/restart 位比较继续开放；只读复核不支持为此盲目修改成熟热恢复。无必要源码改动并不豁免严格标准。

## 版本与证据身份

基点0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668；第三轮真实GPU被测源码9d5be6eff69dbced2fc959d422d3ba419a1bc39e、tree5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066。本轮源码固定为`47034e4e0380fa5b84165b49207a95a9b187a52a`、tree`c932823a00b3ed2f647e22de5002f9b913962fee`；最终证据远端上传校验仍待发布者完成。

所有本轮编译按实际依赖before/after SHA识别，阶段与最终明确分开。其他CPU文件变化不能机械推翻未依赖它们的gasUGKP fixture编译，相关common变动则必须重编。文档不附.o或执行二进制，只附源、命令、日志、哈希和可重放输入。未修改README或Conversation，未合并，未启动PRODUCTION。

## 化学覆盖如何保留

正式fixture仍使用ordinary host owner与独立deviceState，显式上传和同步，包含完整生产CUDA owner。全部16原backing/metadata/step/reject/Newton组合、原起点2、Co=.5、requested4e-6、maximumAttempts=1终止控制和1e-6成功控制保留。

原失败在public replay和rollback前逐cell保存并比较code、thermoCode、reaction、attemptedSteps、rejectedSteps。nonlinearIterations已按用户要求从生产删除，不能伪造第六字段。比较全部实际policy cell/face数组、faceStatus、typed/cell status、transaction flags；独立从实际体积、rho、rhoE、三分量rhoU和species计算每cell与全域库存，验证rollback和独立reference。kind7/15仍有独立新fixture的public replay，比较实际状态，不用后来的成功status覆盖原失败。

原GasBudget transport/initialTransport/chemistrySpecies、host ledger/identity/path/deviceBytes/hostReadBytes与全部ChemistryAudit语义断言已退役，因为相应生产对象或输出工作删除；不是漏检后称通过。acceptedSteps/integratedTime/maxLocalError/nonlinearIterations不再宣称被生产观测。时间覆盖由真实owner accepted-duration trace与另行初始化的逐段物理重放证明；原direct kernel、真实Co owner、terminal全区间控制同次运行。

核心测试直接从实际mass、元素组成、thermo energy计算残差，保留Cantera/analytic输入与原容差。新旧d611原成功lineage用测试内旧API适配器比较实际物理十六进制与attempt/reject/status，不恢复生产audit。CHMT两half测试在测试翻译层围绕真实kernel前后读实际species×对应volume和rhoE，不保留capture钩子；实际species/rho D2H故障控制仍在。host CUDA shim结果仅是host单元证据。

## 删除范围和零物理消费者证明

详细producer→consumer审查及测试索引见evidence/shared_audit_removal/FINAL_STATUS.txt、support_removal、chmt_audit_removal和sst_test_migration。核心内部用于RetryTrial选择的质量/元素/能量残差与自适应step误差保留，只有纯输出重构/累计删除。此前if constexpr意外破坏既有C++14/Werror已被定向测试抓获并恢复常量if，未升级测试标准。

旧checkpoint Budget、ChemistryAudit或support等兼容槽仅按兼容格式保留，未采集，不进求解。零reserved不是守恒残差零。SST验证改由测试独立Euler库存/source/face和RK权重核算，实际source倍率mutation可被捕获。真实GasFilm/GasSolid/PoreGas/Particle动量与物理通量仍需独立验收。

## 验证状态与失败保留

命令级结果列在VALIDATION.json；不累加可能重叠阶段PASS数。原O3多次exit137及CHMT阶段构建被kill均保留，不直接断言OOM原因。最新源码原O3和真实GPU/memcheck尚未通过。最终核心19项、辅助化学61项、CHMT57项host测试各自通过；最终CHMT Ns2/Ns10 native低设备优化编译与正式fixture编译/link均通过，分别由独立receipt固定源码，仍未运行GPU。最后CPU film与material的完整证据补齐见下节；各命令范围分列，不叠加成全项目PASS。

5个辅助化学测试先在精确HEAD复现stale aggregate status旧失败，随后经owner契约核实拆分direct保留aggregate与owner-reset两例，各比较实际状态，61 PASS；旧失败未抹去。3个legacy lineage CPP虽已迁移API，但既有instrument.py的旧duration seam在HEAD与当前均阻断完整suite；不能宣称该suite通过。历史ASan/非法输入/失效拒绝预期与正常物理回归分列。

CHT OpenMP旧before2/12与after1/12失败不是优化改善证据。其冷mesh.cells首次并行缓存诊断只修测试初始化，生产算术不改；原implicit_math拒绝断言及缺历史ref等限制见独立报告。完整native测试与部分旧输入失败不得合并为整体验收通过。

## 重启与蒸发原开放项

只读restart证据evidence/restart_readonly_review：thermal400条paired postCo记录最大T差1.1368683772161603e-13K，约1ULP，p差2.9103830456733704e-11Pa；M/E/V和50条dt相同。wet独立27语义差异仍在，不能以thermal初值解释独占归因或统一说只有1ULP。无需盲目清缓存、改cold seed或额外迭代，strict FAIL保留。

原蒸发900s仅到.00025/.001，300s tiny-dt观察与旧SIGFPE/timeout均保留。finite negative open-edge拒绝删除只扩展普通signed owner-extrapolation，其他真实资源/donor/nonfinite检查保持。native17+8和组件通过证明窄修复，不证明原全失败history完整重放或GPU终点完成。完整gasTrace/gradient/traction/geometry/history输入缺项列于旧native_replay_scope。

旧wet全系统support账约-1.58e-4残差是RETIRED_BY_USER指标，既非物理修复也非PASS。真实气相动量/质量/能量、三应用gross独立库存与原未覆盖范围仍开放。

## 尚未实施的传输优化

evidence/transfer_readonly/report.md仅作只读分析。全域primitive/gradient/flux可考虑按实际interface compact，但尚未实施；owner全状态下载的替代涉及架构。不能把其估算字节数算作已完成优化或实测加速。

## 发布前待补

最终工作树patch/hash与最后host结果、CHMT Ns2/Ns10和正式fixture同源低opt receipts已收录；生产commit/tree已固定，证据远端核实及最新源码原O3/GPU验证仍分别标识。下一轮审查按REVIEW_REQUEST和GAP_RECHECK_PLAN只覆盖受影响链，但SST/RK、core chemistry和三应用gross生产依赖已变化，不能机械全部沿用第三轮。

## 最终本地CUDA收据

CHMT Ns2/Ns10于17:27:02–17:27:44串行编译，exit0，348项selected common/CHMT哈希前后一致，sm_80、CUDA13、-O0 --Ofast-compile=max -Xptxas=-O0。正式fixture于17:28:28–17:28:46完整编译/link，exit0、17.957s、最大childRSS777400KiB，sm_89/--fmad=true，288项实际common/gasUGKP/test依赖前后一致。generated源码SHA175760f07385e87ac21710772c96e1776d9df0c2fa1dc4eb9ba74c73181493d8。无GPU执行；原项目优化未通过，默认flags未改。见evidence/chmt_final_compile与chemistry_final_compile。

## 最终CPU与管理证据补齐

`evidence/film_final_api/migration.md`完整保留删除report后到删除FilmRateBudget/edgeFlux的阶段变化。最终API的film、curved、phase lifecycle三组件均通过；历史checkpoint的17原成功、8新增ordinary对旧terminal同dt、既有25 signed成功只比较实际film/filmAux binary64字段，全部相等。这仍是固定geometry/recorded forcing的CPU replay，不是完整GPU owner历史。

saved-before/final 32组size/order/reconstruction/sign及限定retry/endpoint见证的物理stdout逐字节相等。实际evaluate allocation15→13、advance40→36；24/48/96/192faces且edges=2faces时，evaluate节省3072/6144/12288/24576bytes，advance为其两倍。公式evaluate64*faces+32*edges按此Ns2类型大小验证。仅报告真实分配计数，不推导GPU速度提升。两份约3MB的相同raw stdout保留原证据目录并给SHA索引；源、比较脚本、计数stderr、完整小日志与replay输入已复制。不可取得的6dd13b/8ed8a不可变历史回归保持BLOCKED，这份saved-source对照不是它们的替代PASS。

最终material是16个不同pytest case分别取得通过：首组15 PASS/1 FAIL，只重跑修正oracle的cpu_material为1 PASS/10 deselected，绝非整组重跑全绿。10 native fixture各自用git-baseline/current依赖分别编译并在真实OF10执行，实际state/packet/storage/wall/donor/time和测试侧真实solve/RHS/film调用hex严格相等。首个新增边界oracle的绝对1e-3J在git-baseline自身失败：formation能量约-1.24e10J造成反算温度舍入，残差-0.001373291J。仅测试oracle选定每cell 4个存储能量ULP作为安全预算，再乘(1+边界导热/热容比)传播；系数4是夹具预算假设，尚无四次逐项舍入或严格端到端上界证明。既有物理hex比较未放松，生产未改；旧失败与单项修后日志均保留。完整报告、命令、输入/hash与逐case输出位于evidence/material_final。

`evidence/root_final_checks/receipt.json`记录两项exit0：managed inventory315 mirrors（standalone，peer comparison unavailable），以及5个actual entry adapter编译检查。只删除3个已经不存在的managed输入声明；不宣称另一流体仓已同步，也不把这些入口syntax检查称GPU运行。

补齐证据后再次检查：formal fixture288项输入仍一致；CHMT348项保守清单中仅CPU native测试oracle及测试helper两文件变化，生产common/Backend不引用它们，所有实际生产输入仍一致，故不无依据重编GPU。10对已复制material hex输出亦独立复核字节与SHA一致。

源码对象独立核查详见`evidence/published-source-object-verification.json`：136内容文件本地SHA256及Git blob与publisher已验证远端记录相符、3删除由publisher收据验证，基点到源码变更集合与SOURCE139项完全相同；两mixtureTransport脚本100755和内容保留。目录180000Z只作标识，命令UTC以各receipt为准。
