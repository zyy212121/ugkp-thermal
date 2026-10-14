# PR14 FAST4 完整 reviewer prompt

你是独立 reviewer，只做 PR14_REVIEW_ONLY / FAST。交付目标目录为 review/pr14/developer-deliveries/20261014T180000Z-fast（180000Z仅为目录标识，不代表执行时刻）。本轮正式生产源码为 `47034e4e0380fa5b84165b49207a95a9b187a52a`，tree `c932823a00b3ed2f647e22de5002f9b913962fee`；commit/tree对象可读、139项本地清单与publisher完整远端验证收据一致。请在执行前核对远端交付目录及证据清单；当前remote evidence验证仍为false，不应将未核实的交付上传当已完成。不得启动 PRODUCTION 或合并 PR。

仓库 https://github.com/zyy212121/ugkp-thermal ，PR https://github.com/zyy212121/ugkp-thermal/pull/14 ，分支 perf/chmt-first-round-20261013。历史第三轮证据位于基点0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668的review/pr14/reviewer-deliveries/20261014T120103Z-fast；该轮真实GPU源码9d5be6eff69dbced2fc959d422d3ba419a1bc39e，tree5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066。不要将历史支持patch或开发阶段binary算作最终提交验证。

## 授权与隔离

遵守 review/pr14/PROTOCOL.md 和用户规则。只在已授权隔离 test 副本、test 安装及新 evidence 目录工作，不覆盖正式源码、生产结果或原 checkpoint。既有test区 /home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test 应核实实际挂载、输出与符号链接。不得改 README、PR Conversation/描述、force push或合并；不得新增数值、精度、版本、契约、预检拒绝/rollback/停算门禁。不得放宽输入、终点、FP64、容差或采用CPU CUDA模拟代替GPU验证。

用户已取消wet单scalar方案，随后要求所有仅用于审计/检查且不影响实际守恒量计算的生产工作一概删除。不能要求恢复生产audit来满足测试；测试侧独立库存/flux/source测量可以保留。旧support全系统账指标状态为RETIRED_BY_USER，历史失败保存，删除字段后无残差不能算守恒PASS。真实gas/film/solid/pore/particle物理路径仍需独立验证。

## 明确授权与审查交付

用户已授权在本轮隔离test区完全覆盖、重构、修改和更新测试源码、配置及测试安装；必须保留既有证据，不改正式生产区、正式安装、生产结果或原checkpoint。为解除跨测试阻断，可以在隔离test副本实施最小诊断源码修复，并保存正式原版失败、精确patch及哈希、修复后的独立结果。支持修复后的成功只证明支持版，不能算正式固定源码通过，不能静默改后覆盖原失败。

最终审查结果必须集中到同一 `review/pr14/reviewer-deliveries/<UTC>-fast/` 文件夹，包含报告、完整命令/日志、必要输入与输出数据、诊断patch、source/input/binary哈希和coverage/gap状态。推送到同一PR14供开发者直接读取。此次结果推送已获用户授权，无需再次询问；不扩大为合并权限，不修改PR Conversation或README，不创建新认证凭据或扩大访问权限。

依赖未变且证据范围匹配的既有验收单元直接复用；全部原缺口及本轮受影响的必要测试须逐项覆盖或明确未完成原因，不能为省测试而缩输入、缩终点、删物理检查或把支持版冒充正式版。保持FAST范围，不启动生产长算。资源/用时估计用于调度和如实记录，不构成到时强制停止条件；不引入硬性预算停止、新数值门禁、拒绝边界或契约预检。

## 先核身份和依赖

读取 REPORT、SOURCE、VALIDATION、EVIDENCE_REUSE、GAP_RECHECK_PLAN，第三轮 findings/results/coverage及本轮新证据索引。使用以上固定源码，而不是可移动分支HEAD。建立新的 TEST_PLAN：列最终source/dependency/input/binary哈希、R014/topic/file/function目标、明确复用的acceptance unit与原局限。

本轮shared snapshot/storage、SST hooks、core chemistry、CHMT Backend、outer Budget、film/material报告和CHT codec已变化。SST-RK、gross三应用与core chemistry不能一概复用原结果；仅真正未受影响scope沿用。保留旧FAIL/BLOCKED/INCONCLUSIVE/ASan及缺历史ref记录，区分正常物理回归、非物理解传播和已取消拒绝预期；不要叠加重复命令的PASS总数。

核查删除producer→storage/transfer→consumer全链：无物理消费者、无新的pass/kernel/遍历/D2H替代；真实Co、rhoK/omega/sourceNumber、物理剪切/压力功/热相变、actual donor clamp、RetryTrial实际残差与自适应step变量保留。旧checkpoint兼容Budget/ChemistryAudit/support槽不进计算，未采集必须标明，reserved零不代表测得零。

## 正式构建与CUDA范围

先按项目原flags完成最终受影响native构建。开发低设备优化成功仅为真实CUDA编译/link证据，不能替代原O3/O2 codegen、GPU数值、sanitizer或性能。此前exit137只证明被kill，不能无OOM证据断言原因；阶段构建按对应哈希保存。

CHMT同源Ns2/Ns10及shared gas实际构建应对应最终common接口。保持C++14已有调用兼容，本轮if constexpr回归已恢复常量if；不得升级原测试标准掩盖。只做受影响依赖，不机械全库或生产长算。

## 必做：正式化学fixture完整增量

目标TEST-R3-NEGATIVE-SEMANTICS和R014-010/012/014/026。最终checkout直接使用其generator，不叠S301/S302，不换kernel或owner，不启host shim：

```sh
python3 tests/fixtures/chemistry_trial/prepare_chemistry_trial.py \
  "$CHECKOUT" "$NEW_REVIEW/fixtures/chemistry-formal" \
  --nvcc "$NVCC" --arch "$CUDA_ARCH"
"$NEW_REVIEW/fixtures/chemistry-formal/chemistry_trial"
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$NEW_REVIEW/fixtures/chemistry-formal/chemistry_trial"
```

记录实际机器、版本、绝对命令和哈希。generator原-O3与--fmad=true保持；不能用--Ofast-compile=max/ptxas O0宣布正式构建通过。既有成本只用于安排顺序与资源监测，不新增到时即停的硬性预算。真实资源错误如实保留并报告，不反复无依据重编或缩源码。

### 16负例与事务

保留16原backing/metadata/maximumSteps/reject/Newton组合，所有原输入和阈值。普通host owner与deviceState分离，显式上传、同步和真实CUDA参数；所有比较前同步。逐cell在public replay/rollback前比较并记录code、thermoCode、reaction、attemptedSteps、rejectedSteps。已删除的nonlinearIterations不能伪造为第六项。

- kind10保留StepLimit/reaction0；kind15零区间、thermo消费与未消费basis分类不变。
- kind7/15原失败先记录，后由新独立fixture执行原public interval，比较实际物理数组/库存，不让replay覆盖原失败状态。
- 全部实际policy cell/face、faceStatus、cell/typed status、transaction flags按位/语义比较；padding不充当语义。
- 测试独立计算实际V、rho*V、rhoE*V、3方向rhoU*V、各species*V及聚合库存，比较transaction/reference和rollback恢复。
- retry/terminal标记恢复符合原owner语义，正常rollback不poison。已删除GasBudget ledger/byte counts/ChemistryAudit输出不再是生产断言对象；审阅替代物理覆盖，不能称旧账字段通过。

核对16 failed-cell、16 failed-transaction、32 FAIL_STATUS，以及kind7/15两个独立public-replay成功标记，并审实际值。标记数不能替代断言检查。check/cudaCheck退出进程不等于一般allocation/free/sync失败恢复通过。

### 四原控制同次保留

1. requested4e-6、起点2、targetMaxCo=.5真实Co/refinement，通过实际accepted-duration trace在独立初态逐段reference重放。
2. maximumAttempts=1的public terminal完整原区间。
3. 原1e-6 real-Co requested owner成功。
4. 原1e-6 direct native/public成功。

不硬编码两段accepted，不改CFL、dt或终点。核心pure-output删除触及实际函数源码，因此成功物理和retry/terminal路径都必须新源覆盖。原S305 chronology只按仍相同owner逻辑范围复用，不把S304 stale currentDuration当拒绝dt。

## SST/几何与真实物理核算

本轮生产SST audit13数组/6D2H、sharedSST hooks和budget累加已删除。新验证在tests独立计算actual库存、Euler source/face贡献和RK1/2/3权重；至少覆盖原已相关静态/移动geometry、accepted/rollback、owner/graph与physical字段。开发149项独立测量及倍率mutation证据是host单元范围，不能当GPU执行。

三应用gross M/P/E/species原缺口保留。对实际修改链安排原短case的独立initial/final库存与真实flux/source核算；测量不得写进生产提交、dt或收敛门禁。必要时用测试侧观察器，保存正式原失败和支持patch哈希，不伪造无audit后的零残差。原容差不变。

## CHMT实际消费者、CHT及兼容

复验真实GasFilm/GasSolid/PoreGas/Particle动量、热、species与packet缩放；support反力专用force/profile删除应无物理反馈。两chemistry half以真实相应volume核算species与rhoE，并保留species/rho真实D2H故障和rollback。纯audit D2H故障已不存在，不要求恢复。

wire必须保持V1原25+2Ns列、species offset25；16..24九槽reserved0，config reservedAudit在原offset36零初始化。独立测试old→new/new→old端点的实际species、热量、边界资料，不能只验证新新。此前16列方案已撤回；不新增version gate。

CHT radiation只删重复计算/纯加总，实际能量传递与contact算术应位级相同。保留native before/after物理对照；OpenMP冷mesh cache测试初始化修正不算生产加速。原implicit_math拒绝预期/缺refs/旧非法mesh ASan等独立标注，不隐藏在正常物理PASS中。

## 蒸发原端点和重启

signed finite negative open-edge只删除普通路径旧拒绝，保留原owner extrapolation算术。复用native17原成功bitexact+8新ordinary等于旧terminal同dt的限定机制证据；源码后续去film报告/outer Budget又影响该链，应先受影响native film/curved/lifecycle和实际material owner检查，再原GPU短case至0.001；不新增900s或其他硬性停止预算。原900s仅.00025、300s tiny-dt观察及SIGFPE/timeout保留；不得缩终点、关闭蒸发或加新门禁换绿。

完整原失败history的gasTrace/gradient/traction/geometryEndpoint等尚未齐备，不能称窄replay是完整owner历史重放。若只补新观测，明确范围。

strict thermal/wet restart位差继续开放。只读复核支持保持成熟热恢复，不支持清cache/改cold seed/额外迭代；thermal1ULP不解释全部wet27差异。原严格标准不因误差小或审计删除而豁免。若最终改动触及对应restart依赖，则做最短受影响重启对照；否则复用固定证据及原FAIL。

## 交付结论要求

输出新TEST_PLAN、实际commands/UTC/exit/机器/input/source/binary hashes、完整logs、findings/results/coverage。按命令和独立acceptance unit报告，不给重复相加总PASS。支持补丁前后分别保存。所有未执行GPU、原flags构建、sanitizer、原endpoint或缺失输入均准确NOT_RUN/BLOCKED，不填PASS。

保留14 gap原ID和新依赖对应；support旧指标RETIRED_BY_USER，真实气相M/P/E和gross缺口不关闭。compact CPU下载分析仍未实现，不计优化完成。不启动PRODUCTION，不合并、不修改README或Conversation。

## 最终CPU证据使用说明

已补齐的film最终API三组件、17/8/25checkpoint实际字段对照、32组saved-source物理/分配对照可按对应source范围复用；missing6dd13b/8ed8a仍BLOCKED。Material16不同case是15初轮通过加1修正oracle单项通过，不得写成整组重跑绿；审核其新增4ULP安全预算仅用于测试且git-baseline也被同样核算；系数4是夹具预算假设，并非已证明的严格端到端误差上界，实际新旧hex比较不变。管理315mirrors与5entry checks仅standalone/host syntax，peer未比较。
