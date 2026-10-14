# PR14 第三轮 FAST 开发交付

PR14_REVIEW_ONLY。只处理第二轮新发现的正式源码/夹具问题，并安排必要增量复验。没有新 GPU 运行证据，不宣称整体 merge-ready，不启动 PRODUCTION，不合并。

## 固定输入与当前状态

- 第二轮证据提交：`f5e6c5e2cbde2693f5fa811d874c200b07bbbc66`，目录 `review/pr14/reviewer-deliveries/20261014T072053Z-fast/`。
- 第二轮实际正式源码：`4b5c480b572f234ddecf5e89164fe59b0d2e9735`，tree `0fa2562f4c83394cdf94e9650de3b22c4fb4bb15`。
- 本轮正式 source commit：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`，tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`；五个 source/test 文件与本地编译提交 `0643a7a40767b91896a87af9d06efb042082debd` 的 blob/SHA256 全同，完整 tree 也相同。发布核验见 `evidence/source_publication_verification.json`；编译/host 检查见 `VALIDATION.json`。
- 已读第二轮 REPORT/findings/coverage/results、独立源码审查和 PROTOCOL。18 个限定 PASS 单元及其限制保留；4 个 FAIL、2 个 BLOCKED、1 个 NOT_RUN 不重分类成通过。编译/setup/分析记录不与 GPU 验收计数混加。

## 正式源码修复：GPU-F006

第二轮在合法 wet / ThicknessAveraged / noMelting 热包路径发现零相变质量仍消费未选择的 `phaseCondensed=-1`。`CpuMaterialDriver.C` 的这段代码与旧 802 版本相同，属于既有缺陷被新覆盖触发。原 heat-only/evap SIGFPE 和静态非法消费是证据；没有精确信号 PC 的栈因果证明，不补造该结论。

本轮正式修改限定为：零 phase mass 对 recession/体积不作贡献，也不消费对应未选择材料；热量、力、动量和 film forcing 的其余链条继续执行。合法非零相变沿用原密度/porosity 表达式与原成功路径，不添加新的数值、index、版本或预检拒绝边界。

实际完整 native CPU driver 在相同最终 fixture 下：旧源码 unselected=-1 触发 bounds sanitizer；新源码 unselected、selected noMelting、nonzero melting 三项通过。零质量热包保留非零 heat/force、film forcing、reciprocal energy/momentum budget 与 support impulse；两个合法旧成功控制的 mass/heat/force/sweep hex 输出与修复后完全相同。非零 melting 经真实 coupled-recession solve 校验原 FP64 volume 表达式。该证据只证明 host-driver 范围。D204 只是上一轮隔离诊断，不能直接变成正式 PASS。新 GPU 申请要求同原 wet/evap 输入正式复验，另加有效材料的非零相变控制和 wet restart/continue；完整 P 与全库存仍须独立审计。

## 提交夹具修复：TEST-R2-S1、TEST-R2-S2

提交 fixture 已改为显式 host owner / device state 所有权和同步边界，以及真实 Co/CFL 生成的 accepted timeline。删除“两段 accepted”假设时，必须保留完整请求覆盖与逐 accepted dt 的真实对照，不降低验收标准。

夹具设计采用完整语义 state/audit/status/budget 比较，保留 16 失败配置，accepted 分段沿 native=true 路径按真实 dt 重放；原成功/public-false 对照和 effort 耗尽 terminal 单独保留。这里是代码覆盖范围，GPU 运行尚待审查者验证。生产化学算法、SharedGasTrialPolicy 和补偿时间公式不因维护夹具而改变。第二轮 S208 仍证明其限定生产路径；本轮提交夹具修复还须独立真实 GPU 执行。

## 编译与运行范围

已完成 Ns2/Ns10 CHMT native host 编译与最终应用链接，退出码均为 0；不变 CUDA `Backend.o` 的对象 SHA256 与既有编译记录相符，448 项不变 source/dependency region 与原正式版本核对一致。此范围是“增量 host 编译 + 完整最终链接”，不是全新 CUDA 源重编译，也不是应用执行。首次 OpenFOAM 环境初始化失败发生在 compiler 调用前，原日志单列保留；修正 shell sourcing 后才产生以上成功结果。

修订 chemistry fixture 已 fresh nvcc 编译/链接，退出码 0；保留 30 条既有 constexpr 和 2 条既有 unused-variable warnings。9 项 captured source 输入在编译后哈希不变。源生成支持检查 3 项通过；最终 targeted pytest 3 passed、2 CUDA cases skipped（默认 PATH 无 nvcc），显式 nvcc 编译结果单列，跳过不算 GPU 通过。

实际 host material 回归 3/3 通过，既有 material/film helper 检查通过；仅标记 CPU host-driver 范围，不冒充 CUDA/GPU 数值验收。早期测试支持失败、旧 source bounds-red 和全部最终日志分别保留。本轮没有 CPU CUDA shim 或应用模拟验收。执行命令、日志、退出码、构建产物与源哈希见 VALIDATION 及 evidence。

## 增量证据与剩余缺口

`EVIDENCE_REUSE.md` 给出有效历史证据和受影响复验范围；`GAP_RECHECK_PLAN.md` 保留全部尚未关闭项。特别注意：S214 的 16 粒子都仍为 mobile/no-contact，inactive profile 不能未经消费关系核实就按 active cold 不变量判定生产回归。原严格红和输入全保留，待审查者补充实际 active 完整组证据，不为 stale cache 改生产。

新增 `evidence/strict_momentum_reanalysis/` 用原 GPU 日志做离线精确浮点重算：三个命名 CHMT case 的 gas Px 逐阶段数据能重现 checkpoint 的原 double 累计账位模式，物理 final-flux 重算在原标准内；原累计账 5 个 FAIL 和 dry window1 PASS 均保留。没有新 GPU run，不覆盖 film/particle/global vector P 或其它应用。

`REVIEW_REQUEST.md` 是可独立交给新审查者的中文申请，包含仓库/PR/分支、test 开发区覆盖授权、受保护区域、三条首要证明、复用与补测规则、两项原 300 s timeout 的有限追加预算计划，以及同 PR 的一次性完整交付要求。
