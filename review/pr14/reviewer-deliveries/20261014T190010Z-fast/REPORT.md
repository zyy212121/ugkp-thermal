# PR14 FAST4 独立真实 GPU 审查

PR14_REVIEW_ONLY / FAST。**证据交付完成不等于整个 FAST 通过：原正式化学夹具、旧 API 测试和 strict restart 仍红，14 个原 gap 全部逐项保留。未启动 PRODUCTION，不合并；无 README、PR Conversation/描述、正式源码/安装/原 checkpoint 改动。**

生产固定 `47034e4e0380fa5b84165b49207a95a9b187a52a`，tree `c932823a00b3ed2f647e22de5002f9b913962fee`。记录父 `55d3663830ada63077241c528e2c81b53976e2c4`；本次结果目录 `review/pr14/reviewer-deliveries/20261014T190010Z-fast`。记录提交由 Git 固定历史和推送后 remote verification 收据给出，不自哈希。

开发交付 `review/pr14/developer-deliveries/20261014T180000Z-fast` 在固定远端55d36638逐字节核验：416 tracked 文件、414 SHA256SUMS 项全部匹配；PUBLICATION.json 与校验单本身单列。139项源差分136存续/3删除及1264源码/blob/dependency SHA全部核实。此前 developer false 上传标记不被静默改写。

实际 RTX4060 Laptop/CUDA13.1.115/driver592.00/OpenFOAM10/g++11.4/WSL12GB+4GBswap。所有本轮命令无自动时间终止，30秒资源日志只用于观察；旧300/900/1800超时保留原分类。实际UTC/argv/cwd/exit/input/source/binary/object/资源哈希在results.json、commands.txt和逐命令spec/progress/resource-summary。执行 151 条命令，其中构建、host、GPU、sanitizer、诊断、分析和profiling分列；不加总重复的 PASS。

## 真实结果与正式红

|验收范围|结果与实际数据|局限|
|---|---|---|
|新源原flags构建|CHMT Ns2 O2 197.753s、Ns10 340.379s；gasUGKP/FSH/CHT完整原native build/link均PASS|编译不代替数值；所有产物分别哈希|
|正式化学原generator/O3/fmadtrue|GPU及memcheck均exit11，四原成功控制先PASS；kind7比较失败，8failed-cell后停止、尚无transaction标记；memcheck ERROR SUMMARY0|正式夹具FAIL，不能把零CUDA错误算全矩阵通过|
|S403独立支持夹具|16failed-cell、16failed-transaction、32五成员FAIL_STATUS、2独立public完整interval replays、16flags；四原控制保留；GPU与memcheck0均PASS|只修测试对照/库存，生产owner/kernel不改；原count1/maxRejected0对照count6/100错误保留。病态metadata NaN完成不称有效物理守恒|
|原Ns10 static/moving化学|原2e-5/700步完整；static initcheck0；每例1400实际half launch/11200cell-half/112000species行|实际首/次half与old/newV逐位配对；rhoE*V残差0且rhoE位值保留；移动最大species总量残差 2.062707545239843371724192397484611068165022906287083409069159681E-18kg，元素最大约2.13e-16mol|
|原thermal/wet完整gas库存|原.001/100micro/2窗口；实际最终faceflux×真实ledgerDt+真实macro packet+checkpoint M/E/3P/2species按原标准PASS|thermal窗口最大E残差1.367e-10J、P5.148e-15；wet E3.422e-11J、P4.169e-15。全系统film/pore/particle收支未闭合|
|SST RK1/2/3静态/移动|9真实GPU例，原8cell+新增64cell（每阶段8interior）；D402净/分解gross源、Euler/RK、lowRe壁面omega/floor核算PASS|所有正式原输出byteexact；墙温ZeroGradient→TFix0、owner slots−1的实际输入/调用链证明。floor全部未激活；weighted omega gross包含被约束/实际RHS被抑制的行，仅证明源代数，不是应用库存。selected数值壁owner、graph/rollback、完整macro constraints和三应用gross仍开放|
|合法熔化/凝固|原onecell/nonzero双向物理CSV与9d全字节相等；最终accepted geometry配对后gas M/E/P/species PASS|两个material solves含一次discarded gas candidate；初观测误双计压力的原红JSON保留；不称全material/film账通过|
|原蒸发完整端点|.001，319acceptedwindows，exit0，receipt 958.808s|GNUtime另记16:29.19，客体UTC/单调计时存在差异，不作速度推论。独立完整蒸发库存收支未执行；历史900s只到.00025/SIGFPE/tiny-dt保持|
|strict restart|thermal/wet起始重写checkpoint byteexact，但后续物理CSV各16位差：FAIL|thermal首exchange-2row0 -0x1.89370df15a40dp-10 vs -0x1.89370df15a408p-10；wetrow4 -0x1.479e94335e34ap-9 vs -0x1.479e94335e342p-9。CSV限定范围不取代旧wet27字段；不清cache/coldseed/改历史积分|
|原active16 cold2D|CHT最大9.095e-13K，FSH最大4.430e-10K；double均值+原physicalcarry/pT按1e-9K每输出/全16group PASS；旧对应native文件全部byteequal|FSH原double schema无floatcastcarry字段，S405依旧规则get0不是audit零。新cold1D/restart/多种配置顺序及paired全库存未完整复验|
|真实资源初始化恢复|S406当前原O3 CHT/FSH archives，22槽真实cudaMalloc错误2和copy错误21→只补缺失owner/retry/canonical值/reconfigure/destroy；cold mixed优先级copy retry PASS|不代替species/rhoD2H、free/sync/hostalloc/concurrency/完整owner/leak验收|
|真实双向V1端点|FSH旧→新/新→旧原cold2D heat/boundary/profile全输出bytes相同|旧frontend在9d交付验证producer4b，hash与新相同，backend确实不同。非零混合species和gasUGKP双向端点NOT_RUN，不能由reserved0关闭|
|原测试支持API|原C++14 reacting ALE两scheme真实编译失败，原native material probe引用filmSteps/budgetDelta/solidTargetSweeps编译失败|开发者需迁移物理测试，不恢复生产audit|

## 资源和计时

原thermal100同source/flags/输入，新的Nsight实际runtime：cudaMalloc140/free140、kernel2800、Memcpy11702、MemcpyAsync200、StreamSynchronize400、DeviceSynchronize400。当前六个Co前处理kernel保留；归约前两次warm-up后198次稳定归约kernel，无归约范围内重复Malloc/free，实际396stream sync/198D2H异步复制。这里只是固定8cell的实际观察，不外推更大规模或整求解器。

与runtime相关联的cudaMalloc动态tracked峰值22161B、结束0；64个Device Static模块符号memKind5/correlation0事件未归入这一计数，完整raw表保留。因此不是驱动峰值VRAM/完整泄漏证明。Nsight原输出与非profiling正式原输出逐文件byteexact。

5次非profiling整求解器receipt样本 [1.8781, 1.309514, 1.347627, 1.118088, 1.060305]s，median 1.309514s、sample SD 0.323255s、范围[1.060305,1.878100]s。原GNUtime、RSS、CPU、GPU数据另保留；启动/包装包含在receipt。未测同轮旧新求解器速度比，不用kernel时间宣称整体加速。新的470 thermal与802全CSV/checkpoint/几何逐字节比较尚未做；已有固定Co原机制/缓存证据只按原源范围引用。

## 源审查与支持边界

独立源码 reviewer 未发现新的确定生产算术缺陷。snapshot保留21+Ns物理cell项、六face mirrors、typed statuses/实际CUDA错误；SST13audit数组/6D2H、GasBudget producer/storage/copy/consumer删除未引入替代auditpass。rhoK/omega/sourceNumber、真实剪切/压力功/热相变/供体限幅、RetryTrial真实残差/adaptive h仍保留。V1保持25+2Ns/species25、reserved16..24和config36零，零槽不等于物理测量。

仍有FilmMath kineticPerArea/dissipation及withNormalProfile再计算存FilmAux.kineticEnergy，消费者仅validation/checkpoint，不能宣称用户要求的全部audit-only工作清理完成。实际profile速度/热/剪切/work/kinetic-advection物理项保留。reviewer没有正式生产修复。

S401只识别accepted元数据NOT_COLLECTED，物理阈值/schema规则不放宽；原checkerfloat解析失败保存。S403仅测试生成物；S404当前真正native checkpoint ASan/UBSan只验证所导出实际HostState字段，ModelConfig.physics是默认值，没导出topVelocity/particles，不能据此算全film kinetic/压力体积账。S405 FSH缺字段分析异常按原double schema规则修正，不重跑GPU。D401/D402只读test副本观察不用于性能；全部对应正式原output bytes一致。S407保留原Decimal半dt比较/RK阶段窗口分组红，改为真实double半dt位值和同microstepRK归属；S408按实际提交geometry排除discarded candidate。生产源不改，原失败、精确patch/旧源和全部hash保留。

D401全case Python分析signal11/exit139，RSS227340KiB，OOMcounter0；单独moving faulthandler分析exit0，根因未证明。D402 GCC11内部segmentation fault exit1保留，完全同source/flags复试159.460s成功，原failed102.786s；没有低opt/缩代码/缩输入/硬停换绿。不将历史137或当前signal11无依据称OOM。

## 14原gap与最终决定

coverage.json保留全部14原ID、原固定证据、受影响依赖和本轮新scope。全部为PARTIAL：新真实子范围有明确PASS，但没有把graph、selectedwall owner、三应用gross、radiation/empty-multi mapper、cold1D/restart、Js实际selecteddt、species/rho D2H/free/sync/concurrency、完整蒸发/film/pore/particle/material到几何、当前802/470 first-clock-difference等未执行项填绿。未执行原因是对应新观测器/原配对输入/消费故障或完整库存导出尚未实现，已有subset不能代替它们；不是以改端点/阈值避开失败。

旧support全系统反力专用账为**RETIRED_BY_USER**，历史失败保留，不计算新PASS或删除后残差0。真正gas/film/solid/pore/particle gross要求继续开放。missing6dd13b/8ed8a保留BLOCKED；旧正常physical成功、旧非法mesh ASan、取消的数值拒绝预期分别引用，不混入新原生GPU计数。compact CPU下载分析未实现，不计优化完成。

开发CPU film最终API三组件、17/8/25checkpoint实际字段对照、32saved-source物理/分配对照按原具体源限定引用。material16实际为15初轮PASS+1修正oracle单项PASS；4ULP系数是假设的测试预算，不是已证明严格端到端误差界。149hostSST测量/mutation不是GPU结果；315mirror/5entry syntax不是peer证明。

正式修复仍由开发者处理。当前不满足整体FAST验收/merge-ready。交付含REPORT/TEST_PLAN/results/coverage/findings/commands、环境/源/输入/二进制对象hash、原始GPU/host失败与成功日志、必要全部输出/逐点数据、diagnostic/support patches、immutable旧引用、SHA256SUMS及R014/manifest记录。大原始文本必要时lossless gzip，decompressed SHA另验证，原字节保留在test cache；没有把raw日志改成摘要。推送后固定记录提交/远端核验单独返回。
