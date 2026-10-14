# PR14 第二轮 FAST 独立真实GPU报告

PR14_REVIEW_ONLY。**本轮不满足整体 merge-ready；不合并。** 原F001/F002/F004/F005的指定正式输入已定向复验，F003实际位差链条已定位；新发现既有wet零相变材料索引缺陷GPU-F006(P1)、两个提交夹具支持缺陷(P2)，多组动态cold/完整动量及其余缺口仍保留。诊断通过不是正式提交通过。

## 固定身份与范围

正式生产修复 `4b5c480b572f234ddecf5e89164fe59b0d2e9735`，tree `0fa2562f4c83394cdf94e9650de3b22c4fb4bb15`；本轮起点/开发者交付记录 `75cfd885c028fe80d011a8a081f8bed1f8aeb55e`。本报告所在提交由最终PR提交历史确定，不把commit写入自身。旧被测 `802a11e7dfe986488a8bf8305e86424dcc94c9e1`、旧证据 `ae5a59478023c74e23b64dc1a1f388fb805ef565`；原clock baseline `d6110c02745a8ad726208a426c4bb3a28233c1fd`。新目录 `review/pr14/reviewer-deliveries/20261014T072053Z-fast/`。

实际RTX4060 Laptop8GB/sm89、WSL Ubuntu22.04/OpenFOAM10、CUDA13.1、g++11.4。正式Ns2 CHMT/CHT/FSH及Ns10CHMT均完整原生构建和实际GPU运行；Ns10首次编译900s超时保留，计划追加serial1800s后实际成功。正式库/正式安装/生产结果/原始断点未修改；所有工作在授权test工作树。源、环境、object/binary hashes见SOURCE、versions、environment与metadata/source-hashes。

已读取开发者REPORT/SOURCE/VALIDATION/CLOCK_ANALYSIS、旧REPORT/coverage/results/findings、20文件源差分及各调用链、PROTOCOL；先行TEST_PLAN含复用表与所有范围扩展。独立源码席位仅只读审查，报告在metadata；本席位执行实际GPU。源审查不冒充运行验收。死机后114个完整执行记录日志hash全一致，无孤立spec；只做恢复GPU健康观察，未重跑整套，死机原因未知。

## 正式定向复验

|项目|实际结果/限定|
|---|---|
|F001/native Trial|S208真实requested4e-6先StepLimit后缩步：begin8/commit4/rollback4，四个约1e-6acceptedpart精确覆盖原区间；effort耗尽publicterminal对照完成。全部有语义state/audit/status/budget逐字段/位检查，原1e-6native/public成功对照与16backing/metadata/maxSteps1组合保留。S202/S203/S208只修测试host/device布局、前置Co与观察，未改生产CUDA。提交原fixture仍红；默认targetMaxCo实际0.5，原说明0已明确更正。|
|Ns10|正式原chemistry-full/control实际GPU+原checker，全部CSV/checkpoint bytes等于802；700微步非均匀ALE化学另有1400启动参数证据。full initcheck300s超时；原control17.91s initcheck0，不替代full sanitizer。|
|F002/F004|正式原cold2D、1Dwet100、2Drestart，不叠P007/P010；全部写时doublemean+doublecarry与pT按原1e-9K；真实FSHdouble1D/2D和restart均完成。四初始化温区/非零carry、configure/patch双次序、mixed/toggle/真实copyerror重试均实际CUDA。7900/500/15旧CHTtoken仍不消费，默认1800/710/100；FSH有效token单列。|
|多完整cold组|新S214正确16parcel/cell数据每时刻保留16，但无activecold2Dage，strictprofile/currentpT误差CHT2.076931505e-4K、FSH5.813134476e-5K超过1e-9。保留红，未据inactivecache数据关闭动态整组/完整pairedgas账。S211被旧初始文件覆盖为1、S213绑定错误为3均保留支持范围。|
|F005/Co|正式Ns2原thermal100microsteps initcheck0，所有原CSV/checkpoint/内部geometry bytes等于802(不是d611/clockdiag)。实际空/1/31/32/33/513/32768及NaN/max/OR，freshdata同owner容量cycle/第二owner，Ns2/Ns10 bitexact控制；真实OOM/retry和销毁。六个Co前处理保留。|
|restart/cache|正式与802从同本轮中点checkpoint恢复，全部10对文件byteexact；formal-full与formal-restart仍有gas/exchange/checkpoint严格差，accepted/material相同。说明不是新cache回归，第一处restart原因仍待定位，不按容差关闭。|

## actual clock/ALE/SST/资源

三版真实clock各21400hex记录。baseline与正式首次时钟差在window1 accepted微步6：`0x1.f75104d551d69p-15` → `0x1.f75104d551d6ap-15`；stage/sample时钟自7步不同；物理壁面值全部相同。首次dt、gas momentum、packet差在每window第50微步；802与正式全部21400trace bits相同。与旧double累加/补偿累计末步dt解释相符，用户已批准的补偿算式保持。未用容差把严格跨版差变为PASS。baseline observer三个checkpoint仅编译sourceFingerprint及checksum共66bytes不同；物理state payload及其它config/header bytes全一致，两checksum合法；原whole-byte FAIL保留。其它两版observer所有原输出bytes相同。

真实非均匀moving-GCL100微步、movingchemistry700微步全部两阶段GCL最大5.551115123125783e-17m3，小于原absoluteGeometry1e-14；所有1400Ns10实际kernel old/newvolume args逐位对应，half dt一致；诊断原输出bytes不变。旧CUDA入口观察器零记录FAIL保留，S206使用真实CUDA13 kernel handle接口纠正支持，额外D2H/IO不计效率。

实际CHT/FSH SST22 late/reconfigure/all22allocation realOOM/copyerror/retry/native初始化/销毁通过。SST的Euler/RK2/RK3 K/omega实际transport/source/constraint库存闭合；RK实际最终face/packet阶段权重分别1/2+1/2、1/6+1/6+2/3，原输出bytes相同。SST K残差最大约4.93e-16、omega约1.04e-14；质量/能量/recipient M/E通过。完整bulk-P严格核算残差约1.30e-12..8.03e-12未关闭，diagnosticfilm recipient-P约1.03e-13。具体逐点数据、原阈值、通过项和红分开，未改求解器门禁。

enabled Mie DO真实CHT/solid耦合与disabled对照完成，单coupled面+其它uncoupled实际场写出/profile通过。完整固体账累计全部GPU gas/particle/radiation墙热及实际sideWalls300K外部导热，最大残差3.639908641e-13/2.588730305e-13J，原coupler abs/rel1e-8通过。最初wall-only差5.6e-4J漏了约56W外部导热，原不完整账FAIL未删。此结果不是三个resident应用完整gas/particle/momentum库存通过。

## 新red与诊断

GPU-F006(P1)：CpuMaterialDriver.C:461对zero-phase热包无条件读`condensed[phaseCondensed].rho`，合法ThicknessAveraged/noMelting默认index=-1。wet heat/evap原版均SIGFPE(-8)，静态非法访问确认，精确信号PC尚无栈证明；已知OOB不重跑取数。源与802相同，属于既有缺陷新覆盖。D204仅test内zero-mass体积0不消费unselected backing，保留所有原失败、patch/source/originalGPUobject/binary hashes；heat-only同输入全区间完成、global/perrecipient M/E闭合，P仍严格红；evap同输入300s外部timeout。没有开启melting/改终点、物性、精度或阈值换绿。正式修复须开发者。

两个P2提交支持问题：managed host/device owner在本WSL异步host读取SIGSEGV；fixture强制两acceptedpart与真实CFL冲突。S208补全部semantic audit比较、原支持红保留。新S201–S214支持、D201–D204诊断各有源/补丁hash；旧P003/P008/P009/P012仅按原用途复用，旧诊断不算正式修复。

## 实际效率数据

同原thermal100步flags，Nsight相对802：cudaMalloc/free **339→140各**(各少199)，kernel **3000→3000**，memcpy11702、async200、streamSync400、deviceSync400均未减少。200Co归约NVTX ranges：首次transform有1malloc，后198稳定ranges **0malloc/free**。只跟踪cudaMalloc的peak live **22281B**、销毁后0，某些其它memory事件未全部解析，不能当全GPU/context峰值。所有原输出bytes等于802；六前处理kernel未删。

非profiling3warmup+7alternating样本全部保留、所有输出bytes等于802。802 median1.924639985s(range1.870003..2.229358,stdev0.12648)，formal median1.920505083s(range1.769701..3.424826,stdev0.59848)，ratio 0.99785160；**不能确认稳定整求解器加速**。不从kernel/allocator计数外推整solver。

## 覆盖/复用/尚未关闭

本轮221条已执行记录，command状态{'PASS': 168, 'INCONCLUSIVE': 15, 'FAIL': 35, 'BLOCKED': 3}；含编译/setup/环境/分析/支持红，**不当GPU数值PASS计数**。限定验收单元25，状态{'PASS': 18, 'FAIL': 4, 'BLOCKED': 2, 'NOT_RUN': 1}；跨topic引用不叠加、旧复用不加新run数量。33topic矩阵和14旧gap ID各逐项见coverage/results。没有wholePR全功能PASS声明。

chem owner、全部movingGCL/化学体积、actualclock、SST22及mixed配置/实际singlecoupler已有限定闭合；其余仍需：GPU-F006正式修复、化学fixture2P2维护、active完整cold组/FSH完整paired账、cold-startJs/historyCo新输入、empty/multiple-selectedmapper、grossSST模型/完整bulk-P、wet/evap/pore/melting/全库存、三resident全部stage/最终限幅flux/source/M/E/P账、codec之外hostASan/UBSan及其它实际free/sync/host资源恢复。Ns10full sanitizer和evap诊断分别300s阻断；未改变通过标准。

## 交付与审计

SOURCE/TEST_PLAN/coverage/results/findings/commands、environment/versions、完整原始logs/spec、必要新input/逐点data、全部support/diagnosticpatch、SHA256SUMS与复现入口齐备。已有不可变内容只引用固定commit/hash，见metadata/immutable-references；新执行provenance仍由独立新log/spec记录。原版/诊断/未执行/阻断/失败分开，正式生产源码无diff。

按PROTOCOL追加相关R014 reviewer/recheck history、manifest owned_paths/evidence；行政validator结果在logs。正常push前再次fetch核远端，若变化整合并保留他人提交；不force、不写PR Conversation/描述、不改README、不合并。最终record commit/remote verification由PR历史及本地publication-result给出。
