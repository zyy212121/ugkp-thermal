# 界面组分数学求解修复与限制

本分项实现提交为 dccd75d5 和 9382937c7d1eb040de92f001d02fb311b2470baa；正式库、生产算例、远端均未改动。此处证据对应独立工作树中的三个头文件，精确哈希见 composition-evidence/tested-source-and-products.json。父任务集成后包含其它算子修改，应另以最终集成树编译、验证和记录完整产物来源。

## 结论及未满足项

当前方法扩大了真实原方程可认证解的覆盖，但正常唯一根 CPU 热路径仍比 ea 基线慢约 7–8 倍，未满足“在不损害求解效率的情况下恢复稳定性”。因此本项不能作为整体可合并、效率无损或生产 GPU 性能达标的证据。没有为了计时恢复旧候选算法、放宽真实残差目标或改变物理时间步。真实 GPU 运行和 sanitizer 由父任务执行；本分项只编译了设备 fixture，不能以此声称 GPU 数值或 race/sync 已验证。

## 方程及单一数学方法

保留原组分方程 F_i=J Y_i+a_i(Y_i-b_i)-Y_i Σ_j a_j(Y_j-b_j)-Js_i，a_i=p D_i/(T δ Rmix)，Rmix=ΣR_iY_i，以及 ΣY_i=1、0≤Y_i≤1。没有更换物性、扩散率、通量、物理 dt 或原方程 Jacobian。

在单纯形上将方程表示成实矩阵 M=u rᵀ+v 1ᵀ−diag(d) 的规范化非负特征向量问题。对标量特征值按扩散极点分区，用局部 Bernstein 系数和 Descartes 变号隔离实根；已经隔离单根的区间采用 Newton/括区中点保护。Ns2 使用同一多项式的解析二次根。多根、区间端点及零／重复扩散的极点特征空间均显式处理。极点自由度的最近 bulk 解由带质量、加权均值约束的凸投影决定，未枚举 2^Ns 活跃集。

每个输出都使用真实 compositionResidual 原方程和单纯形认证；多项式接近零、投影后有限或某个候选残差下降均不是收敛证明。根向量仅因舍入偏离单纯形时可产生一次有数学定义的欧氏单纯形投影候选，但只有原方程达到既有容差才能按该投影发布。真实候选的 Newton 精化使用完整原 Jacobian，属于同一标量求根方法的精化，不保留可选择 backup、物理缩步、重放或多套计算算法。

允许多个合法根时按与 suppliedBulk 质量分数的欧氏距离最小、相等固定遍历顺序选择。该确定数学政策不增加经验权重，但可能与旧 Newton 吸引域选出的合法根不同。宽扫描中既有不同合法根，不能将最大 Y 差解释为数值误差，也不能宣称唯一物理解。

正 bulk 且 M 的非对角严格为正时，Metzler/Perron 性质证明唯一非负规范化根。该结构条件成立才以 Collatz 行界缩小同一标量区间；行界含运算次数推导的 γ_n 向外误差界，不仅外扩一个 ULP。已认证该唯一根则停止剩余根遍历。零通量的精确 bulk 解也仅在真实原 F 已达原容差时提前返回。

## 数学试探与真实错误

compositionResidualPolicy<false> 和公开 compositionResidual 使用同一算术和公式，区别只在报告 sink。私有试探不写全局 fatal；最终状态用真实 p 再计算一次原 F，并从真实 finalF 刷新 actualResidual、Converged/AdmissibleApproximation 和 warning。没有复制 Ns10 的 7104 字节 PhysicsConfig，也没有临时更改共享诊断指针。缺失配置、非有限输入、真实最终物理错误仍走公开 producer。

固定诊断位置 3840（输入）、3841（算术）、3842（前端配置）各只有一个实际 producer，保留真实 component/value；3806 只用于有限规范化 warning。既有唯一 callsite 检查和公开 API 编译测试均通过，没有删除检查。最终集成树 catalogue 的行位置由父任务统一生成。

浮点乘、除、加在接近 MAX 的边界采用二进制尺度和指数判定，避免先执行会触发 OpenFOAM FPE 的算术再检查。真实 bad 值不被改写为物理有效量。独立审查发现的极端有限极点输入在 dccd75d5 中 public/private 均 SIGFPE；9382937c 通过同比正尺度构造仿射约束和安全 profile 乘加修复。这个输入未被证明存在物理解，回归只证明数学试探不会 trap、污染 fatal、改变 FPE mask 或在 false 时覆盖哨兵输出。

## 结果语义

Converged 表示真实原 F 达到本次既有物理容差。AdmissibleApproximation 仅表示真实数学过程得到合法单纯形、有限原残差状态，仍输出真实 residual warning；不能称收敛。全 D=0、J=0 而各 Js 相消非零的输入无法精确满足方程，本方法显式保留有限线性最小二乘近似与 warning，报告不将它称为舍入误差。

NoCertifiedCandidate 表示当前数值方法未认证到候选，保持输出哨兵并返回 false；它不是“已证明物理无根”，也不直接写 fatal 断言无解。FP64 Bernstein 系数舍入、重根和极端尺度不能提供任意输入所有实根完备性证明。外层 bool 合约仍可能导致该执行能力未完成路径停止；这是已知边界，未通过伪造 bulk/零值终端放行消除。

## 本轮验证

- 生产 fixture `python3 -m pytest -q validation/tests/test_interface_composition.py`：13 项通过，19.33 s，退出 0。所有真实 CPU 数学用例开启 FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW。
- 原方程制造输入，固定随机种子 74921，Ns2 与 Ns10 各 20,000：均 bad=0。严格目标是生产绝对 1e-12 加默认 relativeMass·Σ|Js|，不是 1e-6 或统一放宽目标；命令、退出码和源见 exact-goal-summary.json 与 manufactured-final20000.cpp。
- 两个 Ns 的严格零组分／纯组分边界各 6,000，均通过；还有 Ns1/2/3/10 的零／重复 D、常 R、弱／负／零总通量、极点根、浮点重根、非极点边界根、typed 状态和警告开关对照。
- 加、乘、除 MAX 边界反例在原 FPE 环境验证；真实非法 R、NaN、D=Inf 和不可表示总通量仍失败并保实际现场；私有试探不污染 fatal。
- 独立审查 fixture 由 interface 分项提交和运行，证据不能用本分项测试数代替；父任务另纳入其最终 7 项冻结红绿。
- `test_production_diagnostic_callsite_ids_are_unique` 与 `test_chmt_direct_api_compiles`：2 通过，退出 0。
- CUDA sm_89 的 Ns2/Ns10 真实 production helper fixture 均 nvcc 链接退出 0；没有在本分项运行 GPU。Ns2 kernel 栈 1168 B／211 寄存器，Ns10 栈 8512 B／255 寄存器，均无 spill。静态求根工作区仍有成本，不能把删除配置 copy 直接等同 GPU 提速。

## CPU 成本（同输入、同 g++ -O2、禁 FMA 收缩、循环内无 I/O）

每组 20,000 输入，累计仅 helper 调用时段；每次读取 steady_clock 有双方相同开销。以下是单轮主机测量，不是端到端或 GPU 性能证明。后置合法性检查和输入生成不计时。`good` 的此成本表使用共同 1e-8 残差指标；上文另外的严格生产目标扫描才是正确性证据。

|Ns/输入|ea 秒/合法数|9382937c 秒/合法数|耗时比|
|---|---:|---:|---:|
|2 宽参数|0.005380 / 19996|0.023034 / 20000|4.28|
|2 正常唯一根|0.002512 / 20000|0.018588 / 20000|7.40|
|2 弱通量|0.002160 / 20000|0.016705 / 20000|7.73|
|10 宽参数|0.333180 / 14452|3.407438 / 20000|10.23|
|10 正常唯一根|0.018657 / 20000|0.143565 / 20000|7.70|
|10 弱通量|0.018878 / 20000|0.141301 / 20000|7.49|

正常唯一根值的差异达到舍入／既有容差量级；宽参数根分支可不同。旧实现宽参数较快同时有大量残差不合格结果；不能用该事实掩盖新方法正常路径的 7–8 倍代价。按父任务要求本轮停止方法研究，将性能限制明确交付。

## 可复查证据和来源

`composition-evidence/` 保存原始关键日志、完整编译参数、退出码、原始 PR 头文件 provenance、产品和源码 SHA256。完整生成式输入和 GPU fixture 在 validation/tests/fixtures/interface_composition/，不复制大随机数据。pole red source manifest 对应冻结 dccd 源；首次未冻结草稿的探索性失败保留在外部目录，未倒推归到某个提交。

本报告不覆盖最终集成树完整 OF10/CUDA 构建、生产二进制或设备算例；由父任务按最终树单独记录。

复跑入口（只写指定输出目录，不修改源码或运行 GPU）：

```sh
python3 docs/diagnostics/math-followup-20261018/geometry/reproduce-composition.py "$PWD" --output /tmp/pr19-composition-reproduce --cuda-compile
```

脚本会保留所有完整命令、退出码、原始 stdout/stderr、严格生产目标生成输入、源码哈希和离线 ea 基线 overlay。
