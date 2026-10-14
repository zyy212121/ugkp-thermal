# PR14 第二轮 FAST：正式修复与增量复验

PR14_REVIEW_ONLY。开发者源码修改、独立源码审查、真实编译与审查者实际 GPU 运行分别记录。本轮不合并 PR，不启动 PRODUCTION。

## 输入证据与范围

上轮实际被测生产为 802a11e7dfe986488a8bf8305e86424dcc94c9e1；证据交付为 ae5a59478023c74e23b64dc1a1f388fb805ef565。已读回报告、覆盖矩阵、发现、补丁及相应原始日志，并确认两个提交 common/applications 生产目录没有差异。

原86项通过、9项失败、14项诊断补丁通过、1项阻断、14项未执行，保留原限定范围；9项失败不等于9个生产缺陷。诊断通过不等于正式修复通过。本轮的生产修复提交和依赖哈希见 SOURCE.json，真实编译和未执行范围见 VALIDATION.json。

## 正式修复

### F001：native 化学 Trial 与 owner 状态处理

修复 nativeRefinement 参数未选择 Trial 的接线，并使 ChemistryFailure 的运输状态不再否决化学通道给出的内部重试。原 requested-interval owner 缩步机制及耗尽后的 public terminal 完成路径保留；CHMT 直接 false 路径不因此改为外部试探拒步。

实际无可消费模型/存储和可继续的数值诊断分别传递。尤其 Thermo InvalidModel 不能一概当作缺失资源。另修复局部迭代数到限后丢失此前失败反应编号的问题，否则缺失 reactants 会被错误转成普通 StepLimit。成功化学步骤的数值表达式不变；失败诊断传播改变。

原 P002 只证明单 kernel Trial 返回，不能证明完整 owner 缩步。新增夹具包括实际 GpuResidentStrict.cu、SharedGasTrialPolicy 和原 owner；完整运行交由 GPU 审查者。具体复验见 REVIEW_REQUEST.md。

### F002：逐面壁热惯量的初始化来源

CHT 新申请的逐面数组原先清零，consumer 看到非空指针后不会使用 scalar fallback。初始化现在一次性填入既有 scalar 墙体物性计算的热惯量，随后真实 coupled patch 覆盖各自的面。原初始化/指针发布状态继续管理复制失败及重试，重复 configure 不抹去已上传值。

没有修改接触传热算式，没有加正值/有限性停算检查。新增初始化阶段一次主机填充与 H2D，替代原一次 memset，不新增每步遍历。主机内存申请失败按现有真实资源错误通道报告。

未耦合面仍采用已有1800/710/100默认物性；旧7900/500/15 token仍未被读取，本修复不冒充该配置功能。coupled 面仍由固体 rho(T)、Cp(T)、kappa(T) 生产。具体复验见 REVIEW_REQUEST.md。

### F004：初次 float profile 的焓余量

保留已有 CHT 8/64节点 float profile 例外和原 cast，把初始化焓与实际存储焓的差加到已有 double carry。非零原 carry 不被清掉。FSH 未启用该 CHT 配对热量宏，原 double 路径不变。

这里仅修复首次表示损失；不修改物理下限、热量公式或验收阈值，不声称只靠表示一致就已验证墙/气全局收支。

### F005：Co 归约结果尾部

显式初始化 double+bool 结构的7字节尾部。原字段、max/OR、16字节布局和遍历保留。这是辅助传输问题，不是已证明的 Co/Js 物理值未初始化。

### 归约临时内存

实际 Nsight 记录新增100个 transform_reduce 与100个 reduce范围，每个有一次malloc/free。现改为 Backend 持有的可复用临时存储，保留两种 Thrust 归约、stream和同步返回语义。多个并存请求不共用活动内存；更大请求按需增长；销毁释放。没有新增容量停算上限。

固定规模稳定阶段不再反复申请临时存储是源码预期，实际分配数量、保留字节、峰值和吞吐待 GPU 测量。缓存会把临时内存保留到 Backend 销毁，必须同时报告这一成本。

原新增的六个 Co 前处理 kernel 保留：当前数组会被 RK、化学、壁面时序及几何准备覆盖或改变，尚无已经证明等价的复用方案。本轮不通过删掉必要准备来制造提速。具体复验见 REVIEW_REQUEST.md。

## F003：保留补偿时间累计，补录首次分歧

保存的实际 GPU CSV 差异只在每个窗口第50微步的交换热量出现。原普通 double 累加与当前补偿局部时间的末步 dt 差量，与这些热量差的相对量级一致。CLOCK_ANALYSIS.txt 列出十六进制数及计算关系。

这不是实际 GPU 首次分歧记录。提供 clock-trace 的只读原生诊断补丁生成器，记录真实 dt/时间/壁面采样/气体/交换包。不能由“更精细”或容差通过直接认定所有差异已解释。用户已确认采纳保留补偿累计时间的建议。生产时间运算本轮未改；本轮接受该时间运算方式的选择，不等于未经 GPU 定位就接受其他来源的差异。

## 测试支持

修正 checkpoint checker 和自身夹具的字段布局 fingerprint，使其与真实 FieldSchema 匹配；物理阈值和断言不变。冻结 face oracle 只补入旧 baseline 已存在的滑移定温逻辑，三应用之外的冻结函数不变。保留上轮失败和支持补丁证据，不把这些修正统计为生产缺陷修复。

## 验证边界

开发者进行源码审查、必要静态检查和实际编译，最终结果见 VALIDATION.json 和 logs。独立审查发现的失败诊断丢失问题已正式修复并再次检查。曾使用的局部串行诊断只用于定位，不计为 GPU 或应用数值验收。

没有 GPU 设备运行证据；本轮修复状态为待正式 GPU 复验，不能称 merge-ready。完整测试入口为 REVIEW_REQUEST.md：原红输入、受影响 Ns2/Ns10/共享前端、性能、F003 首分歧，以及上轮尚未执行的 FAST 缺口。有效且无依赖变化的原通过项复用；不重复全套运行。
