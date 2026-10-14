# PR #14 第二轮 FAST 审查与 GPU 测试申请

PR14_REVIEW_ONLY。本轮是修复后快速交互，不启动生产长算。固定生产提交见本目录 SOURCE.json；交付记录所在提交与生产修复提交分别记录。

## 环境与权限

- 仓库：https://github.com/zyy212121/ugkp-thermal
- PR：https://github.com/zyy212121/ugkp-thermal/pull/14
- 分支：perf/chmt-first-round-20261013
- 上轮被测生产：802a11e7dfe986488a8bf8305e86424dcc94c9e1。
- 上轮证据交付：ae5a59478023c74e23b64dc1a1f388fb805ef565，目录 review/pr14/reviewer-deliveries/20261013T182722Z-fast。
- 用户本地 test 开发区：/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test。该区允许完整重构、覆盖、更新源码、构建、安装和测试输入；保留本轮原始失败证据。正式库、正式安装、生产结果和原始断点不在这项权限内。
- 审查者负责客观源码审查和真实 GPU 证据。为越过测试阻断可在 test 内做最小诊断修复，但必须保存原失败、补丁、源和产物哈希、修后结果。诊断通过不得算成提交原版通过；正式生产修复由开发者完成。
- 可以把完整结果目录和诊断补丁推送到本 PR 分支，不 force push、不合并，不写 PR Conversation 评论或修改描述，不修改 README。
- 不新增因数值、精度、版本或误差阈值拒绝、回滚、停算的生产门禁。真实 CUDA、内存、IO、缺少实际消费资源的错误处理保留。测试断言、测试超时只用于外部观察，不写入求解器。

## 先读什么

1. 本目录 REPORT.md、SOURCE.json、VALIDATION.json、CLOCK_ANALYSIS.txt。
2. 上轮 REPORT.md、coverage.json、results.json、findings.json，对照原失败及诊断补丁范围。
3. 本轮源差分和每项调用链；review/pr14/PROTOCOL.md 说明记录格式。

先建立本轮依赖/证据复用表。有效且未受改动影响的已有结果直接复用，不重新做全套测试。原失败无需为计数重复制造；正式修复版本须对原输入复验。新增调用链或配置只有上轮未覆盖时才补测。任何超出下列范围的测试，先在计划里写明依赖变化或证据缺口原因。不得以 CPU mock、源码字符串检查、编译成功替代 GPU 数值验收。

## A. 化学 native 分步（F001）

- 检查 nativeRefinement=true 接到已有 Trial，public false 的成功表达式保持不变。
- 使用真实 requested-interval owner 和 SharedGasTrialPolicy，验证首次化学 StepLimit 后确实缩步、重试、最终完成整个原请求区间。单 kernel 返回 StepLimit 不构成通过。
- 覆盖内部尝试耗尽后的 terminal 完成，失败 Trial 不泄漏 state、audit 或 budget；普通成功控制及 public false 对照必须保留。
- 区分可继续的元数据/数值诊断和实际缺失 thermo/mechanism backing，包含缺失 reactants 及 maximumSteps=1 的组合。不得恢复整请求数值停算。
- 实际 Ns2 复验并补 Ns10 编译/受影响路径；原成功字段用原比较规则，不扩大容差。
- 新增真实 CUDA 夹具和命令以本轮 SOURCE/VALIDATION 及 tests 中对应 runner 为准。

## B. 冷壁物性与初次 profile（F002、F004）

- 原 cold2D 输入、1D wet100、2D restart 使用本轮正式版本复验，不再叠加 P007/P010。
- 每个输出时刻检查 profile 均值（double 累加）+ double carry 与 pT，沿用 1e-9 K 标准；原始 float cast 的差额必须补入既有 carry。
- 分别验证先 configure 后 patch、先 patch 后 configure、coupled/uncoupled 混合面、重复 configure/toggle，以及实际 copy 失败后的初始化重试。已上传 coupled 值不能被默认值覆盖。
- 未耦合面的物性来自现有 scalar 默认；7900/500/15 旧 token 没有因此变成有效配置。核验真正消费的物性，不能只看输入文本。
- 覆盖非零已有 carry、不同温度相区、多个完整粒子组；FSH 的真实 cold1D/2D double 模板以及 restart 是上轮缺口。
- paired gas/profile/wall ledger 用实际阶段权重和最终热量做短程收支；快照差或有限输出不代替完整收支。

## C. Co 填充区与临时分配（F005、效率）

- 正式 Ns2 100步 thermal：initcheck 必须不再出现原7字节尾部问题，所有原 CSV/checkpoint/几何输出与上轮被测生产 802a11e7 的相同 thermal 输出逐字节比较（不使用旧 d611 作为本项逐字节基准，不加时钟诊断）。
- 补 Ns10、合法单 cell、多 block 和适用的 empty reduction；max/OR 与 NaN 传播表达式不变。
- 缓存仅存 Thrust 临时内存，不存旧 Co 数值。验证同 owner 连续归约、容量增长、重复 window、restart/reconfigure、实际资源失败及销毁。
- 相同输入/步数/flags 下，Nsight 记录 warm-up 与稳定阶段 cudaMalloc/free、kernel、D2H、同步和活跃/峰值字节。预期固定规模归约临时分配不再每微步重复；不要把预期写为已测结果。
- 本轮保留六个当前 Co 前处理 kernel。它们消费的数组被 RK、化学、边界和几何更新影响，当前没有已证明等价的复用；不能为了计数删掉必要计算。
- 非 profiling 重复计时与资源统计分开。保留所有样本和离散程度，不用 kernel 提速外推整求解器。

## D. 时间运算差异（F003）

用户已确认保留补偿累计时间；当前生产时间算式未因这轮修改而改变。已有 CSV 把差异集中到每窗口第50微步；标量运算分析表明旧 double 累计和补偿累计的末步 dt 差量与之相符。这尚非实际 GPU 首分歧记录。

按 clock-trace/RUN.txt 在隔离源码生成只读诊断补丁，对原 baseline 和被测版本取得逐微步时间、dt、壁面采样、气体和交换包十六进制记录，并对正式修复版本进行受影响核对。确认诊断不改变原输出。诊断 I/O 不参与效率计时。若更早发生差异，保留第一处分歧并继续定位，不能用容差通过自动关闭。不要擅自恢复旧时间算式或同时改写历史积分。

## E. 测试支持修正

- checkpoint checker 的 schema fingerprint 与当前真实 FieldSchema 对齐，数值阈值不变。
- 三应用冻结 face oracle 补入 baseline 已有的滑移定温处理，其他冻结函数与断言不变。
- 上轮 P003/P008/P009/P012 的输入、checker 和序列化证据按原用途复用，不把支持补丁算生产修复。任何新增支持改动单独记录。

## F. 上轮尚未执行的 FAST 缺口

以上定向复验通过后，按依赖顺序补齐仍适用的缺口：
1. active SST22 配置/所有权/资源恢复，SST 与 RK 实际预算；
2. 专门 cold-start Js 与历史 Co 效率近似的输入变化；
3. 真实动网格 GCL 和两个化学阶段体积；
4. wet/film/烧蚀/产气及材料到几何的收支；
5. enabled radiation、empty/single/selected-face mapper；
6. host ASan/UBSan 和安全可控的资源分配/复制失败恢复；
7. gasUGKP/FSH/CHT 真实阶段权重、最终限幅通量、源和完整库存收支。

与 A/B 已覆盖的化学 owner、mixed wall、FSH cold 不重复记数。使用上轮 coverage 的原缺口 ID 逐项关闭或如实保留。优先小型有明确对照的输入；建议单例外部预算300秒，需增加时先记录原因。不得靠改终点、物性、精度、通过阈值或忽略异常来得到通过。

## 交付

将结果统一放到 review/pr14/reviewer-deliveries/<UTC时间>-fast/：REPORT.md、TEST_PLAN.md、coverage.json、results.json、findings.json、commands.txt、环境/源/二进制哈希、原始日志、必要输入与逐点数据、诊断补丁及 SHA256SUMS。原版、诊断版、未执行、阻断、失败分开；不要重复上传已有不可变证据，只引用其固定提交和哈希。

按 PROTOCOL 追加相关 R014 响应和 manifest owned_paths/evidence，执行文档校验后推送 PR #14。推送前核对远端变化，不覆盖他人提交。返回固定提交、目录、发现、实际误差/收支/资源结果和剩余缺口。不合并。
