# FAST4 最小增量复验与原 gap 保留

原14个 gap ID 逐项保留。下列“当前状态”是第三轮固定证据的原verdict，开发端局部native通过没有自动改变它。新增复验只由实际差分依赖触发，不是全套重新排期。

| 原 gap ID | 当前状态 | FAST4 处置 |
|---|---|---|
| SST22-owner-graph | CLOSED_SCOPED | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| SST-RK-budgets | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| moving-GCL | CLOSED | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| cold-start-Js | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| chem-native-full-retry-owner | CLOSED_SCOPED | 保留已关闭限定范围；补新正式16负例完整断言的O3/GPU/memcheck，原四控制同次执行。 |
| wet-ablation-material | PARTIAL_BLOCKED | 蒸发窄fix只补实际受影响film链与原.001终点；wet单scalar待批准；其他材料/孔隙范围不改绿。 |
| CHT-radiation-mapper | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| mixed-wall-coupler | CLOSED_SCOPED | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| FSH-cold-profile | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| host-sanitizers | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| allocation-fault-lifecycle | PARTIAL | 沿用原限定证据及未覆盖范围；最终依赖未变则不重跑，本项不因本轮局部修复自动关闭。 |
| resident-full-budgets | FAIL_OPEN | 完整wet物理P保持FAIL；设计审查不算实现。其他三应用gross库存不自动加入本轮。 |
| native-FP64-clock-lineage | CLOSED_SCOPED | 既有批准clock范围复用；strict full/restart红项分列，无必要时不改成熟热恢复。 |
| developer-formal-recheck | PARTIAL | 本轮化学正式新增断言及蒸发最终应用/原终点尚待；wet批准与实现另列。 |

## 新增和附加开放项

1. 正式chemistry：16原配置、32完整status、16native failure、16transaction和kind7/15分离public replay；原O3 fresh build、完整GPU运行、针对同binary的memcheck。项目原数值参数与四成功控制不变。
2. 蒸发：原finite signed open-edge窄差分已应用，native17+8等价及原三组件PASS。最终CHMT受影响host编译/应用链接、原短输入.001 GPU终点与受影响控制仍待。完整原history缺项不补造。
3. wet完整物理P：一scalarEOS压力时间积分设计已完成独立数学/路径复核，但接口审批/实现/原GPU验收均未完成。不得使用旧56/64B方案或提前关闭。
4. strictthermal/wetrestartbitparity：严格FAIL保留；本轮只读审查建议无必要源码改动。是否改变严格标准由用户决定，不从误差小推导豁免。
5. actualfree/sync/hostallocationconcurrentfailure：原NOT_RUN/部分范围照旧；fixture错误退出不等于一般资源恢复。

## 执行顺序与有限预算

发布身份齐备后，先依赖复核和原O3构建，再chemistry GPU与memcheck，再受影响film控制和原蒸发终点；单GPU顺序执行。拟定外部预算：chemistry build600s、fixture300s、memcheck300s；原蒸发终点900s。全部是拟定观察预算，不是已执行结果或新增生产门禁；遇阻保存事实，不自动扩大为长算。wet只有明确授权和最终差分后才能添加实施/验收任务。

不重跑生产长算，不缩输入/终点，不放宽原比较规则，不新增失败边界换绿。所有旧FAIL/BLOCKED/INCONCLUSIVE原证据保留。
