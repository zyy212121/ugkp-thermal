# 设备推进实施记录

## 当前主线

CP1 FP32 已完成验收并成为后续 CHT 的唯一数值与效率基准。工作从 P4 的热求解效率开始，P4、P5 完成后进入 CP2。计划覆盖有界辐射迭代、完整热事件父图和 CHMT 能量交接。[修订后的八阶段计划](DEVICE_ADVANCE_THERMAL_PLAN_ZH.md)

冻结的 CP1 热事件使用 4 MPa、120 μm、固气质量比 0.43 的 case-current，启用 SST。一次宏观验收从交换 59 续算到 60。CHT 只验收 FP32，CHMT 使用小案例。旧时刻仅提供起始状态。CP1 验收时的同精度计算和诊断保存在[检查点报告](../.build/architecture/cp1/case-current/REPORT_ZH.md)。后续直接引用[已接受的 CP1 基线](../.build/architecture/cp1/case-current/baseline.json)。

## 已有实施记录

支线及后续全部阶段遵守 [error／warning 与有界求解契约](ERROR_WARNING_EXECUTION_POLICY_ZH.md)：格式信息只用于读取；实际数据不可用才报 error；有限未收敛候选报 warning 并发布，不缩步或换整套方法重算物理候选。必要算法内部保护保留固定总预算。

已接入的源码：传输 staging 按最后一次复制事件回收；固体设备视图查询不再等待；gas 完整步录入父条件 Graph，设备 CFL/时钟与目录选择；steady 的局部伪时间步、残差判定、样本累计及窗口稳定判定。FP32 保留独立数值类型、归约和比较运算顺序。

gas/steady 已增加不可变主机重启快照，在收集全部场值及粒子数据后预先提交下一计算段，CPU 随后写出上一状态。独立设备快照槽、复制流及后台写线程仍属后续输出收口，尚未完成。DOM 增加 enqueue/awaitCompletion/collect 契约；兼容 solve 包装仍等待，P4 再连接热事件父图。


