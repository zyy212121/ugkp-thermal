# 发布清单

## 已固定并独立验证

- 生产source commit：47034e4e0380fa5b84165b49207a95a9b187a52a
- 生产tree：c932823a00b3ed2f647e22de5002f9b913962fee
- 交付目录：review/pr14/developer-deliveries/20261014T180000Z-fast。180000Z只是目录标识，不是运行时刻。
- SOURCE139项与publisher实际baseline-to-source变更集合完全一致；136本地内容哈希/计算Git blob与其远端核实结果相同，3删除由该完整收据证明。
- 两mixtureTransport脚本原100755和内容保留，偶然mode-only变化排除。
- VALIDATION.source_commit/source_tree与SOURCE一致；原构建、测试与GPU缺口分列。
- REPORT、完整REVIEW_REQUEST、逐gap计划、finding响应、证据复用及所有小型原始证据已齐。
- film最终API、material10native物理hex和分次16case、root315mirrors/5entrychecks已纳入。4ULP明确为选定测试预算，不是严格全链证明。
- 没有.o/ELF二进制；大film raw stdout保留原证据路径并提供SHA索引。

## 仍由root/publisher填写或核实

1. 证据上传完成后，记录实际交付提交SHA及远端tip，核查以上目录和SHA256SUMS；SOURCE.remote_tip_commit当前为null，remote_publication_verified和remote_evidence_verified均为false。
2. 不得用交付提交SHA替换已被测的生产source commit。若后续生产依赖变化，重新确认具体受影响证据。
3. 按review协议完成行政注册/validator。若仅更新发布身份文档，重算SHA256SUMS并独立复核远端文件。
4. 原优化flags、GPU/memcheck、蒸发.001终点、strict restart、真实M/P/E/gross缺口及缺历史refs不因发布而关闭；support指标RETIRED_BY_USER，compact传输尚未实施。

本整理任务只准备文档和仓库交付目录；没有执行上传、提交、推送或合并。
