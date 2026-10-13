# PR14 developer/reviewer evidence index

PR14_REVIEW_ONLY · Current interaction: **FAST**.

This folder is the primary review record, including review requests for already completed first-round, B1–B7, and tool changes. Developer source/compile evidence is distinct from pending real-GPU acceptance. **Production source is frozen and its preclosure delivery tree is remotely verified. Later closure metadata publication, real-GPU acceptance and any merge remain separate.**

## Start here

1. Read [PROTOCOL.md](PROTOCOL.md) for roles, FAST/PRODUCTION scope, objective evidence, repair/recheck and archival rules.
2. Open [manifest.json](manifest.json): stable R014 topic list, shared E014 evidence, exact owned paths, final identity fields and pending status.
3. Use [source-index.json](source-index.json) to find every changed path, base/head blob and current SHA256. Production file/function sidecars are under [files](files). Their function/scope locators are supplementary to immutable blob/hash identity.
4. For each topic under [topics](topics), read developer evidence and the request, then fill the same ID under [responses](responses). Results, developer replies and rechecks append there; do not overwrite earlier failed attempts.
5. Inspect [pending-findings.json](pending-findings.json) and both whole-PR review snapshots linked from the manifest. A historical local approval does not close later dependencies or global findings.
6. Check [object-hashes.json](object-hashes.json) and shared [evidence](evidence): actual compile commands/logs/receipts and immutable report snapshots. Full solver binaries are identified by hashes; historical fixture binaries may be retained as scoped immutable evidence. Byte-exact raw evidence is PR14_REVIEW_ONLY by path and manifest.

## Scope and honest limits

- All requests are FAST: real GPU operator/small-case, valid-case conservation, actual calls/interactions/coupling, restart and relevant concurrency checks. Explicit input/version/criteria/budget comes before execution. No default long production run.
- Developer responsibilities are source review and actual compilation. No further developer CPU-shim/application numerical/parity runs. Old CPU/OF numerical results remain historical; newer unexecuted standalone fixtures are not current validation.
- Reviewer supplies objective GPU evidence; developer makes formal fixes under the same ID; reviewer rechecks. Reviewer diagnostic patches stay isolated as evidence. See protocol for exact result fields.
- Production source is frozen at `a73b9635510fb09d7baa5339e14d1aa609f0ff87` with tree `e343cb9f477dcd86556f7b6595189fa2c9c1c099`. Final production source identity and scoped/global source-review outcomes are recorded in [publication.json](publication.json); the exact preclosure delivery is remotely verified, with explicit CI observations below; the later closure-containing commit and GPU acceptance remain pending. Historical logs may refer to earlier snapshots; their applicability must be checked, not presumed.
- Allowed physical float: CHT cold1D[8]/cold2D[64] profiles and actual copies/publication arithmetic only. Other physical state, including thermal enthalpy residual, stays double. Timing API float and old-format decoding are separate categories.
- Original-success FP64 expression/order/clamp/fallback/inner-retry preservation is separate from formerly rejected input completion and explicitly approved physical corrections. Never add a new computational gate to make a test pass.

## Administrative checks and future removal

Run `python3 review/pr14/validate_records.py` for document/evidence consistency and `python3 review/pr14/cleanup_review.py` for a nonmutating removal dry-run. These are administrative checks only, not computation preconditions. No build/test/startup integration is added.

Do not clean up now. [CLEANUP.md](CLEANUP.md) describes the archive-first, explicit-ownership procedure. No production marker comments are currently inserted or authorized for removal. [marker-proposals.json](marker-proposals.json) is an optional future mapping, not a requested production edit. README files are never changed by this system.

<!-- PR14_TOPIC_INDEX -->
## Stable topic entrypoints

- [R014-001 · 最终身份与远端完整性](topics/R014-001.md)
- [R014-002 · 历史对象与固定源复现](topics/R014-002.md)
- [R014-003 · 材料 effusivity predictor film surface conduction 事务](topics/R014-003.md)
- [R014-004 · SST22 canonical owner与原重复配置](topics/R014-004.md)
- [R014-005 · wire compact restart实际传输](topics/R014-005.md)
- [R014-006 · 原生输入 token dictionaries material mechanism](topics/R014-006.md)
- [R014-007 · EOS 反演 primitive](topics/R014-007.md)
- [R014-008 · geometry boundary reconstruction diffusion periodic positivity](topics/R014-008.md)
- [R014-009 · cell species SST RK audit Co](topics/R014-009.md)
- [R014-010 · rates Jacobian RHS chemistry两半步](topics/R014-010.md)
- [R014-011 · wall analytic BVP exchange packet](topics/R014-011.md)
- [R014-012 · allocation reconfigure import owner graph free错误](topics/R014-012.md)
- [R014-013 · 最终native status和terminal consumer](topics/R014-013.md)
- [R014-014 · CHMT调度 两half HostState rollback](topics/R014-014.md)
- [R014-015 · 冷启动Js初始化与有效历史Co效率近似](topics/R014-015.md)
- [R014-016 · 状态工作删除的实际成本](topics/R014-016.md)
- [R014-017 · field contract与scheduling migration工具](topics/R014-017.md)
- [R014-018 · CHT FSH terminal ABI Create Co Advance mesh](topics/R014-018.md)
- [R014-019 · CHT FSH restart O(N×88)准入扫描](topics/R014-019.md)
- [R014-020 · CHT radiation solid terminal消费者](topics/R014-020.md)
- [R014-021 · 第一轮全PR效率 拒绝 FP32审查](topics/R014-021.md)
- [R014-022 · 第二轮独立全PR审查](topics/R014-022.md)
- [R014-023 · 全tracked物理FP64与唯一profile例外](topics/R014-023.md)
- [R014-024 · 完整生产应用 build link nm](topics/R014-024.md)
- [R014-025 · 真实nvcc小TU编译证据](topics/R014-025.md)
- [R014-026 · 实际GPU数值与异步错误](topics/R014-026.md)
- [R014-027 · 实际GPU及端到端性能](topics/R014-027.md)
- [R014-028 · GPU与host sanitizer](topics/R014-028.md)
- [R014-029 · 全物理生产链与输出重启继续](topics/R014-029.md)
- [R014-030 · 原生修复至merge-ready与授权](topics/R014-030.md)
- [R014-031 · common collision drag contact cold pressure packing reduction payload](topics/R014-031.md)
- [R014-032 · 未常用frontend material adapter额外候选](topics/R014-032.md)
- [R014-033 · Cold 1D/2D actual profile, ledger and publication](topics/R014-033.md)

## Final source and ongoing build

Production source: `a73b9635510fb09d7baa5339e14d1aa609f0ff87`; tree `e343cb9f477dcd86556f7b6595189fa2c9c1c099`. The preclosure delivery is remotely verified; publication of the later closure metadata is pending. See [publication.json](publication.json), [full-build-status.json](full-build-status.json), and [pending-findings.json](pending-findings.json) for separate source-only closure versus GPU acceptance. CHMT/CHT/FSH native build/link receipts are recorded at a164; gasUGKP stages/artifacts completed with wrapper 2 and child exit unrecorded. Final codec C++14 host rebuild/link and 848-record independent qualification are recorded at 970. All GPU execution remains pending.

Final review lineage: a164 category-level global review plus the independently approved 970 codec and a73 mapper deltas. Historical receipts retain their original source identities and caveats. No runtime test or cleanup has been run.

## Latest mapper and history delta

Final source is `a73b9635510fb09d7baa5339e14d1aa609f0ff87`. Review lineage is a164 category-level review, then the separately approved 970 codec delta and a73 mapper consumed-access delta. Exact mapper SHA256 `947decef7e1a2aa552c6d201401674617fdb8be22a331d98c430fe115ee63074`; see [R014-020](topics/R014-020.md), [R014-022](topics/R014-022.md), and [R014-024](topics/R014-024.md). Only the mapper host object and CHT executable were rebuilt for this final delta; native CUDA archive was reused byte-identically. Earlier full builds and codec receipts retain their original identities and caveats, including gasUGKP wrapper 2/child exit unrecorded. Real-GPU acceptance remains pending.

The continuation bundle now embeds source `a73b9635510fb09d7baa5339e14d1aa609f0ff87`, SHA256 `bf69f1b9f4a4c96d27af04bd885296264305393f941ea1db3a81cb4ea4eac5e8`. Its successful isolated import/verification is administrative only. The refreshed bundle/review delivery f207 has been remotely tree-verified with explicit CI observations; publication of the later closure metadata remains pending; the eventual administrative containing commit is not its own embedded source.

<!-- PR14_VERIFIED_PRECLOSURE_PUBLICATION -->
## Verified preclosure publication

- Production source: `a73b9635510fb09d7baa5339e14d1aa609f0ff87`; source tree `e343cb9f477dcd86556f7b6595189fa2c9c1c099`.
- Frozen local delivery: `f207912ac1ad8d3c6dcc8368831a6cc0ea2748d4`; exact tree `76ffd38cf7470879347423a4f8467cf47547aae6`.
- Verified preclosure remote head: `a747db386160a7f3737fa67b25207e34f5c118d9`; tree `76ffd38cf7470879347423a4f8467cf47547aae6`.
- Every path/object/mode matched the frozen delivery, including all 919 review files; base/main remained unchanged and README differences were zero.
- CI observed at `2026-10-13T17:53:14Z` for that exact preclosure head: 0 checks and 0 status contexts. Counts are observations, not an overall CI-pass claim.
- Evidence: [E014-ceba3ce114bee654](evidence/E014-ceba3ce114bee654.json); [publication identity and CI snapshot](publication.json).
- The later commit containing this closure metadata is not self-referenced. Its full remote head/tree comparison and fresh CI observation are delivered directly to the user with fixed-version links. Formal records remain in the authorized PR files; no final PR Conversation comment is requested.
- Real-GPU acceptance remains pending. PR remains draft and unmerged; no merge authorization.
- All review requests, evidence, results, developer replies and rechecks remain in these authorized review files. Do not write PR Conversation descriptions, notes, progress updates or comments, including a minimal entry or final publication comment.
- All earlier evidence, review results and response history are retained. This update contains administrative records only, with no compile, test, runtime or cleanup execution.

## 独立实机GPU FAST交付 20261013T182722Z-fast

固定原版802a11e7，RTX4060/sm89真实执行。当前不建议合并；4个生产/传输发现、FP64微差待解释及14项未测范围保留。补丁仅隔离文件，不是正式修复。

- [中文报告](reviewer-deliveries/20261013T182722Z-fast/REPORT.md)
- [先行计划](reviewer-deliveries/20261013T182722Z-fast/TEST_PLAN.md)
- [33项覆盖矩阵](reviewer-deliveries/20261013T182722Z-fast/coverage.json)
- [原始结果和计数](reviewer-deliveries/20261013T182722Z-fast/results.json)
- [版本/patch/bin](reviewer-deliveries/20261013T182722Z-fast/versions.json)
- [复现](reviewer-deliveries/20261013T182722Z-fast/REPRODUCE.md)
