# PR14_REVIEW_ONLY FAST reproduction

Read SOURCE/TEST_PLAN/REPORT and exact percommand logs/*.spec.json first. Fixedproduction 4b5c480b572f234ddecf5e89164fe59b0d2e9735; oldevidence ae5a59478023c74e23b64dc1a1f388fb805ef565; recordparent 75cfd885c028fe80d011a8a081f8bed1f8aeb55e. Allcommands actuallyexecuted in commands.txt. Source/base/variants/flags/binary hashes in eachspec andversions; rawlogs remainraw. Do notreusean ID/overwriteoriginalFAIL. No productionlongrun ormerge.

Use theuserauthorized test directory, freshisolatedcheckout andinstallation; source inputs/env.sh forOpenFOAM10/CUDA13.1/sm89. r2_paths.json contains actual paths used, adjust thoseonly in anewdelivery foranothermachine. Build originalfullapps/Ns10 from fixedproduction. Most helpers requirepriorprepared work/cases andoldimmutable evidence paths; executein TEST_PLAN dependencyorder, notall r2_*.py wholesale. Setup/publication scripts are records, notauthoritytochange formalinstallation.

Main execution scripts: r2_runner(nativebuild), r2_native(originalCHMT/CHT), r2_fsh/r2_fsh_recheck, r2_fixture_builds, r2_owner_complete/S208 r2_owner_audit, r2_co_prepare/r2_more_gpu, r2_clock/r2_analyze_clock/r2_finish_analysis, r2_thin_prepare/build/r2_f_initial, r2_observer_prepare/r2_more_gpu, r2_gcl_prepare/run, r2_ledger_prepare/run, r2_f_prepare/run/repairedinputs/r2_wet_diagnostic, actualASan reader/S207, r2_group_audit and scopedbudgetanalyses. Numericchecks operate actualnativeproducedfields, neverCPUmock. Current portablehelpers retain exactmath; some auditoutercommands computeinside driverandappendreport hashes, so consult driver-scripts andpointJSON, notjust captured placeholderprint. SpeciesNs2/10 andfixturepredicate/hashes explicit.

S202/S203/S208 are fixtureonly; D204 is diagnosishost source only(originalGPUobjectsameSHA). Preserve originalfixture/wetred anddo notpromote diagnosis toformalpass. Added allgeometry/ledger/clock observers are IOonly andexcludedtiming. Reconstruct removedidenticalfiles from metadata/immutable-references.json beforeusingdirectpath helpers; verify SHA. Any referencedlarge nsys-rep/sqlite/binary remainsin authorizedtestwork path with SHA; leanCSV/pointdata contains emittedproof. Newworkflowinstances mustrecord freshcommands/inputs/hashes/output andbudget.

## 已有不可变数据

先在已获取固定历史提交的本仓库运行 `python3 review/pr14/reviewer-deliveries/20261014T072053Z-fast/inputs/hydrate_immutable.py`，按 metadata/immutable-references.json 验证 Git blob 与 SHA256，再恢复原分析命令使用的缺省路径。脚本只恢复已列明的缺失文件；新 raw logs/specs 和失败数据保留在本轮目录。恢复出来的旧别名不属于本轮新增 owned_paths，不在新 SHA256SUMS 中计为新证据。
