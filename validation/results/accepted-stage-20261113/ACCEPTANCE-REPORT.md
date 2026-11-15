# Accepted dual-precision optimization stage

User accepted the macroscopic comparison on 2026-11-13 and authorized freezing/publishing this stage. This is a review publication, not authorization to merge.

## Provenance and scope
- Remote main/base: 8539aae69638aaec7efe99c0192de4030ff0f9ed, previously called local stage PR41 (random initial-face injection plus its first repairs). That stage was already on main without a GitHub PR numbered 41.
- Accepted solver code: 63263fcabbe049125be1bc78cbfa04ed6b659410.
- Includes CP1, selectable FP32/FP64 CP2, integrated CP3/4/5 and CP6, manual single-scheme case controls and validation preparation fixes.
- Includes FP32/FP64 radiation diagnostic correction: post-relaxation published residual metadata no longer overwrites the actual inner-linear termination warning. No radiation field, iteration algorithm, relaxation or control parameter change.
- Added evidence-only publication commit does not change solver code. Original local development branch/worktree remains intact.

## Validation and result provenance
Historical CP6 validation used ed675527 before the diagnostic-only correction: 19 execution items exited 0; 52 matched CHMT same-precision payload comparisons were byte-identical. Three-way CHT continuation used PR41 baseline and candidate FP32/FP64 from the same saved time around 0.295 to 0.305. The attached historical-review files retain their exact scope and limitations.
The 63263fca diagnostic fix passed both production-function host regressions with failing pre-fix negative controls. The historical three-way GPU run was not rerun merely for the warning fix.

A separate fresh FP32 original-case run using 63263fca completed with exit 0:
- last progress 0.2997924619 s; last complete output 0.29500231779516362 s;
- GNU time wall 5162 s; supervisor elapsed disagrees and is not used;
- 59 solid outer / 29 radiation outer records satisfied their stopping criteria;
- historical solid-inner warnings remain disclosed, not interpreted as 59 new failures.
The user requested direct comparison at saved 0.295, not extending the run to create a 0.3 output.

## Raw, unsmoothed comparison at 0.295
Old original cdbd9973 output time 0.29500248649130967 vs new 0.29500231779516362; 164 wall faces.
Area-weighted relative L2 / integrated change:
- total heat flux: 8.984% / +0.2246%;
- convection: 5.272% / +0.4679%;
- radiation: 5.566% / +1.2427%;
- particle heat flux: 52.342% / -2.753%.
Gas temperature L2 0.1951%; graphite temperature L2 0.3550%; pressure L2 0.0391%. Macroscopic agreement does not erase local particle differences. Both contact contributions are zero; particle flux differences concern rebound.
Attached raw CSVs, plots and metadata have no smoothing or truncation. History sampling differs: old about 0.02 s, new 0.005 s. No interpolation of missing startup samples into observed evidence.

## Limits
- Original cdbd9973 vs 63263fca is NOT a precision-only experiment: inlet sampling and code versions also differ.
- Original log 7568 s ended near 0.2997255 on another machine; new 5162 s is on RTX 4060 Laptop. These are not a valid direct code speedup ratio.
- Historical three-way single-run timings are observations, not statistically established speedups.
- No full open-domain energy closure has been demonstrated; missing integrated inlet/outlet and fixed-solid boundary histories prevent that claim.
- Expanded historical radiation warning samples were misclassified (inner iteration 1 of 200). Unexpanded aggregate warnings remain unresolved, not universally relabeled benign.
- User accepts pure FP32 rounding. Particle/optical local discrepancies remain explicit.
- Evidence is compact selected results, not the large complete simulation case. Full local review archives remain available.

## Separate future work, excluded from this PR
SingleLegacy-based shared gas-particle operator unification; multicomponent existing-L2 integration; CSR free-pressure tiling; stronger preconditioning/multilevel solve; 1D cold-state stable handles. 2D coldWall stays legacy.

Publication text copies use LF line endings; CSV numeric strings are unchanged. Original local archives remain byte-preserved.
