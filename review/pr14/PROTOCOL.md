# PR14 developer ↔ reviewer protocol

PR14_REVIEW_ONLY. This is a removable development/review record system, not a production prerequisite, runtime contract, or permission to merge. All current requests are **FAST**. It applies retroactively to the first-round changes, B1–B7, tooling, and subsequent global-review repairs.

## Authority and division of work

The current agreed loop is: developer source review and genuine compilation → reviewer objective real-GPU evidence → developer formal production repair/reply → reviewer recheck of the exact repaired version. Isolated reviewer diagnostic patches are evidence only; retain their diff/hash and do not silently promote them to production fixes. Earlier external instructions assigning formal repair to the reviewer are superseded by this division of work. The external prompt is only an entrypoint to INDEX.md.

No additional developer CPU-shim, application numerical, or parity runs are requested. Earlier such results are retained as historical scoped evidence, never relabeled current GPU acceptance. Unexecuted new standalone CPU fixtures stay outside production tests. Actual nvcc/OpenFOAM compilation and link scopes must be stated exactly; a small instantiation or relocatable device object does not prove final application linking or execution.

No README modifications. No new numerical/precision/version/metadata rejection, rollback, or stop boundary. Actual missing consumed inputs/backing, unavailable selected implementations, IO and CUDA errors remain distinguishable. Do not reinterpret a completed NaN/Inf numerical result as missing storage. Wrong supplied count/index risk accepted by the user is not authority to add protective admission. Test-process isolation/timeouts are external observation only.

## FAST and PRODUCTION

- FAST: bounded actual GPU operator and small-case verification, full fields and native consumer writes, valid-case conservation, calls and interactions, coupling/transactions, restart/continue, asynchronous/concurrency evidence, and small representative timing when useful. Start here for this PR.
- PRODUCTION: physical long runs, experimental reproduction, large campaign, or long performance study. Only after FAST has stabilized and a separate scope/input/criteria/budget plan has been explicitly requested/agreed. Nothing in this folder starts PRODUCTION by default.
- Every request, result, reply, recheck and escalation states FAST or PRODUCTION. Reuse unchanged valid evidence; explicitly record changed dependencies and why a selective recheck is needed. No overlapping pass-count summation.

## Stable IDs and source identities

- R014-001…: stable topics. Never renumber or reuse an ID; append new topics.
- F014-…: stable SHA256-derived repository-path identity.
- FN014-…: stable path-and-symbol/scope identity. Diff context may be a lexical scope rather than a full function; locator_kind says which. Reviewers correct or refine mappings by appending exact symbol/context, not pretending a regex proved whole-program coverage.
- E014-…: immutable shared evidence identity derived from content SHA256. A revised report/log gets a new ID; do not overwrite its older bytes. Original report paths are provenance labels, not prerequisites for navigating these PR-local copies.
- Q/A/FIX/RECHECK entries append to the same topic's responses/R014-xxx.json. A result must cite topic/file/function IDs and actual source commit/tree/blob/hash. A working-tree SHA256 is an honest draft pin, not a final commit.
- Final source commit/tree, record-containing commit, remote head/tree and CI are separate identities. Do not hash a commit into itself. Record-containing commit can be supplied by immutable PR history; freeze the production tree/source blobs independently. Final publication metadata remains pending until actually verified.

## Required request fields

Before execution the reviewer records:
1. Interaction class, question ID, topic and file/function targets; original-success or formerly-rejected behavior and producer→storage/transfer→consumer→commit chain.
2. Exact source/baseline commit/tree, relevant blobs and dependencies; input names and SHA256; actual binary/object hashes; device/GPU count, driver, CUDA/compiler/OpenFOAM versions, build flags and precision choices.
3. Exact commands, working directory and required environment; execution type (compile-only, GPU numerical, concurrency/sanitizer, integration, timing). Proposed commands are visibly not executed commands.
4. Expected observations and comparison rule for all consumed fields, ledger/state transitions and real errors; absolute/relative/bitwise tolerance with a reason. Do not apply physically valid finite/conservation criteria to pathological inputs in order to reject computation.
5. Explicit small input/step count and external wall-time budget. The initial proposed FAST cap is 300 seconds per case; actual reviewers may agree a more suitable recorded bound. A timeout is a factual test outcome, not a new source-code stop condition. No dangerous intentional OOB experiment.

## Objective result entry

Append to reviewer_results, preserving failed, blocked and inconclusive attempts:

- id, question_id, author, UTC timestamp, interaction_class;
- source_commit, source_tree, source_blobs, input_hashes, baseline_commit, dependency_hashes;
- toolchain/device/driver/build configuration and binary/object hashes;
- exact command, working directory, start/end times, exit status, external timeout or interruption;
- observed versus expected per-field values/bit patterns, ledger/conservation residual, call/launch/copy counts and execution scope;
- full log/artifact paths under this folder and SHA256, rather than just “PASS”; evidence_ids for shared results;
- verdict PASS/FAIL/BLOCKED/INCONCLUSIVE/NOT_RUN and precise limitations. Environment/setup failures are not numerical failures; a reproduced bug is red evidence, not green acceptance.

Use one log/artifact once and reference its evidence ID across affected functions. Do not put credentials, secrets, enormous binaries, or unrelated personal/internal notes in PR history. New reviewer evidence must be added to manifest.evidence and owned_paths; run the administrative validator before committing records. The byte-exact raw logs remain raw; PR14_REVIEW_ONLY ownership is their containing path plus manifest, not a fabricated line prepended to original logs.

## Developer reply and closure

Each developer_replies entry references the exact result/question, states acknowledged or disputed facts with evidence, root cause, full native repair/call-chain scope, affected success/terminal behavior, cost changes, formal fix commit/tree/blob IDs and genuine compiler logs. Preserve unaffected valid evidence and explain invalidation. If a reviewer result is contested, state the specific discrepancy rather than changing the result.

Each recheck_results entry records the repaired source and fresh or valid-reused GPU evidence. Final decision names the actual reviewer, source pin, remaining limitations and unresolved IDs. Local historical approvals do not imply either independent whole-PR pass is complete. “Merge-ready” requires the agreed evidence; it never means user permission to merge. Actual merge always requires separate user approval.

## Removal and archive

Do not clean up now. Prefer sidecars; no source comments have been added by this record system. After final approval, preserve the full records in PR history and create/verify a durable archive before removal. Ensure the archive is actually retained in a user-accessible artifact/PR attachment or repository archive; a temporary local ZIP alone is not delivery.

cleanup_review.py defaults to check/dry-run and never modifies code. It accepts only explicit manifest-owned review files and explicitly listed, byte-exact standalone `// PR14_REVIEW_ONLY R014-...` comment lines. It never deletes production files or README files and never performs source search-and-replace. It archives all current review bytes and any pre-edit source originals before mutation, verifies the archive and all source pins, then removes only listed records/comment lines. Every other production byte must match the declared after hash. There are currently zero approved source-marker edits.

Application is a later explicit action requiring final approval and a new archive destination:
`python3 review/pr14/cleanup_review.py --repo /path/to/checkout --apply --archive /durable/new-pr14-review.zip --approval-reference <verified-final-approval-reference>`
This command is documentation, not authorization or execution. Commit the reviewed cleanup separately, retaining prior PR commits and archive evidence. Re-review the cleanup diff to establish that no production computation or unrelated comments changed. Never run cleanup automatically as part of builds/tests, installation, startup, or CI.
