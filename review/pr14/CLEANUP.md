# PR14 review-only removal plan

PR14_REVIEW_ONLY. Status: NOT RUN. No final approval yet.

The records are temporary development material. Keep them throughout FAST review, developer responses and rechecks. They do not alter simulation behavior. Prefer the existing file/function sidecars over source comments.

After the user/reviewer gives final approval:
1. Freeze the approved production commit/tree and all reviewer results. Commit the complete evidence records to PR history.
2. Confirm the manifest explicitly inventories every review file; any newly added reviewer log requires an explicit manifest entry. Validate records and dry-run cleanup.
3. Select a new durable archive destination outside review/pr14. Provide the actual final approval reference. The explicit cleanup --apply call archives every owned file and any exact source-comment preimage, verifies archive CRC/bytes/hashes and rechecks unchanged inputs before the first mutation.
4. Preserve/upload the archive and SHA256 where the user and future reviewers can retrieve it. Keep prior PR history. An archive on a disposable machine is insufficient retention.
5. Inspect the cleanup diff before its separate commit: only explicit review/pr14 files and, if ever approved/registered, whole standalone marker-comment lines. No production file deletions, README edits, replacement of other comments, whitespace normalization, or computational changes.

Default invocation does not archive, remove, or alter anything. The --apply flag alone is insufficient: archive path and approval reference are required. This tool supplies a mechanical removal capability, never grants approval. Production files are protected by exact before/after hashes and a line-only marker grammar; current source_markers is empty.

If archival fails, no cleanup is started. If concurrent changes are detected, stop; do not broaden ownership or retry destructive actions blindly. Partial filesystem failures can occur after archival: recover exact bytes from the verified archive and inspect before continuing. Run only on a quiescent checkout.
