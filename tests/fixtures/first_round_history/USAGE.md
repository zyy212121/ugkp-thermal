# Offline first-round parity history

Some regression tests use `git show` on the exact commits before each change.
A tree-only/web-upload publication does not include those original Git objects.
Before running these tests in such a checkout, run from the repository root:

```
python3 tests/fixtures/first_round_history/prepare.py
```

This fetches the bundled original history into
`refs/audit/first-round-original-history`, retaining its reachable objects against
normal garbage collection. It does not check out files, change the current
branch, or write FETCH_HEAD. Running it again is harmless when the audit ref is
unchanged. No network access is needed.

The incremental bundle contains exactly one advertised ref:
`refs/heads/perf/chmt-efficiency-refresh-20261011` at
`cab2910989abe37abd62586d2e16990c025f5766`.
Its prerequisite is `d6110c02745a8ad726208a426c4bb3a28233c1fd` and that
commit's existing ancestry. A shallow or tree-only repository without that base
must first obtain the base history; Git reports the actual missing prerequisite.
The script does not synthesize a replacement baseline or suppress Git failures.

This restores test baselines only. It neither claims GPU execution nor changes
any numerical assertions. Shared producer closure and comprehensive tool/audit
work remain outside the first-round scope. In particular, the optional
`particle_field_contract.py` direct-allocation parser is not yet adapted to L3
canonical helpers (FSH `pColdFrozenArea`: expected one, got zero); standard builds
do not invoke it, and that diagnostic alone is not a physical allocation failure.
