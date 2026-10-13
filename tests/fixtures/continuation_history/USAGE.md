# Offline continuation provenance

Some regression tests and PR14 review records identify original local Git
commits that tree-only/web-upload publication does not retain. From the repository
root, import both historical layers before using their pinned baselines:

```
python3 tests/fixtures/first_round_history/prepare.py
python3 tests/fixtures/continuation_history/prepare.py
```

The existing first-round fixture is unchanged. This continuation bundle advertises
only `refs/audit/continuation-original-history`, at `a73b9635510fb09d7baa5339e14d1aa609f0ff87`.
Its single prerequisite is the published first-round head
`82f9c016704c9822b16b87cdeeb11f5bbd44d06f` and that commit's ancestry.
A shallow or tree-only checkout missing that prerequisite must obtain its
published base history first; this offline fixture does not supply that history.

The importer fetches only from the local bundle into its audit ref. It does not
use the network, check out files, move HEAD/current branch, write the index, or
write FETCH_HEAD. Its audit ref retains the original reachable objects against
normal garbage collection. Repeating an unchanged import is harmless; descendant
bundle updates advance only that audit ref. Git reports missing prerequisites or
incompatible history without substituting another baseline.

## Exact continuation commits

- `b236292f4c0b5e7e19ce04dd73e7fa0b54186bef` Complete terminal EOS and primitive publication without changing successful retries
- `1aaee9eb905562ea04bafefcb05e40c88bb94508` Complete gas face outputs and retain successful transport selections
- `cf49400267b0df39017d6ce348ce72cbcab6b513` Complete gas cell SST and stage updates while preserving successful skips
- `fe254252b643589d26af0572b67cdf95ab2cbbad` Complete chemistry stages and actual remaining interval publication
- `bb1057e40856f9562ff60ce31cae51e35f1f3d8b` Complete wall layer profiles exchanges and fresh terminal packet publication
- `5f1a39ac96e2e7ff9a0640a4db344d9cccd34245` Preserve resident gas state across configuration and retain failed cleanup owners
- `55b7192cf269f06017d5286d302c1a0757e8a39c` Complete gas terminal consumers and preserve valid lagged wall inputs
- `0017f07851d261fcfa9fcf4cb86f4e06dfec2f31` Align development probes and migration with resident constructors
- `449cfbaad1cb5432168c36acf3ad538fca202df2` Complete thermal arithmetic publication and remove redundant numerical scans
- `cde979ebf7467cb2d5d02fcf27634fe233cc0fa3` Remove resident metadata admission scans and preserve native execution paths
- `9ac74fc16d19bf809f58fbe565903b5d9e13ca77` Complete cold-wall native thermal publication without numerical admission
- `ee77457f3b60728b201d47bfca32d25505ff7199` Complete native frontend material and supported boundary publication
- `7fbfb2b49e1060bee07574f857f26691d28fce9f` Complete particle and pool native contribution publication
- `a164be44fd28b3d05a1cb430e4e98675b1397bd4` Complete resident mirrors and native source publication without metadata admission
- `7982a97294d4a9f19184517226bcde036f235a56` Preserve offline continuation source history for PR review
- `970d9fedcc061dfa08fc88d427a1a47b3c22f65f` Preserve restart codec dispatch under native C++14 builds
- `dc79925991e8dd4f4fc8f339942a3b9e85e0c633` Pin offline review history to final C++14 source
- `d1abd791774a2abd0af26fed09f3ebee3353c21e` Add PR-local FAST review requests and immutable developer evidence
- `47e6c02ff7cd127f1a92c1fcf712d59450e71bab` Include explicitly owned raw build logs and review evidence
- `a73b9635510fb09d7baa5339e14d1aa609f0ff87` Remove unconsumed mapper configuration and source-size refusals

The face-producer completion tests pin B1 (`b236292...`). Resident terminal
wrapper, storage, and Courant tests pin B6 (`5f1a39a...`). EOS completion and
review regression tests pin the published prerequisite (`82f9c01...`). The
caloric reuse baseline (`60ddb60...`) is supplied by the unchanged first-round
history. Review records additionally identify B2–B7 and captured later heads.
The full commits and their reachable trees/blobs are retained here; merely
computed working-tree hashes and unavailable older provenance are not invented.

This is history delivery only. Administrative bundle/import checks are not
numerical, host-shim, application, compilation, or GPU validation. The pin covers
only committed history through the ID above, not later working-tree edits.
Regenerate after the final source commit is frozen, before committing delivery
metadata; never describe the bundle-containing commit as its own embedded pin.
