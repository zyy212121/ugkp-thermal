# Existing FAST2 momentum evidence: offline arithmetic attribution

PR14_REVIEW_ONLY.

This is developer-side offline analysis, not a new GPU run or independent reviewer acceptance. Original FAST2 evidence is unchanged. No production code, arithmetic, precision, gate, case input, endpoint, or tolerance was changed.

## Result

For the three explicitly named CHMT cases, replaying the recorded binary64 boundary impulse products and gas exchange packet additions in production order reproduces both cumulative checkpoint momentum budgets bit-for-bit, at both window endpoints. Separately integrating the actual final boundary flux with the actual RK ledger time weight in 80-digit decimal arithmetic gives:

| Case, window 2 | Actual gas Px inventory + final boundary impulse | Retained original cumulative-budget residual |
| --- | ---: | ---: |
| SST-SSPRK2 | +2.8099402275e-20 | +8.0149895103e-12, FAIL |
| SST-SSPRK3 | +2.0552727120e-19 | +8.0289771413e-12, FAIL |
| dry-ablation | -3.3259918837e-19 | -3.5785010887e-13, FAIL |

The native budget subtracts two approximately 89.0835 quantities to resolve a gas momentum change around 1e-4 to 1e-3. RESULT.json decomposes the difference between the physical-flux reconstruction and the native budget into cumulative-addition roundoff and product/packet representation effects. Thus these particular gas Px discrepancies are attributed to recorded arithmetic/representation, rather than an unrecorded physical momentum flux. This does not repair the cumulative-budget criterion or change its FAIL to PASS.

The independent physical-flux residuals at all 100 accepted microstep endpoints per case and both macro endpoints are below the same original external criterion: absMass=1e-14 plus relMass=1e-10 times the maximum absolute before/after gas Px inventory. Dry-ablation window 1 already passed the original budget criterion; its status is preserved too.

## Exact coverage and exclusions

- Three cases only: SST-SSPRK2, SST-SSPRK3, dry-ablation, each 8 gas cells, 2 windows and 100 accepted microsteps.
- Actual final face fluxes at all 200, 300 and 100 stages respectively. Each stage has 36 recorded faces, including 24 physical boundary faces: 4 Interface and 20 NoSlip. There are no periodic, inlet or outlet boundaries in this scoped replay.
- Gas-side x momentum only. Each interface stage contains 4 GasSolid and 4 PoreGas packets, including zero-valued channels; the script checks all 8 records. The script checks accepted gas inventory, not per-stage intermediate cell inventory.
- Physical gas mesh Interface faces are 12–15. Persistent packet face identities 16–19 are a different namespace; they must not be used to select physical boundary faces.
- No film inventory or film recipient momentum claim; the D204 wet-film residual remains unexamined by this script. No particle momentum or rigid-solid support/constraint momentum closure. No FSH or CHT resident full-budget conclusion. No complete three-component/vector momentum or whole-system M/E/P conclusion. No gross SST production/dissipation model validation.
- This reconstructs the observed flux balance. It does not certify all possible volume-force/source inputs or infer absent channels for other cases.
- Existing D203 artifact comparisons say all original outputs are byte-equal to their corresponding uninstrumented run; the script verifies that retained comparison flag. It does not rerun the observer.

## Sources and reproduction

All input paths and SHA-256 hashes are recorded in INPUT_HASHES.json. Main evidence is under review/pr14/reviewer-deliveries/20261014T072053Z-fast/:

- logs/ACTUAL-WEIGHTED-LEDGER-{case}.log and matching .spec.json
- artifacts/extra/{case}/actual-weighted-final-fluxes.json
- artifacts/extra/{case}/checkpoint-{0,1,2}-S207-decoded.json and original state.chmt
- artifacts/extra/{case}/complete-independent-budget-audit.json
- patches/D203-actual-stage-ledger.patch

The original GPU source was 4b5c480b572f234ddecf5e89164fe59b0d2e9735. This analysis began from repository base f5e6c5e2cbde2693f5fa811d874c200b07bbbc66. Relevant unchanged production semantics are applications/CHMT/gpu/Backend.cu:193,290–307; applications/CHMT/core/Types.H:121; applications/CHMT/coupling/ExchangeLedger.H:16–20; applications/CHMT/coupling/IntervalAudit.H:143–149. Hashes of these files are included. The archived case input folders are deltas; inherited original input files are not represented as newly present files.

Run from the repository root:

    python3 review/pr14/developer-deliveries/20261014T112400Z-fast/evidence/strict_momentum_reanalysis/reanalyze.py --root .

The script writes only RESULT.json and INPUT_HASHES.json next to itself. It validates exact stage/face/packet/cell coverage, exact reproduction of the native cumulative budget bits, and equality with each original retained bulk-P audit residual. Python binary64 reproduces native product/add operations; Decimal.from_float preserves each logged binary64 operand, with 80-digit arithmetic for independent summation. Those decimal computations are audit arithmetic only, never a new production ledger.
