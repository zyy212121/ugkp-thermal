# Pairing-pass fusion

## Approved scope

The paired energy arithmetic and publication model remain unchanged. Remove the
independent heat clear and candidate-reduction launches by reusing two existing
post-sampling correction traversals and the existing pool clear. Keep the global
acceptance and settlement boundaries, including fused gas recovery. This replaces
the old heat/recovery pair with two cell launches: zero net added launches or
traversals, though heat arithmetic, reads/writes and atomic additions remain.

## Epoch and partition

The pool clear runs after completed gas RK and before sampling/correction. It can
clear optional A/Q and borrowed rhoENext for every cell; no later collision reader
uses rhoENext. Heat-disabled state has null A/Q and performs no thermal work.

Sampling visits each particle slot once, and appends each successfully sampled
stuck particle once to the selected compact list. Invalid cells or nonpositive
sampling pools set status0 before append. At the subsequent mobile-correction
full scan, `stuck && status==2` identifies exactly this deferred selected set.
Capture that predicate before correction. Correct mobile particles normally;
accumulate heat afterward for every nondeferred slot. Dead/invalid/nonpositive
mass slots remain excluded by the existing candidate helper. The selected-stuck
loop corrects/finalizes its listed particle, then accumulates its candidate,
including any detachment, changed velocity and cleared cold profile. No particle
is double counted or evaluated before its last correction.

The two existing kernel completions are the producer barriers. Acceptance must
remain after both; publication must remain after acceptance; gas settlement must
remain after every publication. No grid synchronization, new persistent buffer,
changed collision index order or moved model update is introduced.

## Interfaces and verification

Extract `clearPairedParticleHeatCell(s,c)` and
`accumulatePairedParticleHeatCandidate(s,i,dt)` from the existing standalone
wrappers. Keep wrappers as unfused numeric-reference fixtures, with no production
launch. CHT correction kernels are templated on `PairedHeat`, defaulting to false.
StrictAdvance explicitly selects true with its current physical dt when paired
buffers exist, or false with the original correction body otherwise. The false
specialization prevents a runtime branch from imposing the enabled calculation's
register allocation on disabled heat. Other applications retain their existing
non-template signatures and calculations.

Tests must compare real correction/finalization plus full acceptance, particle
publication and gas settlement against the unfused reference in physical32/64.
Include selected/unselected/mobile/stuck sets, invalid sampling pools, injected
particles, empty cells/lists, heat disabled, cold detachment and signed exchange.
Use actual exchanged energy and bounded double reduction roundoff, not a large
background-energy tolerance. Existing L0/L1, lifecycle, cold and full strictAdvance
fixtures remain required. Native application compilation is serialized by the
root; GPU runtime remains unavailable.

Moving original post-transport moment, packing or directory clears is outside
this minimal repair: it would expand host scheduling contracts merely to optimize
pre-existing passes. Such optional opportunities are separate from removing the
new overhead.

## Frozen implementation and evidence

The hot-path production change is frozen at
`cd5074ba21a9048a5ceef4cca66909a430d288a0`,
based on integrated PR12 `50237c4`. `e95e470` implements the fusion; `cd5074b`
adds the compile-time disabled specialization. The real numerical fixture is
`tests/gas_transport/test_paired_heat_fusion.py` (`cc86fe7`, expanded in
`d8adc02`). It uses production sampling, correction and finalization plus actual
acceptance, thermal publication and EOS settlement in physical32 and64.

The fixture compares candidate sums with an independent long-double accumulation
and compares fused publication with the unfused helper reference. It covers
selected/unselected receivers, invalid pools resetting an old status2, actual
1D/2D detachment, reversed lane order, mixed positive/negative heat, gas near its
floor, empty lists/cells and zero collisions. It verifies exactly-once candidate
atomics, positive-dt null-buffer guards and no thermal evaluations or atomics in
the false specialization with positive dt and non-null buffers. All24 numerical
cases pass. Three deliberate wrong-partition/early-epoch mutants fail.

Before the specialization, native physical64 resources showed mobile correction
rising from50 to84 registers and selected-stuck correction from80 to92. That
proved a runtime heat-disabled branch was insufficient to preserve the old static
resource cost. Final specialized native resources are recorded separately; HOST
results cannot establish register allocation, GPU atomic ordering or speed.

Scratch arrays,72N logical scratch bytes per active step and the candidate/
publication double atomics are unchanged by fusion. Candidate input reads and
caloric arithmetic still cost work; removing a traversal does not eliminate
that work. No extra allocation or synchronization was introduced.

The later `dc25e4d` also removes the adjacent standalone carry-finite scan from
restart upload. Its check runs in the existing full particle validation loop,
with the same message and before particle-array copies. Real extracted-loop
regressions cover valid input, NaN and signed infinity at every position, and the
old invalid-mass case in both precisions. No wire format or precision-rebase
arithmetic changed.
