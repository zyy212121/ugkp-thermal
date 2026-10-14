# Supplemental independent wet momentum review — FAST3

PR14_REVIEW_ONLY. Formal source `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`, tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`; previous source `4b5c480b572f234ddecf5e89164fe59b0d2e9735`. This source seat read the new formal heat-case checkpoints and source, did exact binary64 conversions and 80-digit decimal offline arithmetic, and wrote only this MD/JSON supplement. **No GPU run, compile, production/index/HEAD mutation, gate or old OOB rerun.** Earlier independent-source-review.md/.json remain byte-identical (pins in JSON).

**Conclusion:** the new zero-mass fix retains real nonzero phase/gas forces and their packet ledger consumers. The much larger wet physical momentum residual is localized to the film's net normal packet impulse, while the film remains at zero diagnostic momentum. The existing reduced-film model imposes its velocity profile and has neither a conserved momentum unknown nor a separately observed balancing mechanical reaction. Source explicitly disclaims full rigid-solid/reduced-film momentum closure. This is an existing model/accounting limitation, not an identified regression from the phase-volume ternary. **The requested complete physical global P proof remains unclosed.**

## Actual checkpoint evidence

Input artifacts are `artifacts/WET-FILM-FORMAL/checkpoint-{0,1,2}-full-decoded.json` in this delivery. Their byte hashes, complete calculated vectors and raw packet sums are in the JSON. Inventory arithmetic uses stored mass/meanVelocity/momentum, so it does not use a density guessed from decoder metadata.

Let Pgas be summed GasQ.momentum, Pfilm=ΣMfilm×meanVelocity, B the outward gas non-interface boundary impulse, and S the recorded solid-received supportImpulse. From t0, initial Pgas/Pfilm/B/S are zero; Pfilm remains exactly zero in all 4 faces at both saved endpoints. The nominal balance is R=Δ(Pgas+Pfilm)+B+S:

| Time | Pgas,x | Bx | Sx | Film exchange,x cumulative | Nominal Rx |
| --- | --- | --- | --- | --- | --- |
|0.0005| -8.2417331024189478e-5 |89.083520307010602|-89.083528051641849|+9.0161962219781344e-5|-9.0161962271608898e-5|
|0.001| -2.7159474301006412e-4 |178.16699409750399|-178.16688032344035|+1.5782067936953581e-4|-1.5782067937484499e-4|

The original stock-scaled absolute 1e-14/relative 1e-10 rule remains unchanged. These nominal residuals exceed it by many orders of magnitude. Adding the **already recorded** FilmParticipant exchange to R leaves approximately -5.1827554e-14/-5.3091754e-15, localizing the dominant term. This algebraic rearrangement is not a reaction measurement, new conservation rule, source repair or green global-P result. The gas cumulative audit has its own roughly1e-13 rounding residual too.

Each saved window contains 4 SolidFilm packets, 200 GasFilm packets and200 PoreGas packets. All 404 masses are zero; PoreGas momentum is zero. SolidFilm Px totals 89.083528051641849/89.083352271798489, GasFilm Px totals 89.083437889679595/89.083284613081443. Therefore film net receipt ΣSolidFilm−ΣGasFilm is **+9.0161962254190e-5/+6.7658717046282e-5 per window**, while the model profile's momentum change is exactly zero. All decoded normals are (+x), base/mean/normal-front velocities and kinetic energy are zero, and recorded supportWork/bodyWork are zero. The mechanical discrepancy is normal impulse, not a hidden tangential velocity or mass-transfer direction.

Zero work does not imply zero force: a stationary support can carry impulse without mechanical work. The reaction required to hold an inertial film at zero P would be negative net film packet impulse (in this no-body/no-side-transfer case), but it is **not independently modeled or recorded** here. Do not fill a ledger with that residual-derived number merely to close the audit.

## Decoded model fields are not runtime parameters

`Checkpoint.C:637–654` ignores the supplied expected ModelConfig, skips the serialized configSize and returns only decoded HostState. Thus S303's physics object with liquidRho = 0, e0 = 0, condensedRho = 0 is the standalone caller's default model, not the consumed physical model. Those defaults cannot diagnose missing material data, supply an energy denominator or justify a new admission rule. The original same-input reference contains liquid rho = 1000, e0 = 100000, referencePressure = 171600, and the application D302 R3CONFIG must corroborate the actually consumed values. This P calculation does not require rho; physical energy validation is separate.

## Producer → transfer → consumer → commit

- Native gas `gpu/Backend.cu:256–284` produces GasFilm/PoreGas from actual gas-stage prepared/final flux. `290–307` integrates actual ledgerDt/final-flux scaling and books gas exchange plus outward non-interface B. This force is gas conservative momentum's source, not merely a printed marker.
- CPU material `CpuMaterialDriver.C:292–304` selects its driving history trace; `CpuSurfaceInterface.C:57–71` produces the zero-mass SolidFilm heat/pressure/shear packet. Its normal force uses the constitutive gas pressure (`69`), and its consumer remains intact after the new volume conditional (`460–463`).
- `CpuMaterialDriver.C:117–122` passes mass/species/energy/work flags to CpuFilmForcing, **which has no momentum member** (`CpuFilmDriver.H:9–17`). `418–420` passes constitutive normal kinematics to CpuFilmDrive.
- `CpuFilmDriver.C:184–237` recovers the film and sets coupled normal traces. `FilmMath.H:17–28` derives tangential mean velocity algebraically from wall speed/shear/pressure gradient; `60–80,117–145` imposes normal kinematics and recovers profile diagnostics. No d(Mu)/dt update consumes the net packet impulse.
- Film advance `CpuFilmDriver.C:366–414` updates M/species/H and final profile. `398–403` books powers and M/E/Y, not a film support impulse. FilmRateBudget (`Types.H:84–87`) is a power/energy-rate budget, not a full force budget.
- CPU packet accounting `CpuMaterialDriver.C:35–40,548–559` books reciprocal solid/film exchange momentum; **only ConsumeSolid contributes supportImpulse**. The same definition appears at `Coordinator.H:78–84`. The film report's duplicated exchange channels are suppressed (`582–584`), and `616–619,670–683` retains packets and publishes the full accepted state/budgets. No residual mechanical/constraint impulse is added on commit.
- `Types.H:83–94` gives rigid SolidQ and FilmQ no conserved momentum. `IntervalAudit.H:54–56` explicitly states the audit does not claim a full momentum equation for the rigid solid/reduced thin-film model. Budget (`Types.H:160–170`) has boundaryMomentum/supportImpulse/exchangeMomentum but no separate film constraint, body impulse or edge momentum channel.

The model's **valid conservative momentum stocks** are GasQ.momentum and actual particle momentum where particles are enabled. Mfilm×meanVelocity is a valid instantaneous integral of the assumed constant-density profile, useful for this external physical diagnostic, but it is not a separately evolved momentum state. A rigid body's prescribed-frame momentum/reaction and any normal-constraint or neglected-film-inertia mechanical term must be defined and independently accounted before the total physical P equation can be certified. Current M/E/element closure and pairwise packet force bookkeeping do not supply that missing equation.

All mechanical profile/state/accounting files inspected are identical to the previous source, apart from the intended local zero-volume expression. No numerical gate or changed momentum expression was introduced. Keeping zero phase mass from reading an unselected density fixes GPU-F006; it does not create or repair the reduced-model momentum equation.

## Source-grounded cause and remaining evidence

The CPU phase force uses a representative material-drive pressure over its material substep, while GasFilm uses separately sampled, finalized native-stage flux and ledgerDt integration. This source chain can produce a finite difference in top/bottom normal impulse; it does not require floating-point cumulative rounding to explain a 1e-4 net force. Exact first-stage pressure/limiter/weight lineage and the intended approximation/constraint interpretation still require matched actual observations. This supplement does not identify one sampling choice as the unique root cause.

Persistent identities must remain distinct: actual SolidFilm face values are 12–15, GasFilm 16–19; the CPU phase uses gasMesh.faceIds[gasFace], while native gas identity uses surface.persistentId. Both source paths carry film slot i, but the standalone JSON does not emit packetfilmFace or surfacepersistentId. Grouping equal numeric face IDs would be wrong. D302 prints the phase film/solid slot; expose the matching native slot or use a decoder that prints its actual stored addressing before claiming per-recipient force history.

## P2 qualification — OBS-R3-WET-P-PHYSICAL

`CpuFilmDriver.C:366–414` is the primary consumer reference; exact blob/SHA and related context pins are in the JSON. The legal zero-mass wet input receives net normal film impulse while its profile remains static. This is an observed unclosed physical P balance and a pre-existing reduced-model/ledger gap, not evidence that the repaired volume branch discarded force. It is distinct from the old per-film packet accumulator red around 1e-13 and from the three dry/SST gas-Px roundoff reanalysis.

Require the developer to state the actual reduced-model mechanical balance/constraint assumption and independently expose applicable top/bottom forces, stage quadrature, side/body terms and any real reaction or approximation defect. Preserve the original tolerances, all strict reds and incomplete full-P status. Do not invent a reaction, add a numerical/index gate, change physics or let this reviewer silently modify production.

## D302 source-only review

The D302 file differs from formal source only by cstdio and one post-candidate read-only output loop. It recomputes a local material primitive, safely avoids an unselected density read when phase mass is zero, and prints actual consumed configuration/phase force/state. It adds no return, rejection, rollback, timestep rule or physical state mutation. Extra material recovery/I/O costs are diagnostic; actual byte-identical original output validation and runtime consumedrho evidence belong to the coordinator. This seat did not compile or execute D302.

## Declined to judge and assessment

- A measured compensating film support force: no independent such channel is observed; algebraic negative residual is insufficient.
- Unique first-pressure/limiter cause: endpoint data and source do not supply all new native stage stresses.
- Other wet/evap/melting/tangential/gravity/edge workloads or strict restart equivalence: this supplement reads only the formal heat checkpoints.
- Full physical P acceptance: the requested proof remains unclosed, and is explicitly retained rather than silently excluded.
- Any new production fix, admission gate, tolerance/physics change or old OOB rerun: none authorized or performed.
- D302 runtime/source-output equality: source inspected only; coordinator must attach actual evidence.

**Scoped assessment:** zero-mass force/packet accounting chain is retained; actual nominal physical wet P is red/unclosed with a documented model limitation. No complete physical global P PASS or whole-PR FAST/merge readiness follows.
