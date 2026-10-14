# CHMT GPU→CPU dataflow audit — 2026-10-14

Read-only investigation, no repository source edits. Source HEAD 0aa34bc; concurrent workers are removing audits, so HEAD and observed working tree differ. `head_backend.cu` and `observed_backend.cu` preserve the distinction. No CUDA profiling or performance experiment was run. Counts below are source-level payload/call counts, not measured latency. `sizeof` formulas intentionally retain ABI padding where appropriate.

## Principal findings

1. Current per-stage full-volume/full-face transfers are not required by the CPU material solver. Interface consumers need sparse owners/faces; non-interface boundary budgets need boundary sums. Keeping the CPU algorithms does not justify downloading the interior domain.
2. Every gas microstep downloads the full conservative state for a host rollback snapshot, though material integration happens once per gas window. Device snapshots + window-end download can remove the repeated transfer without migrating CPU material algorithms. This changes snapshot/data ownership and must be designed and reviewed, not silently deleted.
3. Historical original CHMT already used sparse per-microstep wall records and delayed complete gas download until window end. Shared-transport rebuild 8815322 introduced full readBulk/fluxes and per-microstep downloadState; this is an established data-movement regression relative to that design, not a measured runtime claim.
4. Boundary-layer results are already W-sized, but three separately synchronized downloads include large unused struct fields and pointers. Existing sparse backing can be reused; node/workspace arrays need not cross to CPU.
5. Status scans transfer N+F integers just to find an InvalidStorage code. Preserve the existing condition but reduce it on-device; no new gate is needed or proposed.

## History and scope

First CHMT entry commit: 8f0addf; immediately preceding base: 575b7dd3d30cc648142914d6e81765a43e04222c. Initial GPU publication: 733edec. Important transitions: 8815322 shared gas transport/chemistry, 2ca2dc1 shared wall evaluation, e08bfe6 local wall pressure/storage, later PR14 gpu deliveries 48cc31f / 8989d62 / a280428. Initial publication is staged commits, not proof of the true development start date.

The original `GasWindowImplementation.cuh` lines 257–258 downloaded packets, sparse drive/gradient/traction and one budget per microstep. Lines 325–327 downloaded gas, SST, particles, void fraction and gas boundary only after the completed window. Its scalar stability/status reads and explicit stream waits are historical, not additional live calls. The old monolithic Backend also downloaded stage material packets/film aux/recession residual/diagnostics and whole endpoints; that path was replaced. Do not add its costs to the current solver.

This report exhaustively traces current CHMT GPU host-download entry points and shared gas methods reached by them. Related CHT radiation/mapper transfers are separately identified below, not incorrectly presented as CHMT calls. A repository-wide raw search is attached but contains unrelated FSH/CHT/restart/test paths; it is an inventory, not an assertion all execute in CHMT.

## Notation

N gas cells; F gas mesh faces; W coupled surface slots; U distinct owner cells referenced by those slots (U≤W); B non-periodic external non-interface faces; S compiled Ns; R=sizeof(chmt::Real)=8; I=sizeof(int); K actual gas RK stages (typically 1 Euler or 3 SSPRK3). M accepted microsteps in one attempted gas window. Retried/replayed windows repeat corresponding work. Byte counts exclude allocator/PCIe protocol/Thrust temporary traffic.

## Current transfer inventory

### A. `Backend.cu:readBulk`, called by `applyCoupledFaces` once per K stage

Producer: recovered GPU primitive and limited gradient fields.
Payload legacy: rho,T,p,Ux,Uy,Uz (6N scalars), 18 primitive-gradient scalar arrays (18N). 24 D2H calls, 24RN bytes.
Payload mixture: above + soundSpeed(N), species rho(SN), species gradX/Y/Z(3SN). 29 D2H calls, R(25+4S)N bytes.
Consumer chain: owner lookup in applyCoupledFaces → evaluateGasWall → interface flux/packets → gas GPU flux scatter; last stage's owner values also become GasIntervalRecord.gasTrace/gasGradient → CpuMaterialDriver and MaterialDrive slab selection → solid/film/reaction/recession updates.
Each `SharedGasDeviceStorage::read` is blocking cudaMemcpy into a fresh std::vector, hence serial CPU-readiness boundary per field. These are not asynchronous overlapped copies.

Minimum fields preserving current CPU behavior:
- Every LowRe stage closure needs p,T,U[3],Y[S], velocity gradient[9]. No direct rho/sound or rho/T/p/Y-gradient use there.
- The final stage's history requires rho and sound as well: MaterialDrive.H compares density and normalizes velocity changes by sound speed. Dropping these would change material local-step selection. GasPrimitive total = 7+S scalars; add 9 velocity gradients → 16+S.
- Legacy species are constant and sound is already computed from p/rho: only 15 actual transmitted scalars per selected owner are needed for history (6 primitive +9 gradients); mixture needs 16+S. Closure-only stages can omit rho/sound if packing Y directly on GPU.
- BoundaryLayer stage closure uses prepared layer/matching, not bulk/gradient. Material surface physics still uses owner bulk p/T/Y; history still needs full primitive+velocity gradient. Do not conflate matching donor state and owner state: they differ. No new RHS or coarse BVP is needed to gather them.
- Surface Arrhenius uses provisional interface density recomputed from p/T/Y, not a hidden need for every interior density.

Classification: sparse/field-reduced, not wholly removable. Most whole-domain bytes are avoidable. U deduplication is optional; W gathering avoids a CPU gather and may be simpler. Both are O(W), never O(N).

### B. `fluxes(...false)` in `applyCoupledFaces`

GPU raw face fluxes → CPU needs interface momentum projection for mechanicalPressure in BoundaryLayer branch; existing code loads every flux field, alters only interface entries, then uploads every field. Legacy 5 calls /5RF D2H; mixture 6 calls /R(5+S)F. Same byte count and 5/6 H2D copies on writeback.
Needed: BoundaryLayer only dot(momentumFlux, outward)/area per W (one scalar could be computed with existing face expression); LowRe needs no incoming gas flux at all. CPU writes new mass/momentum/energy/species flux only for W interfaces. Non-interface values should stay resident and unchanged. Existing full download is mostly a read-modify-write artifact, not a consumer.
Classification: remove full download; sparse W scatter upload; preserve duplicate/interface mapping semantics. For exact arithmetic, transferring 3 momentum components and using current CPU projection is more conservative than changing projection rounding.

### C. `accountCoupledFaces`: second `fluxes(...false)` per stage

GPU positivity-limited face flux → CPU computes interface acceptance scale → stage-weighted split packets/reserve/budget; also sums B external boundary mass/momentum/energy/species/element budgets.
Current another 5/6 blocking D2H calls, R(5+S)F bytes mixture or 5RF legacy.
Needed interface numerator: selects denominator from raw energy, mass, species; does not use current momentum in that scaling. Thus per W need energy+mass+S species at most (or one scale computed using the identical selection arithmetic on GPU). External boundaries need aggregate mass,momentum[3],energy,species[S], not all interior faces; elements can be derived from species on CPU. Could transfer selected B face values to preserve exact original summation order, or GPU reduce (floating summation changes must be tested).
Classification: sparse + optional reduction. Cannot reuse pre-limit B transfer because positivity has modified accepted flux. Its CPU readiness boundary is logically after limiting, distinct from pre-material closure.

### D. `SharedGasDeviceStorage::downloadWallOutputs`

GPU sparse evaluateGasBoundaryLayerKernel → exchange/output/input arrays → CPU prepared wall split packets and matching history/material reevaluation. Successful normal stage preparation: 3 Buffer::download calls and 3 cudaStreamSynchronize(nullptr), bytes W*(sizeof(GasBoundaryLayerExchange<Real>)+sizeof(WallOutput<Real,S>)+sizeof(WallInput<Real,S>)). Failure path: exchange followed by W int status and W WallStatus instead; no ordinary outputs then.
Exchange download currently reads only `.ready`; mass/force/energy/k/pressure are not consumed in this host function. Input download consumes only pressure and matching state; model tables, pointers, quadrature and CPU-supplied wall values are not dereferenced. Matching = T,U[3],Y[S],k,omega, plus pressure: (7+S)R, vs entire WallInput. WallOutput reused fields include traceTemperature, traceVelocity[3], traceY[S], traction[3], conductiveHeatFlux, wallSpeciesFlux[S]. Other fields are diagnostics or GPU source outputs; current entire output is also copied into gasWallDiagnostics, so remove those payloads only if that diagnostic contract is intentionally narrowed.
The mixture preflight prepares the first stage and its generation/time/dt cache is consumed by that stage; do not automatically count an extra first-stage solve/download. Stale/reprepared paths can add calls. Already sparse: no justification for full N download or wall node/workspace download. Can pack readiness + consumed matching/output, one post-producer async copy and one wait. Failure details can remain on-demand. Reuse existing wallFaces_, wallInput_, wallOutput_, exchange backing; do not add a duplicate wall solver. CPU material wall closure can run again for changed Tw/species; it consumes existing history, not new device downloads. Preserve prepared generation/time/dt reuse rules.

### E. `checkStatus` at trial validation points

GPU cell/face transport status → CPU loops only for InvalidStorage → existing failure/rollback. Each call: one cudaDeviceSynchronize +2 blocking D2H copies, I(N+F) bytes. Shared advanceGasTrial invokes validate after flux and at final boundary completion; mixture adds pre-transport and post-source validation. Hence successful direct CHMT legacy has 2 calls per trial (2 device waits,4 copies), mixture has 4 (4 device waits,8 copies). See common/GpuGasAdvance.cuh:615,696,720,724. Native-refinement-only validation at668 is not active in this direct call. Existing handling ignores other codes here. Minimum is reduced first invalid index/code (or success scalar); implement the same existing condition, not additional rejection. Full vector only for explicitly requested diagnostics. Device-wide wait may be replaced by the relevant stream readiness if no cross-stream producer exists; must verify producer ordering. This proposal removes overhead without adding a gate.

### F. `downloadState` on every microstep commit

GPU conservative rho,momentum[3],energy,species and optional SST → CPU trial.gas / trial.sst (multiply by volume) → next microBase rollback snapshot, window-end material candidate, endpoint surface recovery, final conservation/output/checkpoint.
Legacy 5+2T calls; mixture 6+2T calls (T=1 if SST). R*N*(5+S+2T) mixture or R*N*(5+2T) legacy bytes per microstep.
Current host rollback needs an up-to-date microBase; removing this without a replacement is unsafe. Minimum-frequency design: preserve rollback inventory on GPU (existing RK initial storage may not cover all pre-chemistry/geometry/window state; audit lifetime before reuse), retain host geometry/history, download full gas/SST once at window endpoint for current HostState/conservation/output API. CPU material solver itself uses sparse history during its microsteps and finally copies geometryEndpoint.gas/sst, not a full gas RHS. Future fully resident accepted gas would require broader HostState/IO/audit refactoring and is not necessary for first optimization.
`downloadGasWindow` and public `downloadState` currently only copy existing host objects. Do not count them as additional D2H. `commitGasWindow` uploads candidate gas again; currently CPU merely copies endpoint gas, so this H2D roundtrip is also a candidate to avoid under resident ownership, with generation consistency checked.

### G. Courant selection

GPU convective/diffusive/(SST source) fields → Thrust host result → gasCoMaxStep selects actual dt. 2 host-returning reductions, 3 with SST, once per microstep estimate. First result includes unavailable bool; remaining doubles. Scalar payload O(1), traversals O(N), synchronous return on gasCaptureStream (currently nullptr). Must retain current step decision; using lagged maxima changes numerical timing.
Can fuse reductions into one tuple reduction/one host wait, preserving NaN and unavailable semantics. Current Backend-owned scratch allocator already reuses Thrust allocations. SST does four D2D copies of N Real (save 2 +restore 2); not D2H. No full-field Co download is needed. Do not count cudaGetLastError as a device synchronization.

### H. Chemistry and SST audits (HEAD, being removed by others)

Chemistry half-step GPU audit → firstChemistry/secondChemistry → merged HostState diagnostic. Two D2H calls per chemistry trial, 2*N*sizeof(ChemistryAudit<Real,S>) bytes; sizeof includes S+5 Real and 3 ints plus ABI padding. CPU does not integrate chemistry from these downloads; GPU chemistry inventory update is separate. Do not remove chemistry physics/state/status with audit storage.
SST: 13 full N Real audit buffers (13RN device bytes); six N Real D2H reads per accepted SST trial, 6RN bytes, CPU sums → gasSstAudit/budget diagnostic. No reason to download raw per-cell values merely for totals; assigned removal is stronger where output-only. These are distinct from the two true rhoK/rhoOmega state downloads in F, which remain inventory.

### I. Gas-budget opt-in runtime (HEAD, being removed)

Related gasUGKP owner, not CHMT's current direct path: read V,rho,rhoE,species =4 copies, R*N*(3+S) per snapshot, plus accepted transport R*N*(S+2) and chemistry R*N*S. Snapshot invocation frequency depends on begin/commit/output configuration; do not count unconditional. All are reporting/summing, not gas RHS consumers. Review HEAD GasBudgetRuntime.cuh lines 73–90,278–279. Current file may already be deleted by parallel worker.

## Per-microstep baseline cost excluding audits

Mixture D2H bytes: K*R*((25+4S)N+2(5+S)F) + R*(5+S+2T)N + 4I(N+F) + K*wallStructBytes + O(Co scalars).
Calls: K*(29+6+6)=41K blocking read calls plus state 6+2T, status8; if BoundaryLayer add3K downloads and their3K explicit waits; Co adds2+T host-returning reductions. Legacy has4 status copies and2I(N+F) status bytes. Corresponding stage calls24+5+5=34K and bytes K*R*(24N+10F). H2D full flux write adds K*R*(5+S)F. W=0 does not currently skip readBulk/fluxes, so even uncoupled/no-interface stages deserve a separate fast path if API permits; don't assume W>0.
For N=1e6,F=3e6,S=5,K=3,T=1 mixture baseline above excluding sparse walls = 2.680 GB D2H per microstep (decimal), plus 720 MB full flux H2D. This is an illustrative size, not the user's measured mesh or bandwidth. Computation: stages 3*8*(45e6+60e6)=2.52 GB; state 96 MB; status 64 MB → 2.680 GB including status.

## Motion, material and radiation

Moving mesh: prepareCoupledGeometry computes stage meshes/swept volumes/surface on CPU from host points; finishCoupledGeometry uploads geometry. No separate geometry D2H exists in current CHMT. Recession/film state is CPU-owned, driving WallProgram. Full gas primitive/gradient download is not necessary to compute all mesh points; only interface pressure/shear/mass/heat drive is needed. Static topology maps may survive geometry changes, but stencils/weights/owner distance/quadrature need invalidation when geometry changes.

CHMT radiation: InterfaceMath::grayRadiation produces CPU local surface radiation flux → wall program → accountCoupledFaces integration → material solid/film energy. No current volumetric radiation mapper D2H in CHMT. Do not label CHT's Mie/DO path as CHMT.

Related CHT radiation/mapper (current files changed since historical base): GpuSolidThermalCoupler.C:1769 full resident.downloadToHostMirror on radiationDue, then mobile-only radiation sums and occupied area. radiationDue is exchangeSequence % radiationCouplingFrequency==0, after thermal event due check; not every gas stage.
- Full mirror API has 23 cell scalar fields plus boundary fields/possible restart mirrors. The actual radiationSnapshot uses Tgas, particle mass, mean T, diameter, emissivity, heat factor, elapsed interval. Thus radiation alone only needs gas T plus particle radiation moments, not all 23 gas/particle field arrays. Other consumers/write/restart events may reuse an already required mirror; check combined scheduling before removing it.
- DownloadMobileParticleRadiationSums: 3*N device-real scalars copied to host doubles (conversion implementation may add staging; runtime precision dependent). GPU computes mass, mass*T, mass*diameter; CPU divides and uses all occupied-cell results in CPU Mie/DO solve. Whole volume may genuinely be required by volumetric radiation; not reducible to wall data. Sparse populated cells + host zeros possible if density sparse; packing cost must be measured.
- DownloadParticleWallOccupiedArea: 2*F device-real scalars (represented area and scale), CPU product, only coupled patch face values used to reduce wall emissivity. Can multiply on GPU and gather selected mapped wall faces: Wc scalars, one transfer, no change to CPU DO solver.
- gas wall energy previews and deposited/reflected particle heat ledger previews → mapper combines integrated face energy onto solid faces → CPU solid update; sparse patch exchange is genuine. Internal F-wide buffers can be gathered to actual wall faces. Preserve preview/commit ledger transactional timing; do not reset ledgers early.
- GPU particle enthalpy refresh after radiation has cudaDeviceSynchronize (GpuResidentStrict.cu ~5507), no D2H itself. Can scope wait to dependency/stream only after ordering analysis.
Mapper cannot justify a full gas-volume mirror; its actual input is integrated selected-face energy. Moving CHT DO or material conduction to GPU is a separate architecture project, not needed for narrow payload removal.

## Minimal implementation design (not implemented)

1. Retain existing wall/surface face map. Allocate reusable compact per-W stage host/device buffers (pinned host if overlap is desired), upload map only on mapping/topology changes. Prefer producer-side compact write in existing wall evaluation/face operator if it can preserve correct sampling time without contaminating shared ABI; otherwise one general gather kernel. No matching primitive gather exists in current CHMT shared backend; original window drive storage is historical and cannot simply be called now. Existing common boundary mirrors are F-indexed primitives only, not needed gradients/flux scalars.
2. Pre-closure gather: LowRe owner p,T,U,Y,velocity-grad; BL matching/output +interface pressure projection. Last actual RK stage also gathers history primitive+velocity-grad (rho/sound retained). Pack/read once after all producers, then CPU closure. CPU closure returns W fluxes; upload once and scatter only mapped faces. This is one added scatter kernel if no existing operator output hook can consume compact fluxes. No whole-F roundtrip.
3. After positivity: gather only selected interface accepted numerator values and B external face flux (exact host order) OR reduce boundary sums on GPU after numerical-equivalence review. One readiness wait for packet/accounting consumer. Cannot reuse pre-limit sample.
4. Wall outputs: combine readiness, consumed input matching and output into same pre-closure packet; no new BVP. Node workspace remains GPU resident. Diagnostics fetched only when their existing output contract requires them, not silently dropped.
5. Status and Co can be scalar reductions. Keep established CPU scheduling decisions; do not add contracts, gates or new failure behavior. Reuse transfer allocations across stages. Validity must track stage/time/dt, species mode/SST configuration, geometry/stencil generations, and rollback; snapshots never outlive their producer generation.
6. Deferred state download needs GPU rollback snapshots for full trial composition including chemistry; macro accepted snapshot separate from RK stage scratch. Synchronize full host endpoint only once per completed attempted window, not per gas microstep. Keep output/checkpoint cadence and failure recovery semantics.

## Required validation before implementation claims

Source-count tests are not performance evidence. Use a CUDA API trace and Nsight/CUPTI or equivalent to count memcpy bytes, waits and actual staging; distinguish pageable copies, conversion buffers, graph behavior and error paths. Compare exact state/packet/budget/restart/moving-geometry histories for LowRe and BoundaryLayer, single/frozen/reacting gas, SST on/off, Euler/SSPRK3, wall-free/multiple surfaces/repeated owner maps, moving mesh/recession, replay/rollback and write events. Measure N/F/W scaling and total step time; do not promise speedup from byte ratio alone.
