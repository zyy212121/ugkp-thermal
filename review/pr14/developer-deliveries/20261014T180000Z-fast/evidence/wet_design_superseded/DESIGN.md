# FAST4 wet mechanics: verified cause and minimal quasi-static repair design

Status: read-only design, 2026-10-14. Implementation and publication are not performed. The proposed host-history interface extension remains a decision for the owner. No new runtime rejection, stop, rollback boundary, README edit, or PR Conversation change is proposed.

## 1. Decision

The retained stationary wet failure does **not** require adding evolved film momentum. Its measurable cause is inconsistent time quadrature of the existing top and bottom traction: the gas applies 50 actual pressures per material window, whereas the CPU SolidFilm packet applies the first representative pressure for the entire window. The selected local repair is narrower: integrate only the existing EOS-pressure factor using actual gas-stage Euler/RK time weights, retain the CPU material step's existing solid area/normal and freshly evaluated shear/Gδ closure, then retain the current SolidFilm recipient/support route. It does not change the pressure model or replace thermal phase terms.

This conclusion is specific and stronger than “the reduced model cannot conserve momentum”: here film velocity, phase mass, body force, shear, and geometric motion are all zero, so the inertia-free local equation itself requires equal integrated top and bottom normal traction. The current implementation violates that equation through its quadrature.

This is a PR14 problem. The CHMT files are absent from PR14's merge base `575b7dd3d30cc648142914d6e81765a43e04222c`. The algebraic profile already appeared in the earlier PR14 source commit `02ae6c9`, and the relevant mechanics is unchanged between preceding production `4b5c480b` and tested `9d5be6ef`. Thus it is not introduced by the latest zero-mass density fix, but it cannot be described as unrelated legacy outside this PR.

## 2. Evidence and independent numeric attribution

Tested production: `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`.
Reviewer evidence: `0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668`, directory `review/pr14/reviewer-deliveries/20261014T120103Z-fast/`.

The accompanying `wet_impulse_decomposition.json` and `analyze_existing_wet_impulses.py` use exact Decimal conversion of retained binary64 trace values. They do not rerun the solver or change a verdict. `checkpoint_1_full_model_state.json` uses the native internal checkpoint codec. Its input SHA256 `d4fa517cbe2cfb884b19bfb266461c7102ba4a64491a688736eb253cb92d3562` exactly matches the formal checkpoint-1 hash recorded by the reviewer; the reused older retained binary is therefore the same bytes, not a merely similar case. Only its full decode is evidence here; unrelated narrow operator calculations printed afterward by that helper are not used.

Confirmed from actual serialized model and geometry:

- Four film faces, each area 0.25 m²; gas face-area vector (-0.25,0,0), solid vector (+0.25,0,0), film normal (+1,0,0).
- Static geometry; gas viscosity 0; gravity 0; prescribed pressure gradient, top traction, base velocity, mesh velocity all zero.
- Each window lasts binary64 0.0005 s and has 50 gas stages. All actual limiter scales and ledgerDt/stagePacketDt factors are 1.
- All 400 GasFilm records and all eight SolidFilm records have zero mass transfer. Wall velocities and film mean velocity are zero. There is no mass recoil, viscosity, area-ratio, normal-direction, body-force, or limiter explanation for this witness.

| Window | Bottom effective p [Pa] | First gas-stage p [Pa] | Actual top time-mean p [Pa] | Four-face bottom minus top Jx [kg m/s] |
|---|---:|---:|---:|---:|
| 1 | 178167.05610328369 | 178167.05610328368 | 178166.87577935919 | 9.016196225419826e-5 |
| 2 | 178166.70454359697 | 178166.70454359698 | 178166.56922616288 | 6.765871704628168e-5 |

For each face, actual bottom impulse minus first-stage-pressure × area × whole material duration is only +1.9133e-15 and -1.1953e-15 kg m/s. Cumulative bottom-minus-top impulse is 1.5782067930047994e-4 kg m/s. Its sign is opposite the reported gas+physical-film+boundary+support residual; the ~7e-14 difference is the retained cumulative FP64 accounting remainder. Independent final-stage gas physical balance already closes near 4e-15. This identifies the 1.58e-4 term, rather than attributing it to an accounting residual.

## 3. Existing equations and signs

Let n point from solid to gas, A be the flat conformal face area, δ film thickness, τ the gas-side tangential traction on the film, and G = grad_s(p) - ρ g_s.

`film/FilmMath.H:17–35` computes the instantaneous Stokes/lubrication profile:

    u(z) = u_b + (τ-Gδ) z/μ + G z²/(2μ)
    mean(u) = u_b + τδ/(2μ) - Gδ²/(3μ)
    τ_b = τ - Gδ

This is μ u''=G with no transient or advective momentum evolution. FilmQ contains mass/species/enthalpy, not momentum. Normal traces are prescribed from geometry/mass kinematics.

The actual GasFilm packet is positive into gas:

    J_g,total = m_g u_trace + J_g,stress
    J_g,stress = ∫ (p_mech n_g - traction_g) A_g dt

Film receives its negative. The existing SolidFilm packet is positive into film; solid/support receives its negative:

    J_b = ∫ [p_b n_s A_s - τ_b A] dt
    accountCpuPacket: supportImpulse += -J_b

For the flat conformal reduced model, the physical same-stress relation would be (this is not a proposal to import extra normal-viscous terms into the existing closure):

    J_b,stress = J_g,stress + ∫ G δ A dt
    J_side_pressure + J_body = -∫ G δ A dt

Therefore the inertia-free local stress balance closes without a new state. In this witness G=0 and mass=0, so J_b must equal J_g,stress. The current CPU instead uses p(first) A Δt.

Do not replace J_b with an endpoint/global residual. Do not add film.exchangeMomentum to supportImpulse. Do not equate the whole gas packet with stress when m_g is nonzero. Do not delete the Gδ term in shear/gravity/pressure-gradient cases.

## 4. Source path of the mismatch

- `gpu/GasWallMath.H:91–113`: constructs real mass-advection, pressure and viscous impulse; top energy retains pressure/viscous-work decomposition.
- `gpu/Backend.cu:290–307`: scales actual stage packets with finalized flux scale and ledgerDt/stagePacketDt, then integrates into GasIntervalRecord.
- `coupling/IntervalContracts.H:49–60`: gasTrace/gasTraction are one representative stage per gas record; they are not complete quadrature history.
- `IntervalCursor::takeThroughLocal`: already traverses the consumed records, slices extensive packets, and preserves the final remainder. It is the natural place to slice additional extensive mechanical integrals.
- `materials/CpuMaterialDriver.C:292–304`: chooses the first record covering elapsed; ordinary gas micro-boundaries do not determine the material step.
- `ablation/CpuSurfaceInterface.C:57–69`: independently creates bottom impulse from that representative constitutive pressure and bottomShear times the whole dt.
- `CpuMaterialDriver.C:35–39,548–559,616–619`: existing reciprocal SolidFilm accounting and accepted transaction commit are already present. They should remain the sole support-impulse consumer.

For exact fixed-source reading, use the tested SHA, not the concurrently edited working tree.

## 5. Selected minimum: one scalar EOS-pressure time integral

This is the sole preferred proposal, pending interface authorization. Earlier 56-byte stress decomposition / 64-byte dual-pressure proposals are **not adopted**. They carried quantities not needed to repair the demonstrated pressure quadrature and could invite an unintended mechanical-pressure, normal-viscous, or thermal model replacement. The intermediate three-component mapped-bottom-drive proposal is also not adopted: importing gas-stage traction would replace the CPU closure recomputed at the material top temperature, not merely integrate the existing pressure factor.

Add only:

```cpp
// Transient, per surface face, in GasIntervalRecord:
std::vector<Real> eosPressureTime; // sum_k ledgerDt_k * p_EOS,k [Pa s]
```

One binary64 scalar per face per gas record, eight bytes. No mass-carried momentum, pressure-force vector, shear, mechanicalPressure, EOS/temperature override, or bottom-work field. No persistent FilmQ, HostState, Budget, ExchangePacket, checkpoint format or restart state extension.

### Exact pressure source

In `accountCoupledFaces`, `faceRates[i].trace.pressure` is available from that same actual stage. `makeInterfacePrimitive` sets `state.pressure=pressure` without changing it when solving wall composition/temperature:

- LowRe: this is the gas bulk EOS pressure supplied to the closure.
- BoundaryLayer: this is the matching EOS pressure, also retained in `preparedMatching[i].pressure`.

The CPU existing bottom formula uses `physics.gasState.pressure`; `CpuSurfaceInterface` replaces the input bulk pressure with matching pressure for BoundaryLayer. These are the same pressure **semantic**, unlike `preparedMatching[i].mechanicalPressure`, which must not be substituted. The time samples differ intentionally: all real gas stages replace the first representative stage for the force factor only.

### Exact time weights, deliberately no limiter multiplier

Accumulate:

    I_p,f += ledgerDt * faceRates[f].trace.pressure

`ledgerDt` already contains the actual Euler/RK time weight. `common/GpuGasAdvance.cuh:132–149` calls this accounting callback after final face-flux scaling; `GpuGasHostPolicy` supplies its RK weights. Do not use microDt once per stage or sum all stages with equal dt.

The per-face final flux scale α is a different quantity, not another RK time weight. Existing gas packets use it, but the old CPU bottom EOS-pressure force does not. Therefore this narrowly selected correction does **not** multiply I_p by α. Doing so would add a new coupling rule beyond time quadrature. All actual α in the failing wet witness equal 1, so this distinction does not affect its repair. Cases with α≠1 retain the model's pre-existing gas-versus-bottom force discrepancy and cannot be claimed fully momentum-closed by this patch.

### CPU force formula and unchanged inputs

For the material substep [a,b], after the existing cursor has returned its pressure slice I_p:

    old: J_phase = n_s * (p_first * A_s * dt) - bottomShear * A_ref * dt
    new: J_phase = n_s * (A_s * I_p)           - bottomShear * A_ref * dt

All other quantities remain exactly from the existing CPU evaluation:

    shear = (A_g/A_ref) P_f P_g traction_CPU
    bottomShear = P_s(shear - G*delta)
    G = grad_s(p) - rho*g_s

Here P_n(v)=v-n dot(v,n). The ordering P_s P_f P_g matters for unequal normals. `traction_CPU` is recomputed using the material constitutive temperature; substituting the earlier gas-stage traction would change the physical drive sampling. Do not do that in this patch.

EOS pressure supplied to temperature recovery, film pV conversion, phase rates, phaseAdvective, phasePressureWork, pressure gradient and profile recovery remains unchanged. I_p is consumed only in the normal force product.

## 6. Geometry: what one scalar proves and what it preserves

`CpuSurfaceInterface.C:45–46` obtains A_s and n_s from the actual CPU evaluation HostState solidMesh and its surface.solidFace mapping. The replacement continues to use those exact values for the entire material substep. It is the pressure-time quadrature under the **same material-step geometry approximation** as the old formula. It is not a switch to endpoint geometry, a recomputed mesh, or a gas-grid face ID.

Actual gas-stage geometry is present, but it is deliberately not added to this payload:

- `Backend::prepareCoupledGeometry:195–219` constructs both gasStage and solidStage. Each stage's areaVector is `microBase.*Mesh.areaVectors`, the gas microstep's old/evaluation geometry.
- The gas and solid endpoint meshes are held separately. `finishCoupledGeometry:222–227` replaces `trial.gasMesh/solidMesh/surface` only after the gas transport stages.
- Thus inside `accountCoupledFaces`, `b.trial.solidMesh.areaVectors[surface.solidFace[f]]` and `b.solidStage.areaVector[...]` refer to the current microstep's real evaluation geometry, not its newly generated endpoint. No new geometry transfer would be necessary if a broader mapping were later requested.
- Across microsteps these vectors can change. In general, A_s,CPU n_s,CPU * sum(p_k dt_k) is not equal to sum(p_k A_s,k n_s,k dt_k). The former is selected because it preserves the existing CPU material geometry quadrature; the latter would require at least one Vec3 pressure-impulse moment and would change that discretization.

For the actual stationary planar witness, all these area vectors are identical, so the scalar is sufficient for the exact intended normal force integral. For moving or nonplanar interfaces, the patch preserves the old geometry model; it must not be advertised as integrating all geometric pressure loads or closing all global physical momentum. No new rejection or stop for those existing paths is proposed.

## 7. Minimal function/data path, fused into existing loops

1. `coupling/IntervalContracts.H`: add the scalar vector on GasIntervalRecord. Add corresponding cursor `used` storage and an optional/new overload output for the pressure slice. Keep existing packet and radiation arithmetic/order unchanged.
2. `gpu/Backend.cu::accountCoupledFaces`: allocate the per-face vector once with the record and add `ledgerDt * faceRates[i].trace.pressure` in the existing surface-face loop. No extra pass over faces, new kernel, gas primitive download, closure solve, or stored stage snapshot.
3. `IntervalCursor::takeThroughLocal`: while already visiting each record, scale I_p for partial slices; for the final slice of a record use original-minus-used. Reuse candidate cursor copying so retries are atomic and successful consumption advances once. No separate history search. Preserve the existing packet/radiation call overload for callers that do not request pressure metadata. That overload must use the same internal consumption implementation and merely discard the new output: pressure used/remainder state must still advance with the cursor, so mixing overloads on one cursor cannot duplicate pressure consumption.
4. `CpuMaterialDriver`: ask the existing trial cursor for the pressure slice and pass it into the actual surface evaluation. Keep the representative bulk, thermodynamic data, CPU material clock, shear recomputation and source-refinement decisions as before. Response-only/predictor evaluations that have no consumed history retain their old point-sample behavior.
5. `CpuSurfaceInterface.H/.C` and the small `CpuMaterialDriver::Implementation::evaluate` forwarding wrapper: an optional pressure-integral vector can be passed solely for final CPU-internal SolidFilm packet construction. At the original construction site, replace only `physics.gasState.pressure * dt` in its normal momentum term. The same inputs still feed every energy term. Construct the corrected normal term directly; avoid a post-hoc global balance correction or subtracting nearly equal complete impulses to recover shear.
6. Retain the existing `accountCpuPacket` and accepted ledger transaction as the only consumers of the corrected SolidFilm impulse. Do not add a support-impulse adjustment after accounting. No changes to checkpoint codec, audit thresholds, runtime gates, or end times.

Production Backend records always fill the new field. Existing isolated fixtures/proposal-only callers need an explicit compatibility path or updated fixture input; no new missing-field rejection is part of the design. The implementation should preserve the old overload/default behavior where no real gas history was supplied, rather than making a previously legal calculation stop.

## 8. Energy and scope

The current witness has zero phase mass and zero boundary/material velocity. Its corrected normal reaction does zero work, so keeping all thermal quantities untouched is fully consistent and preserves the original heat/phase trajectory. This is the verified repair scope.

For the wider existing model, the design deliberately does not change these semantics:

- `solidWallSpeed` is phase-front/geometric sweep speed, not automatically rigid-skeleton material velocity.
- `phaseAdvective` includes mass times (condensed energy + p_EOS/rho_s).
- `phasePressureWork` uses p_EOS A_s frontSpeed, while phaseViscousWork is zero under the stationary-rigid-skeleton assumption.
- Liquid bottom trace and kinetic advection are separate. Top gas pressure/viscous work already use their actual stage quantities.

Pressure-advection and phase-front work can cancel for a stationary nonporous skeleton with w=-j/rho_s. Changing those terms or multiplying the new reaction by front speed would disturb that closure. This patch does neither.

Therefore it remains an honest pressure-force time-quadrature repair with old energy and shear approximations preserved. It does not establish arbitrary moving/evaporating interface jump conditions, add neglected film inertia, introduce normal viscous pressure, equate EOS and mechanical pressure, or certify exact transient Δ(M meanVelocity). Those are separate physical/discretization questions, not prerequisites to fixing the demonstrated static error.

## 9. Cost and FP64 impact

At four faces and 50 gas records/window: 4×50×8 = 1,600 bytes of new scalar values, plus a vector object/record (commonly 24 bytes on this target). Cursor used/output vectors cost O(interface faces). Existing history copies duplicate the extra values just as they duplicate existing packets; actual peak memory should be measured rather than inferred as a performance pass.

New work is one multiply/add per existing gas-stage face visit and scalar scale/add/subtract per existing cursor face slice. No full-domain traversal, GPU kernel, device allocation, extra transfer, gas-step forced alignment, or separate history pass.

Old gas, chemistry, heat, phase mass, energy, shear and pressure-gradient expressions retain their arithmetic order. Normal SolidFilm momentum and its derived mechanical budgets intentionally change. Summing multiple binary64 pressure samples need not be bit-identical to p_first*dt even for constant p, so constant-pressure equivalence is checked against the documented arithmetic tolerance rather than promised bitwise. Existing strict full-state restart failure remains a separate issue.

## 10. Validation before claiming the current witness fixed

1. Unchanged original wet case to t=0.001: both windows, original tolerances and actual gas-stage pressure inputs. Show the independent pressure integral predicts each corrected bottom normal impulse before budget accounting. A numeric derivation is not a formal rerun PASS.
2. Constant and time-varying EOS pressure; multiple Euler/RK weights; partial material slice, final remainder, rejected candidate/retry, accepted commit once. Check the new scalar changes no existing packet/radiation consumption.
3. Distinct EOS and mechanical pressure in a BoundaryLayer fixture: the new scalar must follow EOS, not mechanical pressure. Nonzero mass-carried momentum must never enter it.
4. Nonunit α test: verify α is deliberately not applied and retain any corresponding pre-existing force mismatch; do not assert universal momentum closure. This prevents accidental scope drift during implementation.
5. Nonparallel/unequal-area/moving geometry fixture: verify use of the CPU material evaluation A_s n_s and exact preservation of existing shear projection order. Do not compare against a different gas-stage geometry quadrature and call it the implemented rule.
6. Positive/negative phase controls, zero-mass nonzero heat/force, nonzero shear and pressure gradient: old mass/energy/thermal outputs and CPU closure inputs remain unchanged. No density lookup on an unselected zero-mass phase.
7. Transactional and restart tests confirm integral consumption determinism without new persistent state. Native test must exercise the production helper/consumer, not a separate formula-only surrogate.
8. Retain all unrelated FAIL/BLOCKED/NOT_RUN findings, full transient physical-P limitations and strict restart failures until independently resolved. No weakened thresholds or new gate.

## 11. Deliverables

- `DESIGN.md`: selected one-scalar proposal and explicit limits.
- `wet_impulse_decomposition.json`: exact trace attribution and verified geometry/model facts.
- `analyze_existing_wet_impulses.py`: read-only arithmetic reproduction.
- `checkpoint_1_full_model_state.json`: actual native model/state decode from the hash-equivalent formal checkpoint input.

The earlier 56B/64B and 3-vector designs are superseded, not parallel recommendations. Interface implementation remains unapproved and no production source has been changed by this design task.

Independent read-only review: the selected scalar formula and sections 5–7 data path were cross-checked against the tested native source. The review's geometry line citation and mixed-overload cursor-remainder clarification are included above. This is a design review, not implementation verification.
