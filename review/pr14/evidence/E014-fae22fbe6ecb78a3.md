# CW-1: native inverse bracket nonprogress repair

Status: source repair and refreshed native compile-only evidence complete; independent re-review and real-GPU acceptance pending. This v2 folder supplements the unchanged v1 freeze and independent review.

## Request and source proof

Independent review CW-1 identified the newly admitted parameter `meltingTemperatureK = -INFINITY` with finite mushy range 20, solid cp 1273, latent heat 1.16e6, and finite specific enthalpy 1. Both inverse early branches are false, while the initial upper bound and its forward enthalpy are negative infinity. The original `upper *= 1.5` cannot change that bound; both bracket-growth predicates remain true indefinitely. This conclusion follows from the source and IEEE branch behavior; no numerical reproduction was executed under the user's revised validation scope.

## Exact repair

Only `common/wall/GpuColdWallSolidification.H` changed from v1. The bracket loop saves `previousUpper`, executes the exact original `upper *= GPU_R(1.5)`, and breaks out of that growth loop when the new bound equals the previous bound. Execution then enters the unchanged forty-bisection loop and native final return. No refusal, default temperature, synthetic profile, rollback, numerical range, or iteration-count limit was added.

Ordinary advancing brackets execute the same multiplication and revisit the same original predicates. Their forty-bisection arithmetic and order are unchanged. A stationary finite bound, zero, or negative infinity could not have made progress in the original loop; these newly completed paths proceed through existing inverse/profile/phase/history/energy arithmetic. For the CW-1 parameters and direct H=1 inverse, those forty native bisections retain negative infinity, so the inverse result itself remains the native negative-infinite value rather than an invented finite substitute.

Lineage:
- v1 SHA256: `fbd74acbde2fc6685369c9da4f816ac9c4a637b52eca3caf8fafc03410a91480`
- v2 SHA256: `391cbfc0507a3b1b2af027139a6166f798013fd4cbdc38ae3d258e536215da5b`
- Minimal delta: `CW-1.patch`; complete seven-file task delta: `production.patch`.

## Compile-only validation and dependency scope

Four refreshed real nvcc sm80 compilations succeeded (exit 0), 16.66 seconds total compiler wall time:
1. Actual FSH 1D/2D group instantiations, `-c`.
2. Actual CHT paired 1D/2D group instantiations, `-c`.
3. Full selected CHT 1D/2D wrapper headers including input and ledger branches, `-dc`.
4. Full selected FSH 1D/2D wrapper headers, `-dc`.

Four additional dependency-only preprocessing commands succeeded. Their `.d` files pin the actual transitive project headers in each variant. `compile-dependency-manifest.json` records per-variant inputs, hashes, and differences against v1. The only changed actual compile dependency is `common/wall/GpuColdWallSolidification.H`. Unchanged dependencies include the particle worker's material, finite-contact, capillary, wall-interface, and cold-carry helpers; their exact versions are pinned, not inferred from another worker's current files. Generated group/wrapper inputs and separate logs/commands/objects remain in `nvcc/` and `nvcc-wrappers/`.

Warnings #20013-D for existing shared host constexpr numeric_limits calls remain visible in logs. This is selected-header/device compilability, not a full application link: the wrapper probe models touched DeviceState storage and declares external gas helpers. `git diff --check` passed for the repair. No new CPU simulation test, CPU numerical execution, numerical parity run, or GPU execution was performed.

## Fast real-GPU acceptance scenario: CW-1 (user-owned)

Use fully allocated real owner data and finite geometry/time inputs. Set cold solidification melting temperature to negative infinity, mushy range to 20, solid cp to 1273, latent heat to 1.16e6, and all eight/64 stored enthalpy values to 1; use finite remaining parameters and a short finite step such as 1e-9. Keep the actual profile, carry, history, temperature, and wall/gas ledger buffers, and seed output sentinels distinctly from native results.

- Invoke the actual inverse on device with H=1: it must terminate after one nonprogressing growth multiply followed by the original forty bisections; source predicts negative infinity, not a finite fallback.
- Exercise FSH cold1D8, gas-only, and cold2D64 owners and their public wrappers: control must reach actual native profile/history/temperature and applicable ledger publication, with IEEE results determined by those formulas. Zero conductance/no-active-contact identity branches remain authoritative; do not demand profile mutation where the original route is an identity.
- Check that actual history advances by the selected existing interval and that affected temperature/energy/profile values are computed, rather than unchanged output sentinels or fabricated zeros. For gas-only with disabled source, unchanged enthalpy can be the genuine identity while derived temperature/history still publish.
- Retain the original successful finite-parameter GPU controls and compare every FP64 result/substep/PCR/bisection output bit to the frozen baseline. The paired CHT alternative inverse is bounded separately, but the CHT owner build and usual successful-path controls remain required.
- Run compute-sanitizer and owner synchronization/error checks. Do not deliberately access short/out-of-bounds backing.

The complete v1 real-GPU acceptance matrix remains applicable, including exceptional remaining-interval arithmetic, genuine missing-backing/CUDA errors, CHT float-only profile caches with double carry, FSH all-double physical state, full owner linking, and application/performance acceptance. Runtime acceptance is the user's responsibility under the latest request, not established by these compile results.
