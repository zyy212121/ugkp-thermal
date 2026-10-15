# Fused native compiler costs

## Verified build and provenance

The standard CHT build passed for both physical32/time64 and physical64/time64,
including the backend, static library, precision dispatch, OpenFOAM10 frontend
and `bartzSolidReplay`. This establishes compilation/linking, not GPU execution.

- Source: `cd5074ba21a9048a5ceef4cca66909a430d288a0`
- Source tree: `8f4a88066267ef65c2ee68c51871a0798805703b`
- Command: `bash applications/CHT/tools/build_cuda_solver.sh`
- CUDA13.1, target `sm_89`; C++17, `-O3 -lineinfo --fmad=false`
- Exit0; wall time172.75s; maximum descendant RSS790140KiB
  (`getrusage` child maximum, not concurrent-process sum)
- Source stayed clean and unchanged during the build.
- Log: `chmt_completion_20261011/native-fused-cht-cd5074b/build.log`
- Log SHA256: `ea341c5abafd5176c153e192b1a8e55f7311eadb53f2ddabb540f18c5b4d2140`
- CHT executable SHA256: `97953bcfdb257858dd02865ad8be226219833553112ce14cb9442290806c5874`

This pin includes the hot-path fusion and heat-disabled specialization. The later
restart-only validation-loop fusion is integrated as `43da9c8` (`dc25e4d` on the
energy branch). Its complete strictAdvance fixture compiled in both precisions:
2 tests passed,14 deselected,113.32 s. Evidence:
`native-fused-full-advance-43da9c8/result.json`; log SHA256
`319fb990218c94f271d0200255fdbdadcdcbadc88fff41dbd357cbdc14430c74`.
The standard application hashes above remain specific to `cd5074b`.

The earlier failed mixed-precision SST build remains separately recorded in
`native-final-cht/build.log`; it is not substituted for the successful build.

## Selected ptxas resources

Each value is **registers / stack bytes / spill-store bytes / spill-load bytes**.
These are compiler reports, not measured dynamic traffic or runtime timings.
The parser found145 complete physical64 and150 complete physical32 entry records.

| Kernel | Physical32 | Physical64 |
|---|---:|---:|
| Existing pool clear, with fused heat clear |16 /0 /0 /0 |16 /0 /0 /0 |
| Mobile correction, heat disabled |31 /0 /0 /0 |50 /0 /0 /0 |
| Mobile correction, heat enabled |80 /0 /0 /0 |82 /0 /0 /0 |
| Selected-stuck correction, heat disabled |62 /32 /0 /0 |80 /40 /0 /0 |
| Selected-stuck correction, heat enabled |84 /32 /0 /0 |92 /40 /0 /0 |
| New accept paired heat |46 /0 /0 /0 |46 /0 /0 /0 |
| New settle heat + gas recovery |52 /0 /0 /0 |48 /0 /0 /0 |
| Mobile relaxation, either drag specialization |92 /32 /0 /0 |98 /40 /0 /0 |
| Wall-bound relaxation, either drag specialization |102 /32 /0 /0 |112 /40 /0 /0 |
| Cold-wall1D |40 /448 /1180 /2612 |150 /40 /0 /0 |
| Cold-wall2D |197 /32 /0 /0 |216 /40 /0 /0 |

The standalone heat-clear and candidate-reduction kernels are removed. Compared
with the same-toolchain unfused `50237c4` build, pool-clear resources and both
heat-disabled correction variants are unchanged. Enabled correction adds
49/32 mobile registers and22/12 selected-stuck registers in32/64 respectively,
without new spills. The initial runtime-branch fusion imposed that cost even
when heat was disabled; compile-time specialization removes it for that path.
Kernel/resource counts do not imply an executed launch count.

### Existing versus increased cold-wall1D spilling

The preserved earlier native baseline reports:

| Precision | Earlier registers / stack / stores / loads | Final |
|---|---:|---:|
|32 |40 /232 /516 /1208 |40 /448 /1180 /2612 |
|64 |146 /40 /0 /0 |150 /40 /0 /0 |

Thus physical32 already spilled; this repair adds216 stack bytes and increases
reported spill stores/loads by664/1404 bytes. It must not be described as having
no compiler-resource overhead. The unchanged physical32
`__launch_bounds__(256,6)` keeps a tight register budget while double caloric,
PCR/conduction and inlined carry/publication scratch increase live values. This
is a likely code-level explanation, not an isolated runtime attribution.

The old logs are `chmt_energy_evidence_20261011/native/base32-green.log` and
`base64.log`. The former SHA256 is
`745643a72fc297d07a4b71adc085935049aad8c97a946f2233b8c6dde0aec6da`.
All12 checked cold1D/alumina/solidification/frontend source files match
`e9e5fa3` after resolving managed symlinks. Historical records give the same
CUDA13.1/sm89/C++17/O3/fmad=false settings, but the exact old argv was not separately
captured and unrelated gas operators were older. This is preserved cold-kernel
baseline evidence, not a freshly compiled whole-tree matched experiment.

## Memory and scheduling cost

Let N be cell count and P allocated particle capacity. Additional device arrays:

- CHT32 gas authority `rhoE/rhoENext`:8N bytes; CHT64:0.
- Live/compact particle enthalpy carry:16P bytes in both precisions, including
  configurations with gas heat disabled.
- Enabled paired-heat coefficient/budget scratch:16N bytes. Capacity scratch
  reuses dead `rhoENext`; no extra RK/backup array or double node profile.
- Lazy radiation payload:another8N bytes in CHT32; CHT64 unchanged.
- Four additional resident-descriptor pointers:32 constant bytes.

Active heat therefore adds24N+16P bytes for CHT32 or16N+16P for CHT64, excluding
lazy radiation and descriptor bytes. CHMT/gasUGKP already-double energy adds no
persistent energy array. Checkpoints add8 bytes per saved particle; host handling
can also hold an8P-byte carry vector.

The active source uses the existing pool-clear and post-collision correction
traversals for heat preparation. The remaining acceptance and settlement cell
launches replace the old material-heat and standalone recovery pair: **zero net
added traversals or launches**. Their global producer/consumer boundaries remain.
Heat-disabled configurations omit these two heat launches and the16N scratch,
using the original compiled correction body. Publication is fused into existing
relaxation; settlement includes gas recovery. The transaction adds72N bytes of logical scratch reads/writes, three
double atomic additions per eligible candidate and one per publishing particle,
plus existing-input reads and carry traffic. Double arithmetic/inversion also
increases. The emitted old material-heat definition is not an extra executed call.

Restart upload also merges carry validation into its existing particle check;
no additional restart-upload scan remains. Schema8 codec field validation and
FP32 precision-rebase scans remain separate low-frequency host operations.

**GPU runtime: NOT_RUN.** No timing, occupancy, atomic-order or memory-transaction
measurements support a speed claim. Launch bounds were not changed speculatively.
