# Diagnostic transport and completion boundaries

## Ownership

`common/GpuDiagnosticsOwner.cuh` owns one CUDA diagnostic allocation per resident
backend. CUDA `DeviceState` and common gas views carry its non-owning pointer.
The helper processes do not include or link OpenFOAM. Production host wrappers
register the sink from `DiagnosticsFoam.H` after resident creation.

The allocation is initialized once at creation. Neither asynchronous advance,
IPC requests, nor timestep boundaries reset it. First-fatal evidence remains
sticky. Warnings retain one sample and a saturating count per category; host
readback emits count deltas, including overflow notification once. Output
suppression does not change device work or collected evidence.

## Existing completion points reused

For gasUGKP, FSH, and CHT, a bounded diagnostic companion copy is performed:

- after the existing Courant reduction has returned its scalar to the CPU;
- after existing exported field/species/boundary/restart downloads;
- immediately after existing successful `cudaDeviceSynchronize` calls in
  resident setup/configuration/reset paths;
- during teardown, after the original `cudaFree` calls have completed pending
  work and before diagnostic storage itself is freed.

The shared gas trial policy also consumes diagnostics at its existing validation
boundary. There is no newly introduced `cudaDeviceSynchronize` or
`cudaStreamSynchronize`, no per-cell/face/occurrence transfer, and no diagnostic
readback from an asynchronous graph launch. The companion copy adds bounded
bytes at an existing CPU/GPU communication point. Ordinary physics array
allocation and layout remain unchanged.

Device work suppression and detection are separate responsibilities of the
instrumented operators. This transport does not claim complete operator coverage
or make an uninstrumented consumer safe merely by reporting at completion.

## IPC frame

The independent gasUGKP protocol is now 8.0; FSH is 7.0. Operation payloads are unchanged. The request header is now 48 bytes: the
original 24 bytes followed by operation sequence (u64), actual OpenFOAM time
index (i64), and simulation time (binary64). Metadata travels on existing RPCs.
A tiny ordered setup kernel updates context outside graph capture, with no new
copy or synchronization; graph replay never freezes capture-time metadata. The response is explicitly encoded as 32 bytes:

- magic (u32), major/minor (u16 each)
- status (i32), errorBytes (u32)
- payloadBytes (u64), diagnosticBytes (u64)

The body order is legacy error text, diagnostic envelope, operation payload.
Source descriptors for the reported sites accompany structured records.
Warnings are decoded and dispatched before payload exposure, including successful
zero-payload replies. A structured fatal makes the reply unsuccessful even if
an outer status was zero. The helper sends the fatal envelope, exposes no result
payload, and leaves request processing. The client refuses subsequent numerical
requests after receiving a fatal.

The final release response is also diagnostic-bearing. `ReleaseDiagnostics`
retains its final snapshot after storage teardown, so late errors cannot vanish
when the backend handle is cleared. Direct CHT invokes its registered OpenFOAM
sink at the same completed boundaries.

## Evidence and limits

`test_diagnostic_ipc_transport.py` executes both real client receive paths using
socket pairs and both real server response paths for a terminal release snapshot.
It checks successful warnings, empty success replies, payload preservation, fatal
status propagation, and rejection of later numerical requests after a fatal.
`test_gpu_diagnostic_owner.py` uses a fake CUDA allocator solely to test host
ownership, independent instances, readback timing, count deltas, and sticky host
and device fatal snapshots. It is not a GPU execution test.

Actual nvcc compilation and OpenFOAM frontend compilation are tracked separately.
The available environment has no GPU runtime device; runtime graph ordering,
atomic contention, and device-consumer suppression require GPU execution tests.
