# Reassemble and reproduce the evidence

## 1. Verify and unpack the mathematical reference

From the repository root, run:

```sh
cd docs/diagnostics/pr22-math-qualification-20261021
sha256sum -c PARTS-SHA256SUMS
evidence_dir="$PWD"
work_dir="$(mktemp -d)"
cat pr22-math-reference-20261021T0941Z.tar.gz.part-00 \
    pr22-math-reference-20261021T0941Z.tar.gz.part-01 \
    pr22-math-reference-20261021T0941Z.tar.gz.part-02 \
    > "$work_dir/pr22-math-reference-20261021T0941Z.tar.gz"
(cd "$work_dir" && sha256sum -c "$evidence_dir/ARCHIVE-SHA256SUMS")
tar -xzf "$work_dir/pr22-math-reference-20261021T0941Z.tar.gz" -C "$work_dir"
cd "$work_dir/pr22-math-reference-20261021T0941Z"
python3 verify.py
```

The combined archive is **56,523,685 bytes**; SHA256 is
`505cc2aa4ebca0e637222d3311d6be3880c3b3e7dcf071400319520f437fb2d1`.
It contains 748 manifest-verified files. Keep the saved evidence unchanged;
new runs write to its `reproduced/` directory.

Linux x86-64, a C++17 `g++` compiler and Python 3 are required. From the extracted
package, the supported scalar commands are:

```sh
python3 audit.py                 # inspect all shipped raw terminal root records
bash build.sh
bash run-focused.sh             # manufactured controls and old/repaired root replay
bash run-coupled.sh              # 20 steps / 4 microseconds
bash run-coupled.sh 1000         # full 200-microsecond qualification
bash run-cost.sh                 # run ALONE: four sequential processes, no competing work
```

No CPU team/cooperative test is involved. The cost replay excludes diagnostic
capture overhead and uses native HEAD/final discretizations; its result is
relative measured cost, not an equal-discretization speedup. Read the extracted
`RESULTS.md`, `REPRODUCE.md`, `OMITTED-BULK-RAW.json` and `RELOCATION-VERIFICATION.json`
for scope, raw-data coverage and native binary-record ABI details. The complete
large face/sample audit and per-stage history are regenerable but not duplicated;
their hashes and independent audit summaries are included.

## 2. Verify and unpack the shared/CUDA-port evidence

From this evidence directory:

```sh
(cd shared-port && sha256sum -c SHA256SUMS)
port_dir="$(mktemp -d)"
tar -xzf shared-port/shared-math-port-evidence-20261021.tar.gz -C "$port_dir"
python3 - "$port_dir/shared-math-port-20261021" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "SHA256SUMS.json").read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print("Verified evidence files:", len(manifest))
PY
```

This archive is **2,826,484 bytes**; SHA256 is
`5f561028ffedd8de743e1257d9b203f20e003cb688e7ebaa4f592c3e338efa63`.
It records 98 verified evidence files for the reviewed shared-port source.
`PORT-REPORT.json` and the source hashes distinguish this evidence from the
separate mathematical-reference trajectory. The extracted evidence scripts
retain the original capture locations; the checkout's numeric fixtures and
following driver commands do not require those locations.

## 3. Run current checkout checks and user CUDA validation

From the repository root:

```sh
python3 validation/tests/gas_wall/test_affine_directional_scalar.py
python3 validation/tests/gas_wall/test_coherent_shared_numerics.py
python3 validation/tests/gas_wall/test_coherent_wall_storage.py Storage.test_worker_scratch_and_wall_history_are_distinct
python3 validation/cuda/wall_coupling/reproduce.py --cpu-only --output /tmp/pr22-scalar
```

For the user's GPU validation, choose the installed CUDA path and GPU architecture:

```sh
python3 validation/cuda/wall_coupling/reproduce.py \
  --cuda /usr/local/cuda --arch sm_89 --sanitize --output /tmp/pr22-cuda
```

The profile and coherent directional drivers use the current shared scalar
oracle and native `CudaWallTeam` device execution. Legacy BVP checks remain
separately identified. The runner saves commands, outputs, source/binary hashes
and execution status. **nvcc/GPU/CUDA-sanitizer results are UNRUN in this delivery**;
no CPU parallel testing is offered as GPU acceleration evidence.
