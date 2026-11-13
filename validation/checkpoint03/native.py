"""Checkpoint 3 paired dual-precision coverage on fresh synthetic cases; execution is deferred.

This suite uses real native binaries. It reports quantization-scale tolerances,
not bitwise equivalence or production-case convergence. Shared thermodynamic
and wall nonlinear convergence are checked by the companion C++ fixtures.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, os, re, subprocess, time
import numpy as np
from acceptance_support import BranchCollector

ROOT = Path(os.environ.get("UGKP_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()


BITS = 32
PROBE = None
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
inlet = module("fp32_inlet", ROOT / "validation/pr41/native_injection.py")
restart = module("fp32_restart", ROOT / "validation/pr41/native_transient_restart.py")
ENV = dict(inlet.m.ENV)
EPS = float(np.finfo(np.float32).eps)
# These are declared before any run. Particle mass has one FP32 publication;
# the source remainder stays double. 32 eps leaves room for reduction/CSV IO.
MASS_RTOL = 32 * EPS
# Comparisons measure practical field continuity across restart, not strict
# round-trip identity. Geometry remains double; particle velocity is float.
RESTART_RTOL = 5e-4
POSITION_ATOL = 2e-7
BUDGET_RTOL = 256 * EPS
MOMENTUM_ATOL = 1e-8
NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
NONFINITE = re.compile(r"(?<![A-Za-z_])[-+]?(?:nan|inf|infinity)(?![A-Za-z_])", re.I)

def configure_precision(case,bits,select=True):
    # Fixed acceptance-case values, written as one explicit set. No fallback
    # dictionary, per-precision override, or legacy-value inference is used.
    scheme=case/"constant/scheme"
    controls=scheme.read_text() if scheme.exists() else "FoamFile { version 2.0; format ascii; class dictionary; object scheme; }\n"
    controls=re.sub(r'(?m)^precision\s+(?:32|64)\s*;[ \t]*\n?', '', controls)
    molecular_wall=case.name in ("gas-ode","gas-wave")
    wall_rtol='1e-4' if molecular_wall else '.05'
    wall_atol='9.5367431640625e-7'
    def wall(match):
        body=re.sub(r'\brelativeTolerance\s+[^;]+;', 'relativeTolerance '+wall_rtol+';',match[2])
        body=re.sub(r'\babsoluteTolerance\s+[^;]+;', 'absoluteTolerance '+wall_atol+';',body)
        return match[1]+body+match[3]
    if bits==32: controls=re.sub(r'(\bwallLayer\s*\{)([^{}]*)(\})',wall,controls)
    def solid(match):
        body=re.sub(r'\bnonlinearRelativeTolerance\s+[^;]+;', 'nonlinearRelativeTolerance '+'1e-5'+';',match[2])
        body=re.sub(r'\benergyRelativeTolerance\s+[^;]+;', 'energyRelativeTolerance '+'1e-4'+';',body)
        return match[1]+body+match[3]
    if bits==32: controls=re.sub(r'(\bsolid\s*\{\s*nonlinear\s*\{)([^{}]*)(\})',solid,controls)
    scheme.write_text(controls+("\nprecision "+str(bits)+";\n" if select else ""))

def execute(binary, case, report, label="run"):
    configure_precision(case,BITS)
    if binary.name != "CHMT" and (case/"constant/particleProperties").is_file():
        inlet.g.set_entry(case/"constant/particleProperties", "gpuResidentRandomSeed", 12345)
    input_hashes={str(p.relative_to(case)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for name in ("0","constant","system") for p in sorted((case/name).rglob("*")) if p.is_file()}
    log = case / (label + ".log")
    start = time.perf_counter()
    with log.open("w") as output:
        result = subprocess.run([str(binary), "-case", str(case)], env=ENV,
                                stdout=output, stderr=subprocess.STDOUT)
    record = dict(input_sha256=input_hashes,seed=1 if binary.name=="CHMT" else 12345,solver=binary.name, precision=BITS, backend_sha256=hashlib.sha256((ROOT/"bin/.precision"/str(BITS)/binary.name).read_bytes()).hexdigest(), case=str(case), log=str(log), exit=result.returncode,
                  seconds=time.perf_counter()-start,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    report["runs"].append(record)
    assert result.returncode == 0, record
    text = log.read_text(errors="replace")
    # Output-field checks below are authoritative; record existing diagnostics
    # without turning every warning or error-word into a new rejection policy.
    record["warning_lines"] = [line for line in text.splitlines()
                               if "warning" in line.lower()][:30]

def particle_inventory(case, solver, output):
    if solver == "CHMT":
        directory = case / "chmtOutput"
        index = max(int(p.stem.split("-")[1]) for p in directory.glob("particles-*.csv"))
        with (directory / f"particles-{index}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        xyz = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
        mass = math.fsum(float(row["mass"]) for row in rows)
        binary = PROBE
        residual = float(subprocess.check_output(
            [str(binary), str(directory/f"checkpoint-{index}")], text=True))
    else:
        frame = inlet.m.times(case)[-1]
        if solver.startswith("gas"):
            rows = inlet.g.parcel(frame/"gpuResidentStrictParticles.dat")
            xyz = np.array([row[:3] for row in rows])
            mass = math.fsum(row[9] for row in rows)
        else:
            chunks = [data for _, data in inlet.wire.iter_chunks(frame/"gpuResidentStrictParticles.dat")]
            xyz = np.concatenate([np.column_stack([data[k] for k in ("px", "py", "pz")])
                                  for data in chunks])
            mass = math.fsum(float(value) for data in chunks for value in data["pm"])
        values = (frame/"gpuResidentStrictSourceResidual.dat").read_text().split()
        residual = math.fsum(map(float, values[3::2]))
    assert np.isfinite(xyz).all() and math.isfinite(mass) and math.isfinite(residual)
    return xyz, mass, residual

def injection_check(case, solver, expected, output):
    xyz, mass, residual = particle_inventory(case, solver, output)
    error = abs(mass + residual - expected["expected_mass"]) / expected["expected_mass"]
    assert len(xyz) > 1000, len(xyz)
    assert error <= MASS_RTOL, (solver, mass, residual, expected, error, MASS_RTOL)
    assert np.max(np.abs(xyz[:, 0] - expected["x"])) < 2e-6
    normalized = xyz[:, 1:] / np.array([expected["y"], expected["z"]])
    assert normalized.min() >= -1e-10 and normalized.max() <= 1+1e-10
    means, variances = normalized.mean(axis=0), normalized.var(axis=0)
    assert np.max(np.abs(means-.5)) < .03, means
    assert np.max(np.abs(variances-1/12)) < .01, variances
    assert len(np.unique(normalized[:, 0])) > len(xyz)*.9
    return dict(solver=solver, count=len(xyz), mass=mass, residual=residual,
                expected_mass=expected["expected_mass"], relative_mass_error=error,
                mass_relative_tolerance=MASS_RTOL, mean=means.tolist(),
                variance=variances.tolist())

def finite_outputs(case, solver):
    fields = {}
    if solver == "CHMT":
        files = sorted((case/"chmtOutput").glob("*.csv"))
        for path in files:
            with path.open() as stream:
                for index, row in enumerate(csv.reader(stream)):
                    if index == 0:
                        continue
                    for text in row:
                        try:
                            value = float(text)
                        except ValueError:
                            continue
                        assert math.isfinite(value), (path, index, text)
        assert files, case
        fields["csv_files"] = len(files)
    else:
        frame = inlet.m.times(case)[-1]
        for path in frame.rglob("*"):
            if not path.is_file() or path.suffix == ".dat":
                continue
            text = path.read_text(errors="replace")
            if "internalField" not in text:
                continue
            text = text[text.index("internalField"):]
            assert not NONFINITE.search(text), path
            values = [float(s) for s in NUMBER.findall(text)]
            assert all(math.isfinite(v) for v in values), path
            fields[str(path.relative_to(frame))] = dict(min=min(values), max=max(values))
        assert fields, case
    return fields

def field_comparison(left, right, solver):
    errors = {}
    if solver == "CHMT":
        def gas(case):
            directory = case/"chmtOutput"
            file = max(directory.glob("gas-*.csv"), key=lambda p: int(p.stem.split("-")[1]))
            with file.open() as stream:
                rows = list(csv.DictReader(stream))
            return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        a, b = gas(left), gas(right)
    else:
        def gas(case):
            frame = inlet.m.times(case)[-1]
            result = {}
            for name in ("rho", "rhoU", "rhoE", "U", "p", "T", "k", "omega", "nut"):
                path = frame/name
                if not path.exists():
                    continue
                text = path.read_text()
                text = text[text.index("internalField"):]
                result[name] = np.array([float(s) for s in NUMBER.findall(text)])
            return result
        a, b = gas(left), gas(right)
    assert a.keys() == b.keys() and a
    for key in a:
        assert a[key].shape == b[key].shape, (solver, key)
        denominator = np.maximum(1., np.maximum(np.abs(a[key]), np.abs(b[key])))
        error = float(np.max(np.abs(a[key]-b[key])/denominator))
        assert error <= RESTART_RTOL, (solver, key, error)
        errors[key] = error
    return errors

def main():
    global BITS, PROBE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision",type=int,choices=(32,64),required=True)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--branch",required=True)
    args = parser.parse_args()
    BITS, PROBE = args.precision, args.probe.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", precision=BITS, runs=[], injection=[], restart=[], transport=[], coupling=[],
                  criteria=dict(mass_rtol=MASS_RTOL, restart_rtol=RESTART_RTOL,
                                position_atol=POSITION_ATOL, budget_rtol=BUDGET_RTOL,
                                momentum_atol=MOMENTUM_ATOL),
                  limits="Synthetic short-window runs, not production steady-state convergence or a speed benchmark.")
    def save():
        (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    collector=BranchCollector(report,output,save)
    for solver in ("gasUGKP", "gasUGKPSteady", "CHT", "CHMT"):
        if args.branch != solver+"-continuous": continue
        def continuous():
            binary = ROOT/"bin"/solver
            case = output/(solver+"-continuous")
            expected = inlet.prepare(case, solver)
            if solver != "gasUGKPSteady":
                inlet.g.set_entry(case/"system/controlDict", "endTime", 2*inlet.DT)
                expected["expected_mass"] *= 2
            execute(binary, case, report)
            report["injection"].append(injection_check(case, solver, expected, output))
            report["injection"][-1]["fields"] = finite_outputs(case, solver)
            save()
            return expected
        expected=collector.attempt(solver+"-continuous",continuous)
    for name, mass, diameter, steps, slope, velocity in (
            ("hot-constant", 1e-6, .001, 1, 0, 0),
            ("hot-nonlinear", 1e-6, .001, 1, -.22, 0),
            ("closed-drag-heat", .001, .001, 100, 0, 2)):
        if args.branch != name: continue
        def coupling():
            case = output/name
            inlet.g.one_cell(case, mass, diameter, steps, slope, velocity)
            # The historical fixture predates the diameter-schema migration.
            # Its dMin == dMax already specifies the same monodisperse input.
            properties = case/"constant/particleProperties"
            properties.write_text(re.sub(r"\bdSigma\s+[^;]+;", "", properties.read_text()))
            inlet.m.run(["blockMesh", "-case", case], case/"mesh.log")
            execute(ROOT/"bin/gasUGKP", case, report)
            final = inlet.g.dirs(case)[-1]
            before = np.array(inlet.g.budget(case/"0", mass, slope, True, velocity))
            after = np.array(inlet.g.budget(final, mass, slope))
            error = np.abs(after-before)
            allowed = BUDGET_RTOL*np.abs(before)+np.array([1e-10, MOMENTUM_ATOL, 1e-5])
            assert np.isfinite(after).all() and np.all(error <= allowed), (name, before, after, error, allowed)
            particles = inlet.g.parcel(final/"gpuResidentStrictParticles.dat")
            assert particles and 100 < particles[0][6] < 10000
            scalar = lambda field: inlet.g.fields.internal(final/field, 1)[0][0]
            gas_temperature = scalar("T")
            # The FP32 temperature cache may round a positive sub-ULP heat
            # gain back to its initial value. Check the FP64 energy directly.
            temperature_cache_tolerance = 8*EPS*max(1000., abs(gas_temperature))
            initial_gas_energy_density = 1000*1000 + .5*slope*1e6 - 8.31446261815324/.028*1000
            gas_energy_gain = scalar("rhoE") - initial_gas_energy_density
            assert gas_energy_gain > 0, (name, gas_energy_gain)
            assert 1000-temperature_cache_tolerance <= gas_temperature <= 10000+temperature_cache_tolerance, (name, gas_temperature, temperature_cache_tolerance)
            species_error = abs(scalar("rhoY_A")+scalar("rhoY_B")-scalar("rho"))
            assert species_error <= 128*EPS*max(1., abs(scalar("rho")))
            report["coupling"].append(dict(fixture=name, initial_budget=before.tolist(),
                final_budget=after.tolist(), absolute_mass_momentum_energy_error=error.tolist(),
                allowed_error=allowed.tolist(), particle_temperature=particles[0][6],
                gas_temperature=gas_temperature, gas_energy_gain=gas_energy_gain,
                temperature_cache_tolerance=temperature_cache_tolerance,
                species_closure_error=species_error,
                fields=finite_outputs(case, "gasUGKP")))
            save()
        collector.attempt(name,coupling)
    meta = json.loads((ROOT/"validation/molecular_transport/fixtures.json").read_text())
    for name in ("gas-wave", "gas-ode", "cht-contact", "chmt-diffusion"):
        if args.branch != name: continue
        def transport():
            case = output/name
            inlet.m.prepare(ROOT/"validation/molecular_transport/fixtures"/name, case, meta[name])
            solver = meta[name]["solver"]
            execute(ROOT/"bin"/solver, case, report)
            report["transport"].append(dict(fixture=name, fields=finite_outputs(case, solver)))
            save()
        collector.attempt(name,transport)
    assert len(report["branches"]) == 1, "Unknown or nonunique frozen branch"
    exit_code=collector.finish()
    print(json.dumps(report,indent=2))
    if exit_code: raise SystemExit(exit_code)
if __name__ == "__main__":
    main()
