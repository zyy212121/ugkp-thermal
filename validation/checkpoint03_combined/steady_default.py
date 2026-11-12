#!/usr/bin/env python3
"""Two fixed future native nodes. Preparing this source does not run the case."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, os, re, subprocess, traceback
import numpy as np

ROOT = Path(os.environ.get('UGKP_SOURCE_ROOT', Path(__file__).resolve().parents[2])).resolve()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', type=int, choices=(32, 64), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', precision=args.precision, runs=[],
                  default_relative_tolerance=1e-4 if args.precision == 32 else 1e-8,
                  fixed_iterations=20000, residual_interval=1000, minimum_iterations=1000,
                  numerical_execution=True, historical_results_reused=False)
    try:
        native = module('combined_native_steady', ROOT/'validation/checkpoint03_combined/native.py')
        source = module('combined_reservoir_input', ROOT/'validation/steady_cpu/of_cases.py')
        fields = module('combined_steady_fields', ROOT/'validation/steady_cpu/export_of.py')
        cfg = ROOT/'validation/steady_cpu/cases/reservoir.cfg'
        case = output/'reservoir-default'
        source.materialize(cfg, case, particles=False)
        control = case/'system/controlDict'
        scheme_path=case/'constant/scheme'
        original=scheme_path.read_text()
        # Fixed reservoir test: only its explicit steadyGas block is edited.
        match=re.search(r'(steadyGas\s*\{)([^{}]*)(\})',original)
        body=re.sub(r'\brelativeTolerance\s+[^;]+;', '', match[2])
        body=re.sub(r'\bminimumIterations\s+[^;]+;', 'minimumIterations 1000;', body)
        scheme_path.write_text(original[:match.start(2)]+body+original[match.end(2):])
        changed=scheme_path.read_text()
        report['control_changes']=dict(removed='steadyGas.relativeTolerance',
            replaced=['steadyGas.minimumIterations',1000],
            original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            resulting_sha256=sha(scheme_path))
        report['input_sources'] = {str(path.relative_to(ROOT)): sha(path) for path in
                                  (cfg, ROOT/'validation/steady_cpu/of_cases.py',
                                   ROOT/'validation/steady_cpu/gas_reference.cpp')}
        declaration = ROOT/('precision/fp32/applications/gasUGKPSteady/steadyGasLoop.H'
                            if args.precision == 32 else 'applications/gasUGKPSteady/steadyGasLoop.H')
        report['default_selection_evidence'] = dict(
            source=str(declaration), sha256=sha(declaration),
            declaration=[line for line in declaration.read_text().splitlines()
                         if 'steadyRelativeTolerance =' in line],
            input_has_no_relative_tolerance=not re.search(r'\brelativeTolerance\s', re.search(r'steadyGas\s*\{([^{}]*)\}',changed)[1]),
            note='The production binary is selected by constant/scheme precision. No production probe or log modification. The actual converged branch and final reported residual must also satisfy the selected default.')
        native.inlet.m.run(['blockMesh', '-case', case], case/'mesh.log')
        native.BITS = args.precision
        native.execute(ROOT/'bin/gasUGKPSteady', case, report)
        log = (case/'run.log').read_text(errors='replace')
        report['warning_lines'] = [line for line in log.splitlines() if 'warning' in line.lower()]
        report['precision_lines'] = [line for line in log.splitlines() if 'precision' in line.lower()]
        initial_match = re.search(r'Initial steady spatial Linf residual\s+\d*\s*\(([^)]*)\)', log)
        assert initial_match, 'Missing actual initial residual'
        initial = np.array([float(x) for x in initial_match[1].split()])
        checks = []
        pattern = r'Steady iteration\s+(\d+)\s+spatial Linf\s+\d*\s*\(([^)]*)\)\s+max relative\s+(\S+)'
        for iteration, values, relative in re.findall(pattern, log):
            checks.append(dict(iteration=int(iteration), residual=[float(x) for x in values.split()],
                               max_relative=float(relative)))
        marker = re.search(r'Steady spatial residual converged after\s+(\d+)\s+iterations\.', log)
        report.update(initial_residual=initial.tolist(), residual_checks=checks,
                      converged_marker=marker.group(0) if marker else None,
                      criteria_reached=bool(re.search(r'Steady driver finished:\s*criteriaReached=1\b', log)))
        assert marker and report['criteria_reached'], 'Budget exit is not convergence'
        assert checks and 1000 <= int(marker[1]) <= 20000
        assert all(math.isfinite(row['max_relative']) and
                   all(math.isfinite(value) for value in row['residual']) for row in checks)
        final = checks[-1]
        assert final['iteration'] == int(marker[1])
        final_values = np.array(final['residual'])
        assert np.isfinite(initial).all() and np.isfinite(final_values).all()
        assert np.max(np.abs(initial)) > 0, 'A zero-residual start cannot demonstrate decay'
        assert np.max(np.abs(final_values)) <= .1*np.max(np.abs(initial)), 'Initial nonzero residual did not decay'
        # OpenFOAM prints a rounded decimal. This allowance only parses the
        # printed evidence; the real production converged branch is mandatory.
        report['printed_ratio_allowance'] = 1.00001
        assert math.isfinite(final['max_relative'])
        assert final['max_relative'] <= 1.00001*report['default_relative_tolerance']
        manifest = json.loads((case/'reference-input.json').read_text())
        count = int(manifest['input']['cells'])
        frame = max((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?', p.name)), key=lambda p: float(p.name))
        def values(directory, name):
            return np.array(fields.internal(directory/name, count), dtype=float)
        measured = {name: values(frame, name) for name in ('rho', 'U', 'p', 'T', 'rhoE')}
        assert all(np.isfinite(x).all() for x in measured.values())
        assert np.min(measured['rho']) > 0 and np.min(measured['T']) > 0
        decay = {}
        for name, component, target in (('rho', 0, 1.0), ('U', 0, 20.0)):
            before = values(case/'0', name)[:, component] - target
            after = measured[name][:, component] - target
            rms0 = float(np.sqrt(np.mean(before*before)))
            rms1 = float(np.sqrt(np.mean(after*after)))
            decay[name] = dict(initial_rms=rms0, final_rms=rms1, ratio=rms1/rms0)
            assert rms0 > 0 and rms1 <= .1*rms0, (name, decay[name])
        report['perturbation_decay'] = decay
        volume = manifest['input']['length']/count
        report['observable_inventory'] = dict(
            cell_volume_m3=volume, initial_mass_kg=float(values(case/'0', 'rho').sum()*volume),
            final_mass_kg=float(measured['rho'].sum()*volume),
            initial_energy_J=float(values(case/'0', 'rhoE').sum()*volume),
            final_energy_J=float(measured['rhoE'].sum()*volume))
        proxy = measured['rho'][:, 0]*measured['U'][:, 0]
        report['cell_advective_mass_flux_proxy_kg_m2_s'] = dict(min=float(proxy.min()), max=float(proxy.max()), spread=float(proxy.max()-proxy.min()))
        report['conservation_scope'] = 'Open fixed-reservoir boundaries. Integrated boundary mass/enthalpy fluxes are unavailable. Inventory changes and cell advective proxies are observations, not whole-domain closure or measured numerical face fluxes. Real spatial residual convergence is checked above; separate closed fixtures test conservation.'
        report['final_fields'] = {name: dict(min=float(data.min()), max=float(data.max())) for name, data in measured.items()}
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
