#!/usr/bin/env python3
"""Read existing PR14 evidence; write descriptive metrics, never change acceptance rules."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014')
    parser.add_argument('--output', default=str(Path(__file__).with_name('metrics.json')))
    args = parser.parse_args()
    repo = Path(args.repo)
    evidence_commit = '0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668'
    source = '9d5be6eff69dbced2fc959d422d3ba419a1bc39e'
    d = Path('review/pr14/reviewer-deliveries/20261014T120103Z-fast')
    r2 = Path('review/pr14/reviewer-deliveries/20261014T072053Z-fast')
    original_input = Path('review/pr14/reviewer-deliveries/20261013T182722Z-fast/inputs/native/chmt/thermal-full/constant/gasModelProperties')
    refs = {}

    def read(path):
        content = (repo / path).read_bytes()
        key = str(path)
        blob = subprocess.check_output(['git', 'rev-parse', evidence_commit + ':' + key], cwd=repo, text=True).strip()
        fixed = subprocess.check_output(['git', 'show', evidence_commit + ':' + key], cwd=repo)
        refs[key] = {'evidence_commit': evidence_commit, 'git_blob': blob, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content), 'matches_fixed_commit': content == fixed}
        return content

    logs = {name: read(d / f'logs/THERMAL-SEED-thermal-{name}.log').decode().splitlines() for name in ('full', 'restart')}
    parsed = {}
    raw_counts = {}
    for name, lines in logs.items():
        rows = {}
        count = 0
        for line in lines:
            if line.startswith('R3SEED_CELL,'):
                columns = line.split(',')
                if columns[2] == '2':
                    count += 1
                    rows[tuple(columns[1:5])] = [float.fromhex(value) for value in columns[5:]]
        parsed[name] = rows
        raw_counts[name] = {'window2_raw_cell_rows': count, 'window2_unique_cell_keys': len(rows)}
    common = sorted(set(parsed['full']) & set(parsed['restart']))
    post = [key for key in common if key[0] == 'post']
    names = ['temperature_K', 'pressure_Pa', 'mass_kg', 'total_energy_J', 'momentum_x_kg_m_s', 'momentum_y_kg_m_s', 'momentum_z_kg_m_s', 'volume_m3']
    maxima = {field: max(abs(parsed['full'][key][index] - parsed['restart'][key][index]) for key in post) for index, field in enumerate(names)}
    first = ('post', '2', '1', '0')
    first_values = {run: dict(zip(names, parsed[run][first])) for run in parsed}
    first_delta = {field: first_values['full'][field] - first_values['restart'][field] for field in names}
    dt = {run: [line for line in lines if line.startswith('R3DT,2,')] for run, lines in logs.items()}

    cpu_reconstruction = {}
    for run, fields in first_values.items():
        T = fields['temperature_K']
        mass, total, momentum, volume = (fields[key] for key in ('mass_kg', 'total_energy_J', 'momentum_x_kg_m_s', 'volume_m3'))
        density, velocity = mass / volume, momentum / mass
        internal = total / volume - 0.5 * density * velocity * velocity
        R = 8.31446261815324 / 0.028
        evaluated = density * ((1040 * T) - (R * T))
        residual = evaluated - internal
        capacity = density * (1040 - R)
        cpu_reconstruction[run] = {'density': density, 'target_internal_energy_density': internal, 'recomputed_energy_density': evaluated, 'residual': residual, 'correction_K': residual / capacity, 'original_default_energy_tolerance': 1e-10 + 1e-10 * abs(internal), 'original_default_temperature_tolerance_K': 1e-9 + 1e-10 * abs(T)}

    wet = json.loads(read(d / 'artifacts/wet-full-restart-first-difference.json'))
    thermal = json.loads(read(d / 'artifacts/thermal-first-currentCo-seed-difference.json'))
    old = json.loads(read(r2 / 'artifacts/chmt/old802-vs-formal-restart.json'))
    original = json.loads(read(r2 / 'artifacts/chmt/formal-restart-terminal-comparison.json'))
    preservation = {run: json.loads(read(d / f'artifacts/thermal-seed-output-thermal-{run}.json'))['all_files_byte_equal'] for run in ('full', 'restart')}
    read(original_input)
    read(d / 'metadata/final-evidence-scope-review.md')
    read(d / 'REPORT.md')
    source_files = ['common/operators/computeSstFaceFluxKernel.cuh', 'common/gasTransport/MixtureThermo.H', 'common/gasTransport/GasModel.H', 'applications/CHMT/verification/coupled_runtime/check_case.py', 'applications/CHMT/gpu/SharedGasDeviceStorage.cuh']
    lineage = {}
    for revision in ['d6110c02745a8ad726208a426c4bb3a28233c1fd', '802a11e7dfe986488a8bf8305e86424dcc94c9e1', '4b5c480b572f234ddecf5e89164fe59b0d2e9735', source]:
        lineage[revision] = {path: subprocess.check_output(['git', 'rev-parse', revision + ':' + path], cwd=repo, text=True).strip() for path in source_files}
    for path in source_files:
        read(Path(path))
    wet_differences = []
    for item in wet['all_terminal_semantic_differences']:
        row = dict(item)
        row['relative_to_full_magnitude'] = abs(item['difference']) / abs(item['full']) if item['full'] else None
        wet_differences.append(row)
    result = {
        'scope': 'Read-only source/evidence review and CPU offline arithmetic; no new GPU run, no production/test/threshold edits, no acceptance change.',
        'source_commit': source, 'source_tree': '5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066', 'evidence_commit': evidence_commit,
        'recommendation': 'Keep mature FP64 primitive recovery and warm start unchanged; retain strict-bitparity failure as open qualification evidence, not a demonstrated regression of the latest fixes.',
        'raw_log_coverage': raw_counts, 'common_window2_post_rows': len(post), 'window2_post_max_absolute_deltas': maxima,
        'first_post_cell0': first_values, 'first_post_cell0_full_minus_restart': first_delta,
        'window2_dt_row_counts': {run: len(rows) for run, rows in dt.items()}, 'window2_dt_rows_identical': dt['full'] == dt['restart'],
        'temperature_delta_fraction_of_original_default_tolerance': abs(first_delta['temperature_K']) / cpu_reconstruction['full']['original_default_temperature_tolerance_K'],
        'offline_cpu_reconstruction': {'assumptions': 'Original thermal mixtureFrozen input: species S0 mass fraction 1, cp=1040, molar mass=.028, no diffusion; density reconstructed from captured mass/volume. Ordinary host binary64 operation ordering. This is not a GPU branch/iteration witness.', 'values': cpu_reconstruction},
        'thermal_observer_preserves_each_original_run_bytes': preservation,
        'thermal_raw_category_counts': {name: {k: v for k, v in info.items() if k in ('full_count', 'restart_count', 'different_row_count')} for name, info in thermal['actual_window2_categories'].items()},
        'wet_strict_bitparity': wet['strict_bitparity'], 'wet_semantic_difference_count': wet['semantic_difference_count'],
        'wet_first_exchange_difference': wet['first_exchange_difference'], 'wet_terminal_semantic_differences': wet_differences,
        'prior_802_vs_formal_restart': {'all_files_byte_equal': old['all_files_byte_equal'], 'file_count': len(old['files'])},
        'prior_formal_full_vs_restart_strict': original['all_files_byte_equal'],
        'source_blob_lineage': lineage, 'fixed_references': refs,
        'limitations': ['Raw D303 records do not capture caloric iteration count, energy residual/correction, or exact return branch.', 'D303 thermal seed observation is not an exclusive-cause witness for the separate wet case.', 'Pre-PR d6110 source blobs differ; old802 evidence does not establish full pre-PR baseline runtime equivalence.', 'Existing numerical continuation tolerance and strict byte equality are distinct criteria. Neither this script nor this review changes them.', 'No fresh GPU work is necessary merely to keep unchanged numerical code. If exclusive seed causality is required, collect one read-only first-affected native recovery witness using identical conserved inputs and actual controls, for warm/fresh seeds, including iteration/exit/residual and subsequent original output preservation.']
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': args.output, 'post_rows': len(post), 'dt_rows_equal': result['window2_dt_rows_identical'], 'max_deltas': maxima, 'wet_strict_bitparity': result['wet_strict_bitparity']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
