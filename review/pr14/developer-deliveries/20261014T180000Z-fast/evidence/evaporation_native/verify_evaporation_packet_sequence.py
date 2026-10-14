#!/usr/bin/env python3
"""Read-only verification of archived D302 timing and packet arithmetic.

No solver is run or changed. Fraction tracks exact sums of observed binary64
increments; downward conversion reproduces boundedWindowDuration. The receipt
reports observations and mismatches rather than changing solver acceptance.
"""
import argparse
import collections
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/workspace/scratch/98b6339ec81b')
DEFAULT_REPO = ROOT / 'pr14_fast_fix_20261014'
DEFAULT_EVIDENCE = ROOT / 'pr14_review_readonly_20261014T145300Z/review/pr14/reviewer-deliveries/20261014T120103Z-fast'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floored_binary64(value):
    rounded = float(value)
    return math.nextafter(rounded, 0.0) if Fraction(rounded) > value else rounded


def delta_ulps(actual, expected):
    return (actual - expected) / math.ulp(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo, evidence = args.repo, args.evidence
    log = evidence / 'logs/EVAPORATION-PROGRESS-D302.log'
    phase = []
    windows = {}
    packets = {}
    for number, line in enumerate(log.open(), 1):
        fields = line.strip().split(',')
        if fields[0] == 'R3PHASE':
            row = dict(field.split('=', 1) for field in fields[1:])
            if row['window'] == '2':
                row['line'] = number
                phase.append(row)
        elif len(fields) == 8 and fields[0] == 'F003' and fields[1] == '2':
            retry, micro, kind, index, name, value = fields[2:]
            if kind == 'begin' and micro == '1':
                windows.setdefault(int(retry), {})[name] = {'line': number, 'hex': value, 'value': float.fromhex(value)}
            if kind == 'packet':
                packets.setdefault((int(retry), int(micro), int(index)), {})[name] = {'line': number, 'hex': value, 'value': float.fromhex(value)}

    final_retry = max(windows)
    final_window = windows[final_retry]
    begin = final_window['window_begin']['value']
    end = final_window['window_end']['value']
    span = end - begin
    steps = [row for row in phase if row['face'] == '12']
    elapsed = Fraction(0)
    recurrence_errors = []
    half_exponent = 30
    recurrence_max_ulps = 0.0
    first_dt = float.fromhex(steps[0]['dt'])
    for row in steps:
        expected = floored_binary64(Fraction(span) - elapsed) / (2 ** half_exponent)
        actual = float.fromhex(row['dt'])
        ulps = delta_ulps(actual, expected)
        recurrence_max_ulps = max(recurrence_max_ulps, abs(ulps))
        if actual != expected:
            recurrence_errors.append({'line': row['line'], 'actual_hex': actual.hex(), 'expected_hex': expected.hex(), 'delta_ulps': ulps})
        elapsed += Fraction(actual)

    # D302 face 12..15 are material persistent faces / film indices 0..3.
    # Gas history packets have gas persistent faces 16..19; each film has a
    # GasFilm packet followed by a zero PoreGas packet. Pair by the ordered
    # four-face surface, checked explicitly in the recorded packet identities.
    by_film = {film: packets[(final_retry, 1, 2 * film)] for film in range(4)}
    gas_dt = final_window['micro_dt']['value']
    packet_errors = []
    packet_max_ulps = 0.0
    film_states = {}
    for row in phase:
        film = int(row['film'])
        packet = by_film[film]
        actual = float.fromhex(row['forcingMass'])
        expected = -packet['mass']['value'] * (float.fromhex(row['dt']) / gas_dt)
        ulps = delta_ulps(actual, expected)
        packet_max_ulps = max(packet_max_ulps, abs(ulps))
        if actual != expected:
            packet_errors.append({'line': row['line'], 'actual_hex': actual.hex(), 'expected_hex': expected.hex(), 'delta_ulps': ulps})
        state = film_states.setdefault(film, {'phase_rows': 0, 'old_mass': set(), 'new_mass': set(), 'old_enthalpy': set(), 'new_enthalpy': set(), 'old_mean_x': set(), 'new_mean_x': set()})
        state['phase_rows'] += 1
        for field, label in [('oldFilmM', 'old_mass'), ('newFilmM', 'new_mass'), ('oldFilmH', 'old_enthalpy'), ('newFilmH', 'new_enthalpy'), ('oldMeanX', 'old_mean_x'), ('newMeanX', 'new_mean_x')]:
            state[label].add(float.fromhex(row[field]))
    for film, state in film_states.items():
        for label, value in list(state.items()):
            if isinstance(value, set):
                state[label] = sorted(value)
        packet = by_film[film]
        row = next(row for row in phase if int(row['film']) == film)
        mass = float.fromhex(row['oldFilmM'])
        rate = -packet['mass']['value'] / gas_dt
        state['gas_packet'] = packet
        state['first_phase_line'] = row['line']
        state['mass_rate_kg_per_s'] = rate
        state['mass_depletion_scale_s'] = mass / abs(rate)
        state['first_withdrawal_fraction_of_film_mass_ulp'] = abs(float.fromhex(row['forcingMass'])) / math.ulp(mass)
        state['first_solid_film_heat_rate_W'] = float.fromhex(row['heat']) / float.fromhex(row['dt'])
        state['first_solid_film_x_momentum_rate_N'] = float.fromhex(row['mx']) / float.fromhex(row['dt'])

    manifest = evidence / 'artifacts/evaporation-blocked-cause-and-progress.json'
    meta = json.loads(manifest.read_text())
    old = repo / 'review/pr14/reviewer-deliveries/20261014T072053Z-fast/artifacts/extra/wet-evaporation-S209-D204'
    equivalence = []
    for rel in ['checkpoint-1/state.chmt', 'exchange-1.csv', 'material-1.csv']:
        reference = next(item for item in meta['same_completed_observer_vs_formal_files'] if item['path'] == rel)
        actual = sha256(old / rel)
        equivalence.append({'available_prior_path': str(old / rel), 'actual_sha256': actual, 'fresh_formal_manifest_sha256': reference['formal_sha256'], 'fresh_observer_manifest_sha256': reference['observer_sha256'], 'matches_both_manifest_hashes': actual == reference['formal_sha256'] == reference['observer_sha256']})
    source_paths = [
        repo / 'applications/CHMT/coupling/IntervalContracts.H',
        repo / 'applications/CHMT/coupling/WindowTime.H',
        repo / 'applications/CHMT/coupling/ExchangeLedger.H',
        repo / 'applications/CHMT/materials/CpuMaterialDriver.C',
        evidence / 'inputs/CpuMaterialDriver-phase-observer-D302.C',
        repo / 'review/pr14/reviewer-deliveries/20261014T072053Z-fast/inputs/native/extra/wet-evaporation-S209/constant/chmtProperties',
        log, manifest,
    ]
    receipt = {
        'kind': 'READ_ONLY_ARCHIVED_EVIDENCE_ARITHMETIC',
        'script_sha256': sha256(Path(__file__).resolve()),
        'production_source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '9d5be6ef'], text=True).strip(),
        'source_and_evidence_sha256': {str(path): sha256(path) for path in source_paths},
        'window_2_retry_boundaries': windows,
        'final_retry_span': {'hex': span.hex(), 'value': span, 'gas_actual_dt': gas_dt},
        'thirty_halving_recurrence': {
            'expression': 'dt[i] = floor_binary64(actual_retry12_span - exact_sum(previous_observed_dt)) / 2**30',
            'step_count': len(steps), 'first_line': steps[0]['line'], 'last_line': steps[-1]['line'],
            'first_dt_hex': first_dt.hex(), 'first_dt': first_dt, 'last_dt': float.fromhex(steps[-1]['dt']),
            'mismatch_count': len(recurrence_errors), 'max_absolute_delta_ulps': recurrence_max_ulps,
            'mismatches_first_20': recurrence_errors[:20],
            'observed_paid_local_duration': float(elapsed),
            'observed_fraction_of_final_retry_span': float(elapsed) / span,
            'last_logged_absolute_begin_time': float.fromhex(steps[-1]['time']),
            'nominal_initial_window_divided_by_2_to_42': 0.0005 / (2 ** 42),
            'first_dt_difference_from_nominal_ulps': delta_ulps(first_dt, 0.0005 / (2 ** 42)),
        },
        'packet_mass_overlap_scaling': {
            'expression': 'forcingMass = -gasPacket.mass * (local_dt / gas_actual_dt)',
            'phase_row_count': len(phase), 'mismatch_count': len(packet_errors),
            'max_absolute_delta_ulps': packet_max_ulps, 'mismatches_first_20': packet_errors[:20],
            'mapping': 'Film 0..3 corresponds to material face 12..15 and GasFilm packet indices 0,2,4,6 / gas face 16..19.',
        },
        'film_states': film_states,
        'available_prior_bytes_equal_fresh_manifest': equivalence,
        'scope_and_limits': [
            'No native rerun; no production files or README files changed.',
            'Exact recurrence proves observed arithmetic pattern, not the unlogged retry branch or its error.',
            'The phase observer emits after local trial processing and before local state assignment; it does not establish macro acceptance.',
            'The available prior checkpoint and ledger bytes match fresh-run hashes recorded in the evidence manifest; the fresh files themselves are absent from this checkout.',
            'No claim that zero stored mass/enthalpy change alone identifies the limiter.',
        ],
    }
    result = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
