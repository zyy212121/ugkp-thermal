import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from cht_physical_clock import number, replace_entry
from compare_completed_steps import digest, foam_results, mesh


DT = 2**-16
STEPS = 40
SWITCHES = [(8, 'clockTime'), (16, 'runTime'), (24, 'cpuTime'), (32, 'timeStep')]


def run(binary, fixture, case, switched):
    case.mkdir()
    for folder in ('0', 'constant', 'system'):
        shutil.copytree(fixture / folder, case / folder)
    ctl = case / 'system/controlDict'
    for key, value in dict(deltaT=DT, endTime=DT*STEPS, writeControl='timeStep',
                           writeInterval=STEPS, runTimeModifiable='true', adjustTimeStep='false',
                           logInterval=1, maxThermalCouplingEvents=0).items():
        replace_entry(ctl, key, value)
    for key, value in dict(solidContactCouplingInterval=2*DT,
                           radiationCouplingFrequency=1).items():
        replace_entry(case/'constant/scheme', key, value)
    if switched:
        files = ' '.join(f'({(step-1.5)*DT} "$FOAM_CASE/system/control.{mode}")'
                         for step, mode in SWITCHES)
        with ctl.open('a') as out:
            out.write('\nOptimisationSwitches { fileModificationSkew 0; }\n'
                      'functions { switchControl { type timeActivatedFileUpdate;\n'
                      'libs ("libutilityFunctionObjects.so"); fileToUpdate "$FOAM_CASE/system/controlDict";\n'
                      f'timeVsFile ({files});\n' + '} }\n')
        for _, mode in SWITCHES:
            target = case/'system'/f'control.{mode}'
            shutil.copyfile(ctl, target)
            replace_entry(target, 'writeControl', mode)
            interval = {'runTime': STEPS*DT, 'timeStep': STEPS}.get(mode, 100000)
            replace_entry(target, 'writeInterval', interval)

    log = case/'run.log'
    with log.open('x') as out:
        proc = subprocess.run([binary, '-case', case], cwd=case, stdout=out,
                              stderr=subprocess.STDOUT, timeout=120)
    if proc.returncode:
        raise RuntimeError(f'{log}\n{log.read_text()[-1000:]}')
    return inspect(case, switched)


def inspect(case, switched):
    log = case/'run.log'
    text = log.read_text()
    if text.count('Re-reading object controlDict') != (len(SWITCHES) if switched else 0):
        raise ValueError(f'Unexpected write-control changes in {log}')
    events = []
    device_thermal = {int(seq) for seq in re.findall(r'^THERMAL_TIMING sequence=(\d+)', text, re.M)}
    device_radiation = []
    previous_radiation = 0
    for line in text.splitlines():
        event = re.match(r'^THERMAL_EVENT accepted=(\d+) radiation=(\d+) time=(\S+) interval=(\S+)', line)
        if event:
            seq, radiation = int(event[1]), int(event[2])
            events.append((seq, radiation, float(event[3]), float(event[4])))
            if radiation > previous_radiation and seq in device_thermal:
                device_radiation.append(seq)
            previous_radiation = radiation
    if re.search(r'\bsite=34001\b', text):
        raise ValueError(f'Unconverged device DOM in {log}')
    expected = [(seq, seq, seq*2*DT, 2*DT) for seq in range(1, STEPS//2+1)]
    done = f'THERMAL_RUN_COMPLETE heatEvents={STEPS//2} radiationEvents={STEPS//2} '
    if events != expected or done not in text:
        raise ValueError(f'Unexpected committed events in {log}: {events}')
    mode = 'timeStep'
    changes = dict(SWITCHES) if switched else {}
    on_device = []
    for step in range(1, STEPS+1):
        if step % 2 == 0 and mode in ('runTime', 'timeStep'):
            on_device.append(step//2)
        mode = changes.get(step, mode)
    if device_radiation != on_device:
        raise ValueError(f'Unexpected DOM execution path in {log}: {device_radiation}')
    end = STEPS*DT
    terminal, = [p.parent for p in case.glob('*/thermalExchangeState') if float(p.parent.name) == end]
    stats, clock = foam_results(case, end, mesh(case/'constant/polyMesh'))
    if number(terminal/'uniform/time', 'index') != STEPS or any(
        number(terminal/'thermalExchangeState', key) != end
        for key in ('completedSimulationTimeS', 'completedRadiationSimulationTimeS')
    ) or number(terminal/'gpuPendingWallEnergyLedgers.dat', 'checkpointSimulationTimeS') != end:
        raise ValueError(f'Incomplete final checkpoint at {terminal}')
    warm = terminal/'domWarmStart.bin'
    if warm.stat().st_size == 0:
        raise ValueError(f'Empty DOM warm start at {warm}')
    return dict(events=events, device_radiation_events=device_radiation, fields=stats, clock=clock, warm_start_sha256=digest(warm))


def compare(fixed, switched):
    assert fixed['clock'] == switched['clock']
    assert fixed['fields'].keys() == switched['fields'].keys()
    for name, field in fixed['fields'].items():
        for stat, expected in field.items():
            np.testing.assert_allclose(switched['fields'][name][stat], expected,
                                       rtol=2e-5, atol=1e-12, err_msg=name)


def main():
    parser = argparse.ArgumentParser(description='Switch a running eight-cell CHT case between host and device DOM.')
    for key in ('binary', 'fixture', 'output'):
        parser.add_argument(key, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if not re.search(r'\bnCells:\s*8\b', (args.fixture/'constant/polyMesh/owner').read_text()):
        parser.error('expected an eight-cell fixture')
    if number(args.fixture/'constant/scheme', 'precision') != 32:
        parser.error('expected an FP32 fixture')
    args.output.mkdir(parents=True)
    fixed = run(args.binary, args.fixture, args.output/'device', False)
    switched = run(args.binary, args.fixture, args.output/'switch', True)
    compare(fixed, switched)
    (args.output/'results.json').write_text(json.dumps({'device': fixed, 'switch': switched}, indent=2)+'\n')


if __name__ == '__main__':
    main()
