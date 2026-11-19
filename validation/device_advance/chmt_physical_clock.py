import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def load_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def simulate(binary, fixture, case, end, restart=None):
    case.mkdir()
    for directory in ('0', 'constant', 'system'):
        shutil.copytree(fixture/directory, case/directory)
    control = case/'system/controlDict'
    text, count = re.subn(r'\bendTime\s+[^;]+;', f'endTime {end};', control.read_text())
    assert count == 1
    if restart:
        text += f'\nrestartDirectory "{restart}";\n'
    control.write_text(text)
    with (case/'solver.log').open('x') as log:
        subprocess.run([binary, '-case', case], stdout=log, stderr=subprocess.STDOUT, check=True)
    output = case/'checkpointOutput'
    stamps = []
    for stamp in sorted(output.glob('accepted-*.csv')):
        row, = load_rows(stamp)
        step = int(row['accepted_steps'])
        assert (output/f'checkpoint-{step}').is_dir()
        gas = load_rows(output/f'gas-{step}.csv')
        solid = load_rows(output/f'material-{step}.csv')
        assert len(gas) == len(solid) == 8
        mass = math.fsum([float(r['mass']) for r in gas] +
                         [float(v) for r in solid for k,v in r.items() if k.startswith('mass_')])
        energy = math.fsum(float(r['total_energy']) for r in gas+solid)
        packets = load_rows(output/f'exchange-{step}.csv')
        for r in packets:
            assert int(r['consumer_mask']) & 3 == 3
            parts = ('conductive','advective','pressure_work','viscous_work','radiation','liquid_kinetic_advection')
            assert math.isclose(float(r['energy']), math.fsum(float(r[k]) for k in parts), rel_tol=4e-6, abs_tol=1e-12)
        stamps.append(dict(step=step, time=float(row['time']), mass=mass, energy=energy, packets=len(packets)))
    return stamps


def radiating_tail(binary, fixture, output):
    from cht_physical_clock import replace_entry

    period=float(re.search(r'\bcouplingInterval\s+([^;]+);',(fixture/'constant/scheme').read_text())[1])
    def advance(name,end,fields,checkpoints,restart=None):
        case=output/name
        case.mkdir()
        for directory in ('0','constant','system'):
            shutil.copytree(fixture/directory,case/directory)
        scheme=case/'constant/scheme'
        scheme.write_text(re.sub(r'(\bthermalCoupling\s*\{)',r'\1\n    radiationCouplingFrequency 3;',scheme.read_text()))
        control=case/'system/controlDict'
        for key,value in dict(endTime=end*period,fieldInterval=fields,checkpointInterval=checkpoints,
                              diagnosticInterval=1,logInterval=1,writeFinal='true').items():
            replace_entry(control,key,value)
        if restart:replace_entry(control,'restartDirectory','"'+str(restart)+'"')
        log=case/'solver.log'
        with log.open('x') as out:
            subprocess.run([binary,'-case',case],stdout=out,stderr=subprocess.STDOUT,check=True)
        text=log.read_text()
        assert 'radiationFrequency=3' in text
        return [(float(t)/period,float(r)/period) for t,r in re.findall(r'CHMT_REPRO step=\d+ time=(\S+) radiationTime=(\S+)',text)]
    first=advance('radiating',4.25,2,4)
    assert first==[(1,0),(2,2),(3,3),(4,4),(4.25,4.25)],first
    restart=output/'radiating/checkpointOutput/checkpoint-5'
    assert (restart/'domWarmStart.bin').stat().st_size>0
    second=advance('radiating-resumed',6.5,0,0,restart)
    assert second==[(5,4.25),(6,6),(6.5,6.5)],second
    return dict(first=first,resumed=second)


def main():
    parser = argparse.ArgumentParser(description='Resume an eight-cell closed CHMT case from a clipped window.')
    for name in ('binary', 'fixture', 'output'):
        parser.add_argument(name, type=lambda p:Path(p).resolve())
    parser.add_argument('--radiation-fixture',type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    period = float(re.search(r'\bcouplingInterval\s+([^;]+);',(args.fixture/'constant/scheme').read_text())[1])
    clipped = 1.25*period
    first = simulate(args.binary,args.fixture,args.output/'initial',clipped)
    assert [r['time'] for r in first] == [0,period,clipped]
    restart = args.output/'initial/checkpointOutput/checkpoint-2'
    resumed = simulate(args.binary,args.fixture,args.output/'restart',3*period,restart)
    assert [r['time'] for r in resumed] == [clipped,2*period,3*period]
    assert [r['step'] for r in resumed] == [2,3,4]
    all_stamps = first+resumed[1:]
    for field in ('mass','energy'):
        assert all(math.isclose(r[field],first[0][field],rel_tol=32*2**-23) for r in all_stamps)
    assert all(r['packets']>0 for r in all_stamps[1:])
    (args.output/'results.json').write_text(json.dumps(all_stamps,indent=2)+'\n')
    if args.radiation_fixture:
        events=radiating_tail(args.binary,args.radiation_fixture,args.output)
        (args.output/'radiation.json').write_text(json.dumps(events,indent=2)+'\n')
    print('CHMT accepted times', [r['time'] for r in all_stamps])


if __name__ == '__main__':
    main()
