import argparse
import json
from pathlib import Path
import shutil
import subprocess

from cht_physical_clock import replace_entry


def entry(path, key):
    return subprocess.check_output(['foamDictionary', path, '-entry', key, '-value'], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description='Check CHT turbulence input precedence on an eight-cell uniform SST fixture.')
    for arg in ('binary', 'fixture', 'output'):
        parser.add_argument(arg, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    assert '\n27\n' in (args.fixture/'constant/polyMesh/points').read_text()
    assert not (args.fixture/'constant/momentumTransport').exists()
    turbulence = entry(args.fixture/'constant/fluidProperties', 'turbulence')
    assert turbulence.startswith('{') and turbulence.endswith('}') and 'kOmegaSST' in turbulence
    header = 'FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }\n'
    args.output.mkdir(parents=True)
    end = 2**-16
    fields, evidence = {}, {}
    for name in ('inline', 'explicit-sst', 'explicit-laminar'):
        case = args.output/name
        case.mkdir()
        for sub in ('0', 'constant', 'system'):
            shutil.copytree(args.fixture/sub, case/sub)
        fluid = case/'constant/fluidProperties'
        model_file = case/'constant/momentumTransport'
        if name == 'explicit-sst':
            model_file.write_text(header+turbulence[1:-1]+'\n')
            subprocess.run(['foamDictionary', fluid, '-entry', 'turbulence', '-set', '{ simulationType laminar; }'],
                           check=True, stdout=subprocess.DEVNULL)
        elif name == 'explicit-laminar':
            model_file.write_text(header+'simulationType laminar;\n')
        for key,value in dict(deltaT=end/16, endTime=end, adjustTimeStep='no', writeInterval=-1).items():
            replace_entry(case/'system/controlDict', key, value)
        for key,value in dict(precision=32, couplingInterval=1, solidContactCouplingInterval=1).items():
            replace_entry(case/'constant/scheme', key, value)
        with (case/'solver.log').open('x') as output:
            subprocess.run([args.binary, '-case', case], stdout=output, stderr=subprocess.STDOUT, check=True)
        log = (case/'solver.log').read_text()
        marker, = [line for line in log.splitlines() if line.startswith('MOMENTUM_INPUT ')]
        assert ('fluidProperties/turbulence' if name=='inline' else str(model_file)) in marker
        enabled = name != 'explicit-laminar'
        assert ('simulationType=RAS' if enabled else 'simulationType=laminar') in marker
        assert ('Gas turbulence: RAS kOmegaSST' in log) == enabled
        evidence[name] = dict(selection=marker)
        if enabled:
            terminal, = [p for p in case.iterdir() if p.is_dir() and (p/'uniform/time').is_file() and float(p.name)==end]
            fields[name] = {p.relative_to(terminal):p.read_bytes() for p in terminal.rglob('*') if p.is_file()}
            values = {}
            for key in ('k','omega','nut'):
                first, last = [float(entry(p/key, 'internalField').split()[1]) for p in (case/'0',terminal)]
                assert last > 0 and last != first
                values[key] = [first,last]
            evidence[name]['fields'] = values
    assert fields['inline'] and fields['inline']==fields['explicit-sst']
    evidence['matching_terminal_files'] = len(fields['inline'])
    (args.output/'results.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('CHT turbulence selection and SST fields passed')


if __name__ == '__main__':
    main()
