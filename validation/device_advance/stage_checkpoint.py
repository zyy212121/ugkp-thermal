#!/usr/bin/env python3
"""Copy frozen inputs and verified native build outputs; does not run a case.

The resulting prepared.json is reviewable before checkpoint.py freeze. Keep
all successful run slots and prior checkpoint inputs intact on later stages.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tarfile
from checkpoint import ROOT, WORK, checkpoint_directory, PHASES, macro_slots, read, digest, input_digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=int,choices=PHASES,required=True)
    parser.add_argument('--compile-log',action='append',required=True,help='solver=/absolute/log/path; four solvers required for CP1')
    args=parser.parse_args()
    logs={}
    for item in args.compile_log:
        solver,path=item.split('=',1)
        path=Path(path).resolve()
        if solver not in ('gasUGKP','gasUGKPSteady','CHT','CHMT') or solver in logs:
            raise RuntimeError('invalid/duplicate compilation solver')
        text=path.read_text()
        if '\x00' in text or '=== Dual-precision executables ===' not in text:
            raise RuntimeError('incomplete compilation log: '+str(path))
        binaries={}
        for bits in (32,64):
            binary=ROOT/'bin/.precision'/str(bits)/solver
            sha=digest(binary)
            if sha+'  '+str(binary) not in text:
                raise RuntimeError('compiled binary does not match log: '+str(binary))
            binaries[str(bits)]=dict(path=str(binary),sha256=sha)
        logs[solver]=dict(path=str(path),sha256=digest(path),binaries=binaries)
    required={'CHT','CHMT'}|({'gasUGKP','gasUGKPSteady'} if args.checkpoint==1 else set())
    if set(logs)!=required:
        raise RuntimeError('missing required solver compilation evidence')
    available=read(WORK/'inputs-case-current-inlet/inputs.json')
    requested_slots=macro_slots(args.checkpoint)
    inputs={slot:available[slot] for slot in sorted(requested_slots)}
    directory=checkpoint_directory(args.checkpoint)
    if (directory/'manifest.json').exists() or (directory/'cases').exists():
        raise RuntimeError('checkpoint directory exists; refusing to overwrite evidence')


    for entry in inputs.values():
        if input_digest(Path(entry['case']),entry['input_files'])!=entry['input_digest']:
            raise RuntimeError('prepared input changed: '+entry['case'])
    (directory/'cases').mkdir(parents=True)
    (directory/'bin').mkdir()
    (directory/'evidence').mkdir()
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    (directory/'evidence/source.patch').write_bytes(patch)
    names=subprocess.check_output(['git','ls-files','-z','--others','--exclude-standard'],cwd=ROOT).split(b'\0')
    with tarfile.open(directory/'evidence/new-source.tar','w') as archive:
        for name in names:
            if not name:continue
            relative=Path(name.decode())
            if relative.parts[0] in {'applications','common','gpu','legacy','precision','tool','validation','docs'}:
                archive.add(ROOT/relative,arcname=str(relative),recursive=False)
    for solver,entry in logs.items():
        path=directory/'evidence'/(solver+'-compile.log')
        shutil.copy2(entry['path'],path)
        entry['path']=str(path)
    plan=dict(completed_phases=PHASES[args.checkpoint],compile_evidence=logs,slots={},
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              validation_scope='Single-card macro CHT and separate small CHMT tests. Dedicated gas/steady cross-step loops have compile/static coverage only at CP1.',
              run_budget='One successful run per case/precision; no separate baseline, warm-up or repeats.')
    for slot,entry in inputs.items():
        case=directory/'cases'/slot
        shutil.copytree(entry['case'],case)
        kind='CHMT' if slot.startswith('CHMT') else 'CHT'
        bits=str(entry['precision'])
        binary=directory/'bin'/slot
        shutil.copy2(logs[kind]['binaries'][bits]['path'],binary)
        plan['slots'][slot]={**entry,'case':str(case),'binary':str(binary)}
    save(directory/'prepared.json',plan)
    print(json.dumps(dict(prepared=str(directory/'prepared.json'),checkpoint_opened=False)))


if __name__=='__main__':main()
