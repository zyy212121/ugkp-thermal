#!/usr/bin/env python3
"""Run one authorized macro case/precision at a frozen architecture checkpoint.

No baseline solve, warm-up, repeat loop or automatic retry is provided. A
zero-exit run consumes its slot immediately, before numerical review. An error
may release that slot only through an explicit failure record with evidence.
A user-stopped run may resume only with recorded user authorization; its
interruption remains distinct from numerical failure and successful computation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.build/architecture'
PHASES = {1: ['P0','P1','P2','P3'], 2: ['P4','P5'], 3: ['P6','P7']}
SLOTS = {'CHT32','CHT64'}

def macro_slots(cp: int) -> set[str]:

    return SLOTS if cp == 1 else {'CHT32'}


def checkpoint_directory(cp: int) -> Path:

    return WORK/'cp1/case-current' if cp==1 else WORK/f'cp{cp}'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save(path: Path, value) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def read(path: Path):
    return json.loads(path.read_text())

def source_digest() -> str:
    paths = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).split(b'\0')
    h=hashlib.sha256()
    for item in sorted(set(paths)):
        if not item: continue
        relative=Path(os.fsdecode(item))
        if relative.parts[0] not in {'applications','common','gpu','legacy','precision','tool','third_party','validation'} and relative.name!='Allwmake': continue
        path=ROOT/relative
        if path.is_symlink(): data=os.readlink(path).encode()
        elif path.is_file(): data=bytes.fromhex(digest(path))
        else: continue
        h.update(item+b'\0'+data+b'\0')
    return h.hexdigest()

def input_digest(case: Path, paths: list[str]) -> str:
    h=hashlib.sha256()
    for relative in sorted(paths):
        path=case/relative
        if not path.is_file(): raise RuntimeError('missing frozen input: '+str(path))
        h.update(relative.encode()+b'\0'+bytes.fromhex(digest(path))+b'\0')
    return h.hexdigest()

def freeze(cp: int, prepared: Path) -> None:
    plan=read(prepared);directory=checkpoint_directory(cp);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists(): raise RuntimeError('checkpoint is already frozen')
    if plan.get('completed_phases')!=PHASES[cp]: raise RuntimeError('required phases have not been completed')
    if set(plan.get('slots',{}))!=macro_slots(cp): raise RuntimeError('incorrect CHT macro slots for this checkpoint')
    if not plan.get('compile_evidence'): raise RuntimeError('compile evidence is required before opening a checkpoint')
    for slot,entry in plan['slots'].items():
        case=Path(entry['case']).resolve();binary=Path(entry['binary']).resolve()
        if not case.is_relative_to(directory.resolve()) or not case.is_dir(): raise RuntimeError('case must be a checkpoint-local copy')
        if not binary.is_relative_to(directory.resolve()) or not binary.is_file(): raise RuntimeError('freeze an independent native binary')
        if entry.get('heat_events')!=1 or entry.get('radiation_events')!=1: raise RuntimeError('each run must cover exactly one heat event and one radiation event')
        if not entry.get('input_files'): raise RuntimeError('frozen input file list is required')
        entry['input_digest']=input_digest(case,entry['input_files'])
        entry['binary_sha256']=digest(binary)
        if cp>1:
            previous=read(checkpoint_directory(cp-1)/'manifest.json')['slots'][slot]
            previous_runs=read(checkpoint_directory(cp-1)/'runs.json').get(slot,[])
            accepted=[run for run in previous_runs if run['status']=='accepted']
            if len(accepted)!=1: raise RuntimeError(f'{slot}: previous checkpoint has no accepted result')
            if previous['input_digest']!=entry['input_digest']: raise RuntimeError('comparison requires the same starting state and input')
            entry['baseline']=str(checkpoint_directory(cp-1)/slot)
        else:
            entry['efficiency_comparison']=bool(entry.get('historical_baseline_verified'))
            if not entry.get('historical_baseline_verified'): entry['comparison']='analysis_only'
    plan['source_sha256']=source_digest();plan['opened_at']=time.time();plan['checkpoint']=cp
    save(directory/'manifest.json',plan);save(directory/'runs.json',{})

def revise(cp: int,evidence: Path, resume: bool=False) -> None:
    """Compile a fix after a documented checkpoint failure without resetting slots."""
    directory=checkpoint_directory(cp);report=read(evidence)
    if not report.get('reason') or not report.get('compile_evidence'):
        raise RuntimeError('revision requires the concrete correction and fresh compilation evidence')
    with (directory/'runs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read(directory/'runs.json');plan=read(directory/'manifest.json')
        interrupted=report.get('interrupted_slot') if resume else None
        if resume:
            if not report.get('user_authorization') or not report.get('process_stopped_evidence'):
                raise RuntimeError('resume requires explicit user authorization and evidence that the old process stopped')
            if interrupted not in ledger or ledger[interrupted][-1]['status']!='interrupted':
                raise RuntimeError('resume must identify the last user-interrupted attempt')
            if interrupted not in report.get('recompiled_slots',[]):
                raise RuntimeError('resume must include the interrupted slot')
        else:
            failed=report.get('failed_slot')
            if failed not in ledger or ledger[failed][-1]['status']!='failed':
                raise RuntimeError('a revision must refer to an actual failed checkpoint attempt')
        changes={}
        for slot in report.get('recompiled_slots',[]):
            if slot not in SLOTS: raise RuntimeError('invalid slot')
            if any(item['status']!='failed' and not (
                    item['status']=='interrupted' and (item.get('resume_authorization')
                    or (slot==interrupted and item is ledger[slot][-1])))
                    for item in ledger.get(slot,[])):
                raise RuntimeError('a successful or unresolved run cannot be replaced')
            entry=plan['slots'][slot]
            if input_digest(Path(entry['case']),entry['input_files'])!=entry['input_digest']:
                raise RuntimeError('a code correction must preserve the frozen physical input')
            changes[slot]={'old':entry['binary_sha256'],'new':digest(Path(entry['binary']))}
            entry['binary_sha256']=changes[slot]['new']
        if not changes: raise RuntimeError('identify the recompiled slots')
        if resume:
            ledger[interrupted][-1]['resume_authorization']={
                'at':time.time(),'user_instruction':report['user_authorization'],
                'evidence':str(evidence.resolve()),'process_stopped_evidence':report['process_stopped_evidence']}
            save(directory/'runs.json',ledger)
        plan.setdefault('revisions',[]).append({'at':time.time(),'previous_source_sha256':plan['source_sha256'],
            'report':report,'binaries':changes})
        plan['source_sha256']=source_digest();save(directory/'manifest.json',plan)

def run(cp: int, slot: str) -> int:
    directory=checkpoint_directory(cp);plan=read(directory/'manifest.json');entry=plan['slots'][slot]
    if source_digest()!=plan['source_sha256']: raise RuntimeError('source changed after checkpoint freeze')
    binary=Path(entry['binary']);case=Path(entry['case'])
    if digest(binary)!=entry['binary_sha256']: raise RuntimeError('binary changed after checkpoint freeze')
    if input_digest(case,entry['input_files'])!=entry['input_digest']: raise RuntimeError('frozen physical input changed')
    with (directory/'runs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read(directory/'runs.json');attempts=ledger.setdefault(slot,[])
        if any(item['status']!='failed' and not (
                item['status']=='interrupted' and item.get('resume_authorization')) for item in attempts):
            raise RuntimeError('this case/precision slot is already occupied')
        output=directory/slot;output.mkdir(exist_ok=True)
        log=output/f'run-{len(attempts)+1}.log'
        attempt={'status':'running','started_at':time.time(),'log':str(log),'binary_sha256':entry['binary_sha256'],
            'command':[str(binary),'-case',str(case)],'input_digest':entry['input_digest'],
            'source_sha256':plan['source_sha256'],
            'environment':{key:os.environ.get(key) for key in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES',
                 'UGKP_GAS_GRAPH','WM_PROJECT_VERSION','WM_OPTIONS')}}
        attempts.append(attempt);save(directory/'runs.json',ledger)
        begin=time.monotonic()
        try:
            with log.open('wb') as stream:
                result=subprocess.run(attempt['command'],cwd=case,stdout=stream,stderr=subprocess.STDOUT,check=False)
            attempt['exit_code']=result.returncode
            attempt['status']='computed' if result.returncode==0 else 'failed'
        except BaseException as error:


            attempt['status']='interrupted';attempt['exception']=repr(error)
            raise
        finally:
            attempt['elapsed_seconds']=time.monotonic()-begin;attempt['finished_at']=time.time()
            save(directory/'runs.json',ledger)
        return attempt['exit_code']

def elapsed_to_exchange(attempt: dict, sequence: int) -> float:
    from analyze import dictionary
    case = Path(attempt['command'][-1])
    markers = [p for p in case.glob('*/thermalExchangeState')
               if dictionary(p)['exchangeSequence'] == sequence]
    if len(markers) != 1:
        raise RuntimeError(f'{case}: cannot identify the completed exchange {sequence}')
    return markers[0].stat().st_mtime_ns / 1e9 - attempt['started_at']

def compare_performance(cp: int, slot: str, attempt: dict, sequence: int) -> dict:
    previous = read(checkpoint_directory(cp - 1) / 'runs.json')[slot]
    reference, = [r for r in previous if r['status'] == 'accepted']
    if reference['input_digest'] != attempt['input_digest']:
        raise RuntimeError('performance comparison requires identical inputs')
    timing = {
        'to_manifest_s': [elapsed_to_exchange(r, sequence) for r in (reference, attempt)],
        'to_exit_s': [r['elapsed_seconds'] for r in (reference, attempt)],
    }
    if not all(math.isfinite(t) and t > 0 for pair in timing.values() for t in pair):
        raise RuntimeError('invalid checkpoint timing evidence')
    return {'reference_checkpoint': cp - 1, 'reference_log': reference['log'],
            'candidate_log': attempt['log'], 'timing': timing,
            'regressed': [name for name, (old, new) in timing.items() if new > old]}

def review(cp: int,slot: str,status: str,evidence: Path) -> None:
    directory=checkpoint_directory(cp);report=read(evidence)
    if not report.get('reason'): raise RuntimeError('a concrete review reason is required')
    if status=='accepted' and (report.get('heat_events')!=1 or report.get('radiation_events')!=1
            or not report.get('conservation') or not report.get('single_run_spatial_temporal_statistics')):
        raise RuntimeError('acceptance requires event counts, conservation and single-run macroscopic statistics')
    performance_failure = None
    with (directory/'runs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(directory/'runs.json')
        attempt=ledger[slot][-1]
        if status=='accepted' and attempt['status']!='computed': raise RuntimeError('only a completed run can be accepted')
        if status=='accepted':
            from analyze import dictionary
            slot_plan=read(directory/'manifest.json')['slots'][slot]
            case=Path(slot_plan['case'])
            states=sorted((dictionary(p) for p in case.glob('*/thermalExchangeState')),
                          key=lambda record:record['exchangeSequence'])
            if [record['exchangeSequence'] for record in states]!=[slot_plan['start_sequence'],slot_plan['end_sequence']]:
                raise RuntimeError('Saved CHT exchanges disagree with the declared restart and endpoint')
            initial,final=states
            if (final['previousExchangeSimulationTimeS']!=initial['completedSimulationTimeS']
                or final['completedRadiationSimulationTimeS']!=final['completedSimulationTimeS']
                or initial['completedRadiationSimulationTimeS']>=initial['completedSimulationTimeS']):
                raise RuntimeError('Expected one committed radiation exchange with its saved history')
            banner=Path(attempt['log']).read_text(errors='replace')
            if not any(line.startswith('Gas turbulence: RAS kOmegaSST') and 'wallTreatment=wallFunction' in line
                       for line in banner.splitlines()):
                raise RuntimeError('The macro run did not enable the archived SST wall treatment')
            if cp > 1:
                comparison = compare_performance(cp, slot, attempt, slot_plan['end_sequence'])
                report['performance_comparison'] = comparison
                if comparison['regressed']:
                    performance_failure = 'Checkpoint performance regressed: ' + ', '.join(comparison['regressed'])
                    report.update(status='failed', performance_status='failed',
                                  numerical_review_reason=report['reason'], reason=performance_failure,
                                  error_evidence=comparison)
                    status = 'failed'
        if status=='failed' and not report.get('error_evidence'): raise RuntimeError('retry requires actual error evidence; statistical repeats are forbidden')
        if report.get('performance_comparison'):
            verdict = directory / slot / f'review-{len(ledger[slot])}-verdict.json'
            save(verdict, report)
            attempt['submitted_review_file'] = str(evidence.resolve())
            evidence = verdict
        attempt['status']=status;attempt['review']=report;attempt['review_file']=str(evidence.resolve())
        save(directory/'runs.json',ledger)
    if performance_failure:
        raise RuntimeError(performance_failure)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','revise','resume','run','accept','fail'])
    parser.add_argument('--checkpoint',type=int,choices=PHASES,required=True)
    parser.add_argument('--slot',choices=sorted(SLOTS));parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    if args.action in ('freeze','revise','resume'):
        if args.evidence is None: parser.error('--evidence prepared.json is required')
        if args.action=='freeze': freeze(args.checkpoint,args.evidence)
        else: revise(args.checkpoint,args.evidence,resume=args.action=='resume')
    else:
        if args.slot is None: parser.error('--slot is required')
        if args.action=='run': raise SystemExit(run(args.checkpoint,args.slot))
        else:
            if args.evidence is None: parser.error('--evidence review.json is required')
            review(args.checkpoint,args.slot,'accepted' if args.action=='accept' else 'failed',args.evidence)

if __name__=='__main__': main()
