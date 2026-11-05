#!/usr/bin/env python3
"""Actual applications: expanded transport dictionaries and CHT wall outputs.

Requires sourced OpenFOAM 10 and the native solver binaries. --old-bin-dir
optionally demonstrates both defects with the pre-fix PR26 binaries.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import native_matrix as matrix


def use_references(case):
    path=case/'constant/gasTransportProperties'
    body=re.sub(r'FoamFile\s*\{[^}]*\}', '',path.read_text(),count=1)
    body=re.sub(r'\bmuRef\s+([^;]+);',r'referenceMu \1; muRef $referenceMu;',body)
    body=re.sub(r'\bTRef\s+([^;]+);',r'referenceT \1; TRef $referenceT;',body)
    includes=case/'constant/transport';includes.mkdir()
    (includes/'coefficients.inc').write_text(body)
    (includes/'sutherland.inc').write_text('#include "coefficients.inc"\n')
    path.write_text('FoamFile { version 2.0; format ascii; class dictionary; object gasTransportProperties; }\n#include "transport/sutherland.inc"\n')


def rows(path):
    with path.open() as stream:result=list(csv.DictReader(stream))
    assert result,path
    assert all(value is not None and math.isfinite(float(value)) for row in result for value in row.values()),path
    return result


def wall_output(case,profiles=True,sst=True):
    receipts=[]
    written=matrix.times(case)
    assert len(written)==2,written
    assert all(abs(float(p.name)-expected)<1e-15 for p,expected in zip(written,(2e-5,4e-5)))
    assert len(list(case.rglob('gasBoundaryLayer.csv')))==2
    for time in written:
        uniform=time/'uniform';diagnostics=rows(uniform/'gasBoundaryLayer.csv')
        assert len(diagnostics)==4
        assert all(float(row['viscosity'])>0 for row in diagnostics)
        flow=uniform/'gasBoundaryLayerProfiles.csv';turbulence=uniform/'gasBoundaryLayerSstProfiles.csv'
        assert flow.exists()==profiles
        assert turbulence.exists()==(profiles and sst)
        if profiles:
            nodes=rows(flow);assert len(nodes)==8*len(diagnostics)
            for face in {row['face'] for row in nodes}:
                group=[row for row in nodes if row['face']==face]
                assert [int(row['node']) for row in group]==list(range(8))
                assert all(float(a['y'])<float(b['y']) for a,b in zip(group,group[1:]))
            for row in nodes:
                assert float(row['T'])>0
                assert abs(sum(float(value) for key,value in row.items() if key.startswith('Y_'))-1)<1e-12
                assert abs(float(row['outputTime'])-float(time.name))<1e-15
                assert abs(float(row['stageTime'])-float(time.name))<1e-15
            if sst:assert len(rows(turbulence))==7*len(diagnostics)
        receipts.append({'time':float(time.name),'wall_rows':len(diagnostics),'profile_rows':32 if profiles else 0,'sst_rows':28 if profiles and sst else 0})
    return receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir',type=Path,required=True)
    parser.add_argument('--old-bin-dir',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    models=json.loads((matrix.HERE/'fixtures.json').read_text());results={}
    def prepare(name,label):
        case=work/label;matrix.prepare(matrix.HERE/'fixtures'/name,case,models[name]);return case
    def run(case,solver,old=False):
        binary=(args.old_bin_dir if old else args.bin_dir).resolve()/solver
        matrix.run([binary,'-case',case],case/'run.log')
    def save():(work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    for solver,name in [('gasUGKP','gas-single'),('CHT','cht-wall'),('CHMT','chmt-mixture')]:
        literal=prepare(name,solver+'-literal');referenced=prepare(name,solver+'-references');use_references(referenced)
        run(literal,solver);run(referenced,solver)
        if solver=='CHMT':
            parity=matrix.compare_chmt(literal,referenced)
            assert matrix.terminal(literal).read_bytes()==matrix.terminal(referenced).read_bytes(),'expanded transport configuration differs'
        else:
            parity=matrix.compare_fields(matrix.times(literal)[-1],matrix.times(referenced)[-1])
            assert max(parity.values())==0
            assert (matrix.times(literal)[-1]/'gasTransportIdentity').read_bytes()==(matrix.times(referenced)[-1]/'gasTransportIdentity').read_bytes()
        restart=matrix.resume(referenced,work/(solver+'-reference-restart'),solver,args.bin_dir.resolve()/solver)
        results[solver+'-dictionary']={'literal_reference_parity':parity,'restart':restart}
        if solver=='CHT':
            results['cht-reacting-output']=wall_output(literal)
            for time in matrix.times(literal):
                for file in (time/'uniform').glob('gasBoundaryLayer*.csv'):
                    assert file.read_bytes()==(referenced/time.name/'uniform'/file.name).read_bytes()
            if args.old_bin_dir:
                old=prepare(name,'CHT-before');run(old,solver,True)
                assert not list(old.rglob('gasBoundaryLayer*.csv')),'old binary did not reproduce missing outputs'
                parity=matrix.compare_fields(matrix.times(old)[-1],matrix.times(literal)[-1]);assert max(parity.values())==0
                results['cht-old-defect-reproduced']={'missing_csv':True,'physical_fields_unchanged':True}
        if solver=='CHMT' and args.old_bin_dir:
            old=prepare(name,'CHMT-before');use_references(old)
            with (old/'run.log').open('w') as output:
                result=subprocess.run([str(args.old_bin_dir.resolve()/solver),'-case',str(old)],env=matrix.ENV,stdout=output,stderr=subprocess.STDOUT)
            assert result.returncode!=0,'old binary did not reproduce dictionary failure'
            assert 'preprocessor directives require an expanded host dictionary' in (old/'run.log').read_text()
            results['chmt-old-defect-reproduced']={'returncode':result.returncode}
        save();print(solver,'expanded dictionary and restart PASS',flush=True)
    for name,label,profiles,sst,thermal,uncoupled in [
        ('cht-ode','cht-ode-output',True,False,False,False),
        ('cht-wall','cht-profiles-disabled',False,True,False,False),
        ('cht-wall','cht-thermal-write',True,True,True,False),
        ('cht-wall','cht-uncoupled-write',True,True,False,True)]:
        case=prepare(name,label)
        if not profiles:
            p=case/'constant/fluidProperties';assert 'writeProfiles true' in p.read_text();p.write_text(p.read_text().replace('writeProfiles true','writeProfiles false'))
        if thermal:matrix.entry(case/'system/controlDict','writeInterval',-1)
        if uncoupled:
            p=case/'constant/particleProperties';assert 'gpuResidentSolidThermalCoupling true' in p.read_text();p.write_text(p.read_text().replace('gpuResidentSolidThermalCoupling true','gpuResidentSolidThermalCoupling false'))
        run(case,'CHT');results[label]=wall_output(case,profiles,sst)
        if not profiles:
            parity=matrix.compare_fields(matrix.times(case)[-1],matrix.times(work/'CHT-literal')[-1])
            assert max(parity.values())==0,'profile output changed physical fields'
        save();print(label,'PASS',flush=True)
    case=prepare('cht-contact','cht-no-boundary-layer');run(case,'CHT')
    assert not list(case.rglob('gasBoundaryLayer*.csv'))
    results['cht-inactive-output']={'no_boundary_layer_files':True};save();print('inactive wall output PASS',flush=True)
if __name__=='__main__':main()
