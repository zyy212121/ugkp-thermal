#!/usr/bin/env python3
"""Prepare the authorized MSS7 inputs; never launches a solver or alters originals."""
from pathlib import Path
import json
import re
import shutil
from checkpoint import ROOT, WORK, input_digest

CASES=ROOT/'cases/mss7-baselines'
SOURCES={'CHT':CASES/'case-current'}
START={'CHT':'0.29500248649130967'}

def clean(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)

def block_span(text,name):
    match=re.search(r'(?<![\w])'+re.escape(name)+r'\s*\{',text)
    if not match: raise ValueError('missing dictionary '+name)
    begin=match.end();depth=1
    for i in range(begin,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0:return match.start(),begin,i,i+1
    raise ValueError('unclosed dictionary '+name)

def block(text,name):
    _,begin,end,_=block_span(text,name);return text[begin:end]

def without_header(path):
    text=clean(path.read_text());begin,_,_,end=block_span(text,'FoamFile')
    return text[:begin]+text[end:]

def values(text,names):
    result=[]
    for name in names.split():
        match=re.search(r'(?<![\w])'+re.escape(name)+r'\s+([^;{}]+);',text)
        if not match: raise ValueError('missing legacy control '+name)
        result.append(name+' '+match.group(1).strip()+';')
    return '\n'.join(result)

def set_entry(text,key,value):
    pattern=r'(?m)^\s*'+re.escape(key)+r'\s+[^;]+;'
    result,count=re.subn(pattern,'\n'+key+' '+str(value)+';',text)
    if count>1:raise ValueError('ambiguous top-level control '+key)
    return result if count else result+'\n'+key+' '+str(value)+';\n'

def scheme(case,kind,bits):
    solid=block(clean((case/'constant/solidRegionProperties').read_text()),'solidThermalCoupling')
    radiation=block(block(clean((case/'constant/radiationProperties').read_text()),'model'),'do')
    nonlinear=values(solid,'nonlinearMaxIterations nonlinearRelativeTolerance energyRelativeTolerance energyAbsoluteTolerance nonlinearAitken nonlinearInitialRelaxation nonlinearMinimumRelaxation nonlinearMaximumRelaxation')
    linear=block(block(clean((case/'system/graphite/fvSolution').read_text()),'solvers'),'TsolidCandidate')
    lines=[
        'FoamFile { version 2.0; format ascii; class dictionary; object scheme; }',
        '// Numerical values migrated from this archived input; physical dictionaries retained.',
        f'gpuPrecision {bits};',
        'gasTimeStep { minDeltaT 1e-12; maxDeltaT 1e-6; maxCo 0.7; courantUpdateInterval 1000; maximumTimeStepGrowth 1.1; }',
        'thermalCoupling { '+values(solid,'couplingInterval solidContactCouplingInterval radiationCouplingFrequency')+' }',
        'gasNumerics { '+without_header(case/'system/fluid/fvSchemes')+' }',
        'solidDiscretization { '+without_header(case/'system/graphite/fvSchemes')+' }',
        'solid { nonlinear { '+nonlinear+' } linear { normal { '+linear+' } final { '+linear+' } } }',
        'radiation { do { '+values(radiation,'radiationActiveEpsilonMin maxSourceIterations sourceRelativeTolerance phaseBalanceTolerance maxPhaseBalanceIterations angleParallelThreads sourceRelaxation linearSolverTolerance linearSolverRelTol linearSolverMaxIterations')+' } }',
        'particle { tracking { maxFaceWalkHops 12; } packing { iterations 20; } }',
        'coldWall1D { solidificationIterations 4; }',
    ]
    if kind=='CHMT':
        legacy=clean((case/'constant/chmtProperties').read_text())
        lines.append('wallLayer { '+values(block(block(legacy,'sst'),'boundaryLayer'),'nodes maxIterations relativeTolerance absoluteTolerance stretch')+' }')
        lines.append('chmt { material { conduction { '+values(block(legacy,'cpuMaterial'),'nonlinearMaxIterations nonlinearRelativeTolerance energyRelativeTolerance')+' } } }')
    return '\n\n'.join(lines)+'\n'

def migrate_constant_inlet(case, start):
    """Publish the archived constant inlet schedules through native field patches."""
    physical=clean((case/'constant/particleProperties').read_text())
    patch=re.search(r'\bgpuResidentInletPatch\s+(\w+)\s*;',physical)[1]
    region=case/'constant/fluid'
    field_dir=case/start/'fluid'
    effective={}
    for name,table_key in [('p','gpuResidentPressureTable'),('epsilonS','gpuResidentVolumeFractionTable')]:
        filename=re.search(r'\b'+table_key+r'\s+([^;\s]+)\s*;',physical)[1]
        table=region/filename
        rows=re.findall(r'\(\s*([^()\s]+)\s+([^()\s]+)\s*\)',clean(table.read_text()))
        if not rows or any(float(v)!=float(rows[0][1]) for _,v in rows):
            raise ValueError(f'{table} does not define a constant MSS7 reference condition')
        value=rows[0][1]
        path=field_dir/name
        data=path.read_text()
        _,field_start,field_end,_=block_span(data,'boundaryField')
        patches=data[field_start:field_end]
        _,begin,end,_=block_span(patches,patch)
        boundary=patches[begin:end]
        if not re.search(r'\btype\s+fixedValue\s*;',boundary):
            raise ValueError(f'Unexpected {name} boundary layout on {patch}')
        old=re.search(r'\bvalue\s+uniform\s+([^;\s]+)\s*;',boundary)
        if old is None:
            raise ValueError(f'{name} is missing a uniform value on {patch}')
        if float(old[1])!=float(value):
            offset=field_start+begin+old.start(1)
            path.write_text(data[:offset]+value+data[offset+len(old[1]):])
        effective[name]={'patch':patch,'table':str(table.relative_to(case)),
            'source_value':float(old[1]),'effective_value':float(value)}
    return effective

def prepare():
    """Copy the original 120 µm case into independent one-exchange validation inputs."""
    target=WORK/'inputs-case-current-inlet'
    if target.exists():
        raise RuntimeError('The case-current input bundle is already frozen')
    target.mkdir(parents=True)
    source=SOURCES['CHT']
    start=START['CHT']
    manifest={}
    for bits in (32,64):
        slot=f'CHT{bits}'
        case=target/slot
        case.mkdir()
        for part in ('constant','system','assets',start):
            shutil.copytree(source/part,case/part,symlinks=False)
        inlet=migrate_constant_inlet(case,start)
        momentum=case/'constant/fluid/momentumTransport'
        if not momentum.exists():
            original=clean((case/'constant/fluid/fluidProperties').read_text())
            turbulence=block(original,'turbulence')
            momentum.write_text('FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }\n'+turbulence+'\n')
        (case/'constant/scheme').write_text(scheme(case,'CHT',bits))
        changes=dict(startFrom='startTime',startTime=start,endTime='0.30000348649130967',
            stopAt='endTime',maxThermalCouplingEvents=1,runTimeModifiable='false',logInterval=1000)
        control=(case/'system/controlDict').read_text()
        for key,value in changes.items():
            control=set_entry(control,key,value)
        (case/'system/controlDict').write_text(control)
        paths=sorted(str(p.relative_to(case)) for p in case.rglob('*') if p.is_file())
        manifest[slot]=dict(source=str(source),start_time=start,case=str(case),input_files=paths,
            input_digest=input_digest(case,paths),precision=bits,heat_events=1,radiation_events=1,
            start_sequence=59,end_sequence=60,expected_turbulence='kOmegaSST',inlet=inlet,control_overrides=changes)
    (target/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':prepare()
