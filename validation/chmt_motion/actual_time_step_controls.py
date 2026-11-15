#!/usr/bin/env python3
pass


import argparse,hashlib,importlib.util,json,math,os,re,shutil,subprocess,time
from pathlib import Path
DT=2.0**-22
LIMIT=DT/2
WINDOW=4*DT
END=2*WINDOW

def set_entry(path,key,value):
    text=path.read_text();pattern=r'\b'+re.escape(key)+r'\s+[^;{}]+;'
    text,n=re.subn(pattern,key+' '+str(value)+';',text)
    if not n:text+='\n'+key+' '+str(value)+';\n'
    path.write_text(text)

def module_entry(path,module,key,value):
    text=path.read_text();pattern=r'(\b'+re.escape(module)+r'\s*\{)([^{}]*)(\})'
    match=re.search(pattern,text)
    assert match,(path,module)
    body,n=re.subn(r'\b'+re.escape(key)+r'\s+[^;{}]+;',key+' '+str(value)+';',match[2])
    if not n:body+='\n    '+key+' '+str(value)+';\n'
    path.write_text(text[:match.start()]+match[1]+body+match[3]+text[match.end():])

def copy_inputs(source,target):
    target.mkdir()
    for name in ('0','constant','system','assets'):
        if (source/name).exists():shutil.copytree(source/name,target/name)

def configure(case,bits,adaptive,app):
    control=case/'system/controlDict';scheme=case/'constant/scheme'

    scheme.write_text(re.sub(r'\badjustTimeStep\s+[^;{}]+;','',scheme.read_text()))
    control.write_text(re.sub(r'\b(?:maxCo|maxDeltaT)\s+[^;{}]+;','',control.read_text()))
    for key,value in {'adjustTimeStep':'true' if adaptive else 'false','deltaT':DT,
        'endTime':END,'startFrom':'startTime','startTime':0,'logInterval':1,
        'writeControl':'timeStep','writeInterval':1,'timePrecision':17,'writePrecision':17}.items():set_entry(control,key,value)
    set_entry(scheme,'precision',bits)
    for key,value in {'maxCo':.4,'maxDeltaT':LIMIT,'courantUpdateInterval':1,'maximumTimeStepGrowth':2}.items():module_entry(scheme,'gasTimeStep',key,value)
    if app in ('CHT','CHMT'):
        module_entry(scheme,'thermalCoupling','couplingInterval',WINDOW)
        if app=='CHT':module_entry(scheme,'thermalCoupling','solidContactCouplingInterval',WINDOW)
    ct=control.read_text();st=scheme.read_text()
    assert len(re.findall(r'\badjustTimeStep\s+',ct))==1
    assert not re.search(r'\badjustTimeStep\s+',st)
    assert not re.search(r'\b(?:maxCo|maxDeltaT)\s+',ct)
    gas_body=re.search(r'\bgasTimeStep\s*\{([^{}]*)\}',st)[1]
    for key,value in [('maxCo',.4),('maxDeltaT',LIMIT)]:
        vals=re.findall(r'\b'+key+r'\s+([^;{}]+);',gas_body);assert len(vals)==1 and float(vals[0])==value,(key,vals)
    config={'precision':bits,'adaptive':adaptive,'initialDeltaT':DT,'schemeMaxDeltaT':LIMIT,
        'maxCo':.4,'window':WINDOW,'end':END,'controlDictSHA256':hashlib.sha256(ct.encode()).hexdigest(),
        'schemeSHA256':hashlib.sha256(st.encode()).hexdigest()}
    (case/'actual-input.json').write_text(json.dumps(config,indent=2));return config

def run(solver,case,app,adaptive):
    env=dict(os.environ,OMP_NUM_THREADS='4');start=time.monotonic()
    with (case/'run.log').open('w') as log:
        result=subprocess.run([str(solver),'-case',str(case)],stdout=log,stderr=subprocess.STDOUT,env=env)
    (case/'run.exit').write_text(str(result.returncode)+'\n')
    assert result.returncode==0,(app,case,result.returncode)
    text=(case/'run.log').read_text(errors='replace')
    if app!='CHMT':
        rows=[(int(s),float(t),float(dt)) for s,t,dt in re.findall(r'UGKP_REPRO step=(\d+) time=([^ ]+) dt=([^ ]+)',text)]
        assert rows,(case,'no actual-step logs')
        assert all(math.isfinite(t) and math.isfinite(dt) and dt>0 for _,t,dt in rows)
        assert abs(rows[-1][1]-END)<=END*2e-6,(case,rows[-1])
        if adaptive:
            assert all(dt<=LIMIT*(1+2e-6) for _,_,dt in rows),(case,rows)
            assert any(dt<DT*.75 for _,_,dt in rows),(case,'adaptive did not reduce initial dt')
            assert len(rows)>=16,(case,rows)
        else:
            assert len(rows)==8,(case,rows)
            assert all(abs(dt-DT)<=DT*2e-6 for _,_,dt in rows),(case,rows)
        return {'elapsedSeconds':time.monotonic()-start,'actualSteps':rows}
    rows=[(int(s),float(t),int(n)) for s,t,n in re.findall(r'CHMT_REPRO step=(\d+) time=([^ ]+) gasMicrosteps=(\d+)',text)]
    assert len(rows)==2,(case,rows)
    assert all(math.isfinite(t) and n>0 for _,t,n in rows)
    assert abs(rows[-1][1]-END)<=END*2e-5,(case,rows)
    if adaptive:assert all(n>=8 for _,_,n in rows),(case,rows)
    else:assert all(n==4 for _,_,n in rows),(case,rows)
    return {'elapsedSeconds':time.monotonic()-start,'actualWindowGasMicrosteps':rows,
        'observedMeanMicrostepPerWindow':[WINDOW/n for _,_,n in rows]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--binary-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--warm-root',type=Path,required=True);a=p.parse_args()
    root=a.source_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    script=root/'validation/transient_particles/check_driver.py';spec=importlib.util.spec_from_file_location('actual_step_driver',script);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    results=[]
    for app in ('gasUGKP','CHT','CHMT'):
      for bits in (32,64):
        for adaptive in (False,True):
          case=out/(app+'-'+str(bits)+'-'+('adaptive' if adaptive else 'fixed'))
          if app=='gasUGKP':
            driver.prepare_adaptive_case(case,adaptive)

            pp=case/'constant/particleProperties'
            pp.write_text(re.sub(r'\bdSigma\s+[^;]+;', 'dn 1;', pp.read_text()))
          else:
            donor=a.warm_root/(app.lower()+'-warm'+str(bits))/('below' if app=='CHT' else 'reboundContact')
            copy_inputs(donor,case)
          config=configure(case,bits,adaptive,app)
          solver=(a.binary_root/str(bits)/app).resolve()
          report={'app':app,'case':str(case),'input':config,'binarySHA256':hashlib.sha256(solver.read_bytes()).hexdigest()}
          try:report.update(run(solver,case,app,adaptive));report['passed']=True
          except Exception as error:
            report.update(passed=False,error=str(error));results.append(report);(out/'results.json').write_text(json.dumps(results,indent=2));raise
          results.append(report);(out/'results.json').write_text(json.dumps(results,indent=2));print(app,bits,adaptive,'PASS',flush=True)
    print('All 12 actual fixed/adaptive application runs PASS',flush=True)
if __name__=='__main__':main()
