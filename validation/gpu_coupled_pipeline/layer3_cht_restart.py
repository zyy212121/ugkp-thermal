#!/usr/bin/env python3
"""Resume CHT with the existing physical-field restart oracle."""
import argparse
import json
from pathlib import Path
import re
import shutil
import layer1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();a.reference=a.reference.resolve();a.work.mkdir(parents=True,exist_ok=False)
    times=[x for x in layer1.matrix.times(a.reference) if float(x.name)>0]
    assert len(times)>=2
    checkpoint=times[0];case=a.work/'case';layer1.matrix.copy_input(a.reference,case)
    shutil.copytree(checkpoint,case/checkpoint.name)
    control=case/'system/controlDict'
    text,count=re.subn(r'startFrom\s+\w+;', 'startFrom latestTime;',control.read_text())
    assert count==1;control.write_text(text)
    result={'status':'RUNNING','reference':str(a.reference),'checkpoint':str(checkpoint),'binary_sha256':layer1.digest(a.binary)}
    def save():(a.work/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        result['elapsed_seconds']=layer1.matrix.run([a.binary,'-case',case],case/'run.log')
        actual=layer1.matrix.times(case)
        assert len(actual)==len(times)
        result['restart_field_errors']={}
        for old,new in zip(times,actual):
            assert float(old.name)==float(new.name)
            result['restart_field_errors'][old.name]=layer1.matrix.compare_fields(old,new,restart=True)
        result['oracle_scope']='Existing native restart oracle: gas and solid physical fields at 2e-12 scaled tolerance; empty-particle dMeanCell is an existing bridge limitation and is reported, not omitted.'
        result['status']='PASS';save()
    except BaseException as error:
        result.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
