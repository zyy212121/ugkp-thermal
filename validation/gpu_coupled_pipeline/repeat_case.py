#!/usr/bin/env python3
"""Rerun an unchanged, previously validated input with another native binary.

Reuses the existing baseline evidence; this is numerical validation, not an
alternating performance comparison. A fresh case contains only 0/constant/system.
"""
import argparse
import hashlib
import json
from pathlib import Path
import layer1


def input_hash(case):
    digest=hashlib.sha256()
    for directory in ('0','constant','system'):
        for path in sorted((case/directory).rglob('*')):
            if path.is_file():digest.update(str(path.relative_to(case)).encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--solver',choices=('CHMT','CHT'),required=True)
    parser.add_argument('--grid',type=int,default=2)
    args=parser.parse_args()
    args.reference=args.reference.resolve();args.binary=args.binary.resolve();args.work=args.work.resolve()
    args.work.mkdir(parents=True,exist_ok=False);case=args.work/'case'
    layer1.matrix.copy_input(args.reference,case)
    reference_hash=input_hash(args.reference);candidate_hash=input_hash(case)
    assert reference_hash==candidate_hash
    report=dict(reference=str(args.reference),binary=str(args.binary),binary_sha256=layer1.digest(args.binary),input_sha256=reference_hash,scope='unchanged-input numerical repeat; not a performance comparison')
    (args.work/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    report['elapsed_seconds']=layer1.matrix.run([args.binary,'-case',case],case/'run.log')
    if args.solver=='CHMT':
        report['parity']=layer1.compare_chmt(args.reference,case)
        report['audit']=layer1.audit_chmt(case,closed=True)
    else:
        report['parity']=layer1.compare_fields(args.reference,case)
        report['audit']=layer1.audit_cht(case,args.grid)
    report['warnings']=dict(foam_warning_count=(case/'run.log').read_text().count('FOAM Warning'))
    (args.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
