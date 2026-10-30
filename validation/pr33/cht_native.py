"""Native CHT contact/solid energy audit using this checkout's executable."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'validation/molecular_transport'),str(ROOT/'validation/gpu_coupled_pipeline')]
import native_matrix as matrix
import layer1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve()
    if output.exists():raise SystemExit('Use a new output directory')
    meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
    matrix.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact',output,meta)
    binary=ROOT/'bin/CHT'
    seconds=matrix.run([binary,'-case',output],output/'run.log')
    report={'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'seconds':seconds,'audit':layer1.audit_cht(output)}
    (output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
