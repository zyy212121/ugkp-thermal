#!/usr/bin/env python3
"""Verify and extract only the selected ARC3696 snapshots, using stdlib."""
import argparse,hashlib,json,tarfile,shutil
from pathlib import Path,PurePosixPath
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();here=Path(__file__).resolve().parent;out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);m=json.loads((here/'snapshot-manifest.json').read_text())
 for case in m['cases'].values():
  archive=here/case['archive'];assert sha(archive)==case['archive_sha256'],archive
  seen=set()
  with tarfile.open(archive,'r:gz')as tar:
   for member in tar:
    path=PurePosixPath(member.name);assert member.isfile()and not path.is_absolute()and '..'not in path.parts and member.name in case['members']
    assert member.name not in seen;seen.add(member.name);dest=(out/member.name).resolve();assert dest.is_relative_to(out)
    expected=case['members'][member.name];assert member.size==expected['bytes'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.is_file()and sha(dest)==expected['sha256'],f'Existing different file: {dest}'
    else:
     with tar.extractfile(member)as src,dest.open('wb')as dst:shutil.copyfileobj(src,dst)
    assert sha(dest)==expected['sha256'],dest
  assert seen==set(case['members']);print('VERIFIED',case['archive'],len(seen),'files')
 print('Extracted',out)
if __name__=='__main__':main()
