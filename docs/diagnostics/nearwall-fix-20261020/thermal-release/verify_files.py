from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parent
bad=[];count=0
for line in (root/"SHA256SUMS").read_text().splitlines():
    expected,name=line.split("  ",1);path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(name)
    count+=1
print(f"SHA256 verified: {count-len(bad)}/{count} files")
if bad:print("\n".join(bad));sys.exit(1)
