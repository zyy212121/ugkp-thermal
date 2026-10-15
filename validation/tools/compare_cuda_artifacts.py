from pathlib import Path
import hashlib,json,re,subprocess,sys
import argparse
parser=argparse.ArgumentParser(description="Offline SASS and resource comparison; no CUDA execution")
parser.add_argument('--before',type=Path,required=True)
parser.add_argument('--after',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--require-unchanged',action='store_true',help='Fail on changed or added functions; removed functions are reported')
args=parser.parse_args()
current=args.after;dest=args.out;dest.mkdir(parents=True,exist_ok=True)

def dump(obj,kind):return subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-'+kind,str(obj)],text=True)
def demangle(names):
    q=subprocess.run(['c++filt'],input='\n'.join(names)+'\n',capture_output=True,text=True,check=True)
    return dict(zip(names,q.stdout.splitlines()))
def sass(t):
    parts=re.split(r'Function : (\S+)',t);names=parts[1::2];maps=demangle(names);result={}
    for n,b in zip(names,parts[2::2]):
        # Offsets in instruction annotations are relative to each function.
        # Retain instructions, registers, branch targets and scheduling words.
        instructions=[]
        for m in re.finditer(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',b,re.S):
            instructions.append([m[1],re.sub(r'\s+',' ',m[2]).strip(),m[3],m[4]])
        assert instructions,n
        assert maps[n] not in result,maps[n]
        result[maps[n]]=instructions
    return result
def resources(t):
    parts=re.split(r' Function (\S+):\n',t);names=parts[1::2];maps=demangle(names)
    return {maps[n]:dict(re.findall(r'(\w+(?:\[\d+\])?):(\d+)',b.split('\n')[0])) for n,b in zip(names,parts[2::2])}
rows=[]
for obj in sorted(current.glob('*.o')):
    old=args.before/obj.name;assert old.exists()
    dump_data={}
    for label,p in [('before',old),('after',obj)]:
        for kind in ['sass','resource-usage']:
            t=dump(p,kind);(dest/(obj.stem+'-'+label+'-'+kind+'.txt')).write_text(t);dump_data[label,kind]=t
    a,b=sass(dump_data['before','sass']),sass(dump_data['after','sass'])
    ar,br=resources(dump_data['before','resource-usage']),resources(dump_data['after','resource-usage'])
    assert set(a)==set(ar) and set(b)==set(br),(obj.name,len(a),len(ar),len(b),len(br))
    changed=[]
    for n in sorted(set(a)&set(b)):
        if a[n]!=b[n] or ar[n]!=br[n]:
            semantic_same=[x[:2]+x[3:] for x in a[n]]==[x[:2]+x[3:] for x in b[n]]
            changed.append(dict(function=n,instructions_before=len(a[n]),instructions_after=len(b[n]),instructions_and_schedule_equal=semantic_same,machine_words_equal=a[n]==b[n],resources_before=ar[n],resources_after=br[n]))
    row=dict(configuration=obj.stem,before_functions=len(a),after_functions=len(b),removed=sorted(set(a)-set(b)),added=sorted(set(b)-set(a)),changed=changed)
    rows.append(row);print(obj.stem,'same',len(set(a)&set(b))-len(changed),'changed',len(changed),'removed',len(row['removed']),'added',len(row['added']),flush=True)
(dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Offline machine-code comparison saved; no CUDA execution or timing',flush=True)

if args.require_unchanged and any(r["changed"] or r["added"] for r in rows):sys.exit(1)
