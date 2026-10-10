"""Small ASCII input helpers; all writes target disposable cases only."""
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
ROOT=Path(__file__).resolve().parents[2]

def module(relative):
    path=ROOT/relative;spec=importlib.util.spec_from_file_location('case_dependency_'+hashlib.sha1(str(path).encode()).hexdigest(),path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out

def header(name,kind='dictionary'):
    return f'FoamFile {{ version 2.0; format ascii; class {kind}; object {name}; }}\n'

def scalar(name,dims,value,patches,vector=False):
    internal=('nonuniform List<scalar>\n'+str(len(value))+'\n(\n'+'\n'.join(format(x,'.17g') for x in value)+'\n)') if isinstance(value,list) else 'uniform '+str(value)
    return header(name,'volVectorField' if vector else 'volScalarField')+f'dimensions [{dims}];\ninternalField {internal};\nboundaryField {{\n'+''.join(f' {p} {{ {spec} }}\n' for p,spec in patches.items())+'}\n'

def write_field(case,region,name,dims,value,patches,vector=False):
    p=Path(case)/'0'/region/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(scalar(name,dims,value,patches,vector))

def replace_entry(path,key,value):
    """Replace one top-level primitive entry, independently of line layout."""
    path=Path(path);text=path.read_text()
    # Keep offsets while excluding comments/quoted values from key detection.
    masked=re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
        lambda m:re.sub(r'[^\n]',' ',m.group()),text,flags=re.S)
    depth=0;entry_start=True;matches=[]
    for token in re.finditer(r'[{};]|[A-Za-z_]\w*',masked):
        word=token.group()
        if word=='{':depth+=1;entry_start=False
        elif word=='}':
            depth-=1
            if depth==0:entry_start=True
        elif word==';':
            if depth==0:entry_start=True
        elif depth==0 and entry_start:
            entry_start=False
            if word==key:
                end=masked.find(';',token.end())
                if end<0 or any(c in masked[token.end():end] for c in '{}'):
                    raise ValueError('expected primitive dictionary entry: '+key)
                matches.append((token.start(),end+1))
    if len(matches)>1:raise ValueError('duplicate top-level dictionary entry: '+key)
    replacement=key+' '+str(value)+';'
    if matches:
        start,end=matches[0];text=text[:start]+replacement+text[end:]
    else:text+='\n'+replacement+'\n'
    path.write_text(text)

def copy_inputs(source,dest):
    source=Path(source);dest=Path(dest)
    for directory in ('0','constant','system'):
        if (source/directory).is_dir():
            shutil.copytree(source/directory,dest/directory,ignore=shutil.ignore_patterns('README*','readme*','*.png','*.pdf'))
    for p in dest.rglob('*'):
        if p.is_file() and p.read_bytes()[:60].startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Input is an unresolved Git LFS pointer: '+str(p))

def fingerprint(case):
    case=Path(case);h=hashlib.sha256()
    for p in sorted(case.rglob('*')):
        if p.is_file() and (p.relative_to(case).parts[0] in ('0','constant','system')):
            h.update(str(p.relative_to(case)).encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()

def block_mesh(x0,x1,y0,y1,z,nx,ny,patch_names=('inlet','outlet','wall','top','sides'),grading=1.):
    inlet,outlet,wall,top,sides=patch_names
    return header('blockMeshDict')+f'''convertToMeters 1;
vertices (({x0} {y0} 0) ({x1} {y0} 0) ({x1} {y1} 0) ({x0} {y1} 0)
 ({x0} {y0} {z}) ({x1} {y0} {z}) ({x1} {y1} {z}) ({x0} {y1} {z}));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 {grading} 1)); edges ();
boundary (
 {inlet} {{ type patch; faces ((0 4 7 3)); }}
 {outlet} {{ type patch; faces ((1 2 6 5)); }}
 {wall} {{ type wall; faces ((0 1 5 4)); }}
 {top} {{ type patch; faces ((3 7 6 2)); }}
 {sides} {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
); mergePatchPairs ();
'''
