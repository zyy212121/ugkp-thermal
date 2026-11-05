"""Real one-cell gas donor shared by two faces of a conformal L-shaped solid.
Conservation is checked from conservative species/masses and actual packets.
A before-fix executable can be supplied to demonstrate the original failure.
"""
import argparse,csv,json,re,shutil,subprocess
from pathlib import Path
from chmt_particles_native import m,ROOT,table

def prepare(case,faces,moving=False,empty=False):
 m.copy_fixture(ROOT/'validation/molecular_transport/fixtures/chmt-mixture',case)
 for name in ('blockMeshDict','blockMeshSolidDict'):
  p=case/'system'/name
  if faces==2:shutil.copyfile(ROOT/'validation/pr31/fixtures/shared_surface_donor'/name,p)
  else:p.write_text(p.read_text().replace('(2 2 2)','(1 1 1)'))
 p=case/'constant/chmtProperties';s=p.read_text()
 s=re.sub(r'couplingInterval [^;]+;','couplingInterval .001;',s)
 s=re.sub(r'gasMaxDt [^;]+;','gasMaxDt .001;',s)
 s+='''\nmaterial {enableSurfaceReactions true; enablePoreOutflow false;
 surfaceReactions {oxidize {A 1;temperaturePower 0;activationEnergy 0;
 condensedNu (-.028 0);gasNu (-.028 .056);order (0 0);gasOrder (0 0);}}}
'''
 if moving:s+='meshMotion {policy CoupledRecession;}\n'
 p.write_text(s)
 for name,value in [('Y_S0',0 if empty else 1e-6),('Y_S1',1 if empty else .999999)]:
  p=case/'0'/name;p.write_text(re.sub(r'internalField uniform [^;]+;',f'internalField uniform {value};',p.read_text()))
 p=case/'constant/gasModelProperties';p.write_text(p.read_text().replace('diffusion { model none; }','diffusion {model constant;coefficients (10000 10000);}'))
 m.entry(case/'system/controlDict','deltaT',.001)
 m.run(['blockMesh','-case',case],case/'mesh.log')
 m.run(['blockMesh','-case',case,'-region','solid','-dict',case/'system/blockMeshSolidDict'],case/'mesh-solid.log')

def check(case):
 output=case/'chmtOutput';g0=table(output/'gas-0.csv')[0];g1=table(output/'gas-1.csv')[0]
 solid0=table(output/'material-0.csv');solid1=table(output/'material-1.csv')
 consumed=float(g0['mass_S0'])-float(g1['mass_S0'])
 product=float(g1['mass_S1'])-float(g0['mass_S1'])
 material=sum(float(a['mass_C0'])-float(b['mass_C0']) for a,b in zip(solid0,solid1))
 assert float(g1['mass_S0'])>=0 and float(g1['mass_S1'])>=0
 assert abs(product-2*consumed)<2e-14,(product,consumed)
 assert abs(material-consumed)<5e-13,(material,consumed)
 physical=0.
 for before,after in zip(solid0,solid1):physical+=float(before['volume'])-float(after['volume'])
 return dict(reactant_remaining=float(g1['mass_S0']),reactant_consumed=consumed,product_generated=product,
     material_consumed=material,solid_volume_change=physical)

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
 a.add_argument('--solver',type=Path,default=ROOT/'applications/CHMT/.build/native/CHMT')
 a.add_argument('--before',type=Path);args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 reports=[]
 for faces,moving,empty in ((1,False,False),(2,False,False),(2,True,False),(2,False,True)):
  case=out/f'faces{faces}-moving{int(moving)}-empty{int(empty)}';prepare(case,faces,moving,empty)
  if args.before and faces==2 and not moving and not empty:
   old=out/'before';shutil.copytree(case,old)
   with (old/'run.log').open('w') as log:run=subprocess.run([args.before,'-case',old],env=m.ENV,stdout=log,stderr=subprocess.STDOUT)
   assert run.returncode!=0 and 'coupled interface trace/packet failed' in (old/'run.log').read_text()
  seconds=m.run([args.solver,'-case',case],case/'run.log');result=check(case)
  if moving:assert abs(result['solid_volume_change']-result['material_consumed']/1000)<3e-15,result
  if empty:assert result['reactant_consumed']==result['product_generated']==result['material_consumed']==0,result
  reports.append(dict(faces=faces,moving=moving,empty=empty,seconds=seconds,**result))
 (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
