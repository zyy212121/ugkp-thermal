from pathlib import Path
import sys, shutil, subprocess
r=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');out=Path(__file__).parent
before=out/'baseline';shutil.copytree(r/'common',before/'common',dirs_exist_ok=True)
for p in (out/'before/common').rglob('*'):
 if p.is_file():q=before/'common'/p.relative_to(out/'before/common');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
sys.path.insert(0,str(r/'tests/gas_transport'))
from test_gas_budget_audit import physical_fixture
from test_legacy_ale_transport import fixture as sst_fixture
from test_boundary_layer_operators import SETUP
pre=(r/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes')[0]
body=physical_fixture()+r'''
void dump(State&s){for(int c=0;c<2;++c){std::printf("%a %a %a %a %a %a %a\n",double(s.rho[c]),double(s.rhoE[c]),double(s.rhoUx[c]),double(s.rhoUy[c]),double(s.rhoUz[c]),double(s.gasSpecies.rho[c]),double(s.gasSpecies.rho[2+c]));}}
int main(){for(int rk=1;rk<=3;++rk){State s;initialise(s);if(rk!=1)for(int c=0;c<2;++c){threadIdx.x=c;saveGasConservativeStateKernel(&s);}stage(s,.001,1);if(rk!=1){stage(s,.001,2);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(rk==2?.5:.75),Real(rk==2?.5:.25));}}if(rk==3){stage(s,.001,4);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/Real(3),Real(2)/Real(3));}}dump(s);}}
'''
sst=sst_fixture()+SETUP+r'''
int main(){for(int moving=0;moving<2;++moving)for(int rk=1;rk<=3;++rk){State s;initialise(s);wall(s);if(moving){geometry(s);oldV[0]=1;newV[0]=2;oldV[1]=1;newV[1]=2;}
s.sstPhiRhoK[0]=3;s.sstPhiRhoOmega[0]=-2;s.sstPhiRhoK[1]=-.4;const double dt=.01;
for(int c=0;c<2;++c){threadIdx.x=c;applySstWallFunctionStateKernel(&s);saveGasConservativeStateKernel(&s);}
auto step=[&](){for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);recoverSstPrimitivesKernel(&s);}};step();if(rk>1){step();for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(rk==2?.5:.75),Real(rk==2?.5:.25));recoverSstPrimitivesKernel(&s);}}if(rk==3){step();for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/Real(3),Real(2)/Real(3));recoverSstPrimitivesKernel(&s);}}
for(int c=0;c<2;++c)std::printf("%a %a %a %a\n",double(s.rhoK[c]),double(s.rhoOmega[c]),double(s.k[c]),double(s.omega[c]));}}
'''
# Fixture already exposes stage geometry via moving(State&) rather than an assumed helper.
sst=sst.replace('geometry(s);oldV[0]=1;newV[0]=2;oldV[1]=1;newV[1]=2;','s.gasGeometry.enabled=true; s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;')
sst=sst.replace('int main(){','int main(){Real oldV[2]={1,1},newV[2]={2,2};',1)
for name,code in [('gas',body),('sst',sst)]:
 src=out/(name+'_comparison.cpp');src.write_text(pre+code)
 for tag,common in [('before',before/'common'),('after',r/'common')]:
  exe=out/(name+'_'+tag);cmd=['g++','-std=c++17','-O0','-I'+str(common),'-I'+str(common/'gasNumerics'),'-I'+str(r/'tests/gas_transport'),str(src),'-o',str(exe)]
  built=subprocess.run(cmd,capture_output=True,text=True);(out/(name+'_'+tag+'.build.log')).write_text(built.stdout+built.stderr);assert built.returncode==0,built.stderr
  run=subprocess.run([str(exe)],capture_output=True,text=True);(out/(name+'_'+tag+'.txt')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
 assert (out/(name+'_before.txt')).read_bytes()==(out/(name+'_after.txt')).read_bytes(),name+' mismatch'
 print(name,'bitwise hex-float output matches')
