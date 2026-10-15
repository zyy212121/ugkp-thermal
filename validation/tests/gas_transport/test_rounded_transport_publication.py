"""Actual cell-update publication, including ALE volume units and rejection."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def test_repeated_conservative_updates_keep_original_density_closure(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasPhiRho[0]=Real(.0137);
s.gasSpecies.flux[0]=s.gasPhiRho[0]*Real(.23);s.gasSpecies.flux[3]=s.gasPhiRho[0]-s.gasSpecies.flux[0];
for(int step=0;step<20000;++step)for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,Real(.0001));
 ck(s.gasSpecies.cellStatus[c]==0,"valid repeated transport rejected");
 const Real sum=s.gasSpecies.rho[c]+s.gasSpecies.rho[2+c];
 ck(std::abs(sum-s.rho[c])<=s.gasSpecies.densityClosureTolerance*s.rho[c],"independent rounded updates drifted beyond original density gate");}
}
''')


def test_finite_density_and_flux_defects_publish_actual_candidates(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int mode=0;mode<3;++mode){State s;initialise(s);
 if(mode==0)s.gasSpecies.rho[0]+=Real(.01);
 else if(mode==1){s.gasPhiRho[0]=0;s.gasSpecies.flux[0]=Real(.01);}
 else {s.gasSpecies.rho[0]=Real(-.01);s.gasSpecies.rho[2]=Real(1.01);s.gasPhiRho[0]=0;s.gasSpecies.flux[0]=Real(-2);s.gasSpecies.flux[3]=Real(2);}
 for(int repeat=0;repeat<4;++repeat){
  const Real expectedA=s.gasSpecies.rho[0]+Real(.01)*(-s.gasSpecies.flux[0]);
  const Real expectedB=s.gasSpecies.rho[2]+Real(.01)*(-s.gasSpecies.flux[3]);
  const Real expectedRho=s.rho[0]+Real(.01)*(-s.gasPhiRho[0]);
  threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01));
  ck(s.gasSpecies.cellStatus[0]==0,"finite defect refused");
  ck(s.rho[0]==expectedRho&&s.gasSpecies.rho[0]==expectedA&&s.gasSpecies.rho[2]==expectedB,"actual complete candidate not published");
 }
}}
''')


def test_ale_publication_uses_each_actual_old_and_new_volume(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweep[3]={Real(.1),0,0};
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;
s.gasGeometry.absoluteGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();s.gasGeometry.relativeGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();
const Real initialMass=s.rho[0]*oldV[0]+s.rho[1]*oldV[1];
const Real initialA=s.gasSpecies.rho[0]*oldV[0]+s.gasSpecies.rho[1]*oldV[1];
s.gasPhiRho[0]=Real(.0137);s.gasSpecies.flux[0]=s.gasPhiRho[0]*Real(.23);s.gasSpecies.flux[3]=s.gasPhiRho[0]-s.gasSpecies.flux[0];
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,.001);
 ck(s.gasSpecies.cellStatus[c]==0,"valid ALE publication rejected");
 ck(std::abs(s.gasSpecies.rho[c]+s.gasSpecies.rho[2+c]-s.rho[c])<=s.gasSpecies.densityClosureTolerance*s.rho[c],"ALE density closure");}
const Real eps=Real(32)*std::numeric_limits<Real>::epsilon();
ck(std::abs(s.rho[0]*newV[0]+s.rho[1]*newV[1]-initialMass)<=eps*initialMass,"ALE total mass changed");
ck(std::abs(s.gasSpecies.rho[0]*newV[0]+s.gasSpecies.rho[1]*newV[1]-initialA)<=eps*initialA,"ALE constituent mass changed");
}
''')


def test_opposite_finite_faces_publish_their_actual_cancellation(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasSpecies.flux[0]=Real(.01);s.gasSpecies.flux[1]=Real(-.01);
// Both faces have zero total mass flux, but their species defects cancel at cell0.
const Real before[]={s.rho[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoE[0]};
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01));
ck(s.gasSpecies.cellStatus[0]==0,"finite per-face defects blocked actual cancelling transport");
const Real after[]={s.rho[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoE[0]};
ck(std::memcmp(before,after,sizeof before)==0,"bad face published state");}
''')


def test_incoming_legal_defect_is_preserved_without_repair_allowance(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);
s.gasSpecies.rho[0]=Real(1)+Real(32)*std::numeric_limits<Real>::epsilon();s.gasSpecies.rho[2]=0;
const Real before[]={s.rho[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoE[0]};
for(int repeat=0;repeat<4;++repeat){threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01));
 ck(s.gasSpecies.cellStatus[0]==0,"original legal closure tolerance was tightened");
 const Real after[]={s.rho[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoE[0]};
 ck(std::memcmp(before,after,sizeof before)==0,"zero-flux update erased existing legal defect");}
s.gasSpecies.rho[0]=Real(1)+Real(65)*std::numeric_limits<Real>::epsilon();
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01));
ck(s.gasSpecies.cellStatus[0]==0&&s.gasSpecies.rho[0]==Real(1)+Real(65)*std::numeric_limits<Real>::epsilon(),"finite existing defect was refused or normalized");
}
''')


def test_actual_published_ale_mass_elements_and_energy_match_final_face_budget(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweep[3]={Real(.1),0,0};
const double dt=.001;s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=dt;
s.gasGeometry.absoluteGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();s.gasGeometry.relativeGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();
s.gasPhiRho[0]=Real(.0137);s.gasSpecies.flux[0]=s.gasPhiRho[0]*Real(.23);s.gasSpecies.flux[3]=s.gasPhiRho[0]-s.gasSpecies.flux[0];s.gasPhiRhoE[0]=Real(4.19);
for(int c=0;c<2;++c){
 const long double sign=c==0?-1:1;
 const long double m0=(long double)s.rho[c]*oldV[c],e0=(long double)s.rhoE[c]*oldV[c];
 long double species0[2]={(long double)s.gasSpecies.rho[c]*oldV[c],(long double)s.gasSpecies.rho[2+c]*oldV[c]};
 threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);ck(s.gasSpecies.cellStatus[c]==0,"ALE budget update rejected");
 const long double m1=(long double)s.rho[c]*newV[c],e1=(long double)s.rhoE[c]*newV[c];
 const long double eps=64.L*std::numeric_limits<Real>::epsilon();
 ck(std::abs(m1-m0-sign*dt*s.gasPhiRho[0])<=eps*std::max(std::abs(m0),std::abs(m1)),"actual published total mass budget");
 ck(std::abs(e1-e0-sign*dt*s.gasPhiRhoE[0])<=eps*std::max(std::abs(e0),std::abs(e1)),"actual published physical energy budget");
 // Two independent element counts, not a closed-reactor zero-change check.
 const long double element[2][2]={{2/.028L,0},{1/.028L,3/.028L}};
 for(int e=0;e<2;++e){long double before=0,after=0,transfer=0;
  for(int k=0;k<2;++k){before+=element[e][k]*species0[k];after+=element[e][k]*(long double)s.gasSpecies.rho[k*2+c]*newV[c];transfer+=element[e][k]*sign*dt*s.gasSpecies.flux[k*3];}
  ck(std::abs(after-before-transfer)<=eps*std::max(std::abs(before),std::abs(after)),"actual published element transport budget");
 }
}
}
''')


def test_tiny_positive_carrier_falls_back_for_fresh_update_roundoff(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
struct State3:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,3> gasSpecies;};
int main(){State backing;initialise(backing);State3 s;
using Base=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;
static_cast<Base&>(s)=static_cast<Base&>(backing);
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
Real partial[6]={Real(.6),0,Real(.4),0,Real(1e-30),0},flux[9]={};int cellStatus[2]={},faceStatus[3]={};
s.gasSpecies.rho=partial;s.gasSpecies.flux=flux;s.gasSpecies.cellStatus=cellStatus;s.gasSpecies.faceStatus=faceStatus;
ck(partial[0]+partial[2]+partial[4]==s.rho[0],"test requires exact incoming closure");
const Real phi=Real(.0137),phiA=phi*Real(.23),phiB=phi-phiA;
s.gasPhiRho[0]=phi;flux[0]=phiA;flux[3]=phiB;
Real dt=0,rawA=0,rawB=0;
for(int i=1;i<=10000;++i){const Real step=Real(i)*Real(.0001);
 const Real bulk=s.rho[0]-step*phi,a=partial[0]-step*phiA,b=partial[2]-step*phiB;
 // A genuinely fresh rounding residual: the populated first two components
 // exceed the independently rounded bulk. The tiny last cannot absorb it.
 if(a+b>bulk){dt=step;rawA=a;rawB=b;break;}}
ck(dt>0,"no deterministic fresh-roundoff witness found");
const Real tiny=partial[4];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,dt);
ck(cellStatus[0]==0,"tiny positive dependent component blocked a valid update");
ck(partial[4]==tiny,"tiny carrier was modified instead of safe populated fallback");
ck(partial[0]==s.rho[0]-rawB && partial[2]==rawB,"largest populated carrier replacement differs from bounded target");
ck(std::abs(partial[0]+partial[2]+partial[4]-s.rho[0])<=s.gasSpecies.densityClosureTolerance*s.rho[0],"actual fallback publication closure");
}
''')
