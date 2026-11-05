"""Real steady-only parcel/gas transaction kernels, executed serially on CPU."""
import pytest
from test_mixture_state import compile_probe, ROOT
from test_mixture_transport import fixture


def paired_fixture():
    base=fixture().replace('1000,0,-1e6,1000,0,2e6','1000,.1,-1e7,1100,.15,-2e7')
    return base+r'''
#include "gpu/GpuCouplingMath.H"
ThreadIndex gridDim;
struct PairedState:State {
 int count=3; int*particleCountDevice=&count; int preBaseCellOffset[3]={0,2,3};
 int particleCapacity=4,cellParticleOffset[3]={0,2,3},sortedParticleIndex[3]={0,1,2};
 int pStatus[4]={1,1,1,0},pCellId[4]={0,0,1,0};
 double pm[4]={.2,.4,.3,.1},pd[4]={.001,.0002,.0005,.001};
 double pux[4]={-10,30,2,999},puy[4]={3,-4,1,999},puz[4]={1,2,-2,999};
 double puxOld[4]={},puyOld[4]={},puzOld[4]={},pTheta[4]={2,4,3,999},pT[4]={300,1200,900,999};
 double epsGPrev[2]={.8,.6},rhoSolid=2500,particleThermalRho=2500,particleCp=800,particleThermalCapacity=2e6;
 double dragParameter0=.8,dragParameter1=0,dragParameter2=0,dragParameter3=0;
 double gasPrOneThird=std::cbrt(.71),TpMin=100,TpMax=2000;
 double gravityX=.2,gravityY=-.3,gravityZ=.4;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
};
#include "PAIRED_HEADER"
void setup(PairedState&s){initialise(s);gridDim.x=1;s.gasMu=1.8e-5;s.gasPrClamped=.71;s.gasThermalConductivity=.06;s.V[0]=.7;s.V[1]=1.3;
for(int c=0;c<2;++c){s.rhoUx[c]=s.rho[c]*5;s.rhoUy[c]=s.rho[c]*-2;s.rhoUz[c]=s.rho[c]*1;s.rhoE[c]+=.5*s.rho[c]*(25+4+1);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}}
void closeValue(long double a,long double b,long double tol,const char*why){if(fabsl(a-b)>tol){std::fprintf(stderr,"%s: %.21Lg != %.21Lg difference %.21Lg\n",why,a,b,a-b);std::abort();}}
struct Budget {long double mass=0,x=0,y=0,z=0,energy=0;};
Budget budget(const PairedState&s){Budget b;
for(int c=0;c<2;++c){long double v=(long double)s.epsGPrev[c]*s.V[c];b.mass+=v*s.rho[c];b.x+=v*s.rhoUx[c];b.y+=v*s.rhoUy[c];b.z+=v*s.rhoUz[c];b.energy+=v*s.rhoE[c];}
for(int i=0;i<s.particleCapacity;++i)if(s.pStatus[i]==1){long double m=s.pm[i];b.mass+=m;b.x+=m*s.pux[i];b.y+=m*s.puy[i];b.z+=m*s.puz[i];b.energy+=m*(.5L*((long double)s.pux[i]*s.pux[i]+(long double)s.puy[i]*s.puy[i]+(long double)s.puz[i]*s.puz[i])+1.5L*s.pTheta[i]+800.L*s.pT[i]);}return b;}
void conserved(const Budget&a,const Budget&b){closeValue(a.mass,b.mass,1e-14L,"mass");closeValue(a.x,b.x,1e-12L,"momentum x");closeValue(a.y,b.y,1e-12L,"momentum y");closeValue(a.z,b.z,1e-12L,"momentum z");closeValue(a.energy,b.energy,5e-8L,"formation-inclusive total energy");}
'''.replace('PAIRED_HEADER',str(ROOT/'common/gasTransport/MixtureParticleCoupling.cuh'))


@pytest.mark.parametrize('model',['SchillerNaumannDeviceDrag','GidaspowErgunWenYuDeviceDrag'])
@pytest.mark.parametrize('csr',[True,False])
def test_actual_paired_exchange_conserves_heterogeneous_frozen_mixture(tmp_path,model,csr):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);auto before=budget(s);double species[4];std::memcpy(species,s.gasSpecies.rho,sizeof(species));
for(int c=0;c<2;++c){ck(s.rhoE[c]<0,"fixture not negative formation energy");threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::MODEL>(&s,.3,true,CSR);ck(s.rhoE[c]<0,"invented positive energy floor");ck(s.gasSpecies.cellStatus[c]==0,"mixture EOS source failure");}
conserved(before,budget(s));ck(std::memcmp(species,s.gasSpecies.rho,sizeof(species))==0,"exchange changed species");
for(int c=0;c<2;++c)closeValue(s.gasSpecies.rho[c]+s.gasSpecies.rho[2+c],s.rho[c],1e-14,"species sum density");
ck(s.puxOld[0]==-10&&s.puxOld[1]==30&&s.puxOld[2]==2,"post-collision snapshot");
ck(s.pTheta[0]<2&&s.pTheta[1]<4&&s.pTheta[2]<3,"unresolved drag missing");
ck(s.pT[0]>300&&s.pT[1]<1200,"two-way finite heat bath missing");
ck(s.pT[3]==999&&s.pux[3]==999&&s.pTheta[3]==999,"inactive parcel changed");
}
'''.replace('MODEL',model).replace('CSR','true' if csr else 'false'))


def test_paired_drag_target_uses_individual_rates_and_actual_theta_change(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);s.solveParticleTemperature=0;const double dt=.03;
double a[2],old[2],theta[2],w=0,sum=0;const double mass=s.epsGPrev[0]*s.V[0]*s.rho[0];
for(int i=0;i<2;++i){old[i]=s.pux[i];theta[i]=s.pTheta[i];double b=mixtureParticles::beta<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(s,0,i,s.epsGPrev[0]);a[i]=-expm1(-b*dt);w+=s.pm[i]*a[i];sum+=s.pm[i]*a[i]*old[i];}
const double target=(mass*s.Ux[0]+sum)/(mass+w);const auto before=budget(s);threadIdx.x=0;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,dt,true);
for(int i=0;i<2;++i){closeValue(s.pux[i],old[i]+a[i]*(target-old[i]),1e-13,"individual finite bath velocity");closeValue(s.pTheta[i],theta[i]*(1-a[i])*(1-a[i]),1e-13,"individual theta rate");}
closeValue(s.Ux[0],target,1e-13,"gas finite bath target");conserved(before,budget(s));}
''')


def test_heat_clamp_reverses_actual_parcel_change(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);s.TpMax=600;const auto before=budget(s);threadIdx.x=0;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,3.,false);
conserved(before,budget(s));ck(s.pT[1]==600,"fixture did not exercise clamp");ck(s.gasSpecies.cellStatus[0]==0,"clamped heat EOS");}
''')


def test_external_kick_never_reverses_gravity_or_pressure_to_gas(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);for(int c=0;c<2;++c){s.gradPx[c]=100;s.gradPy[c]=200;s.gradPz[c]=-300;}
threadIdx.x=0;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,.01,false);
const double gas[]={s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]},old=s.puxOld[0],vx=s.pux[0],vy=s.puy[0],vz=s.puz[0];
mixtureParticleExternalKickKernel(&s,.01);
closeValue(s.pux[0],vx+.01*(.2-100./2500),1e-14,"external x kick");closeValue(s.puy[0],vy+.01*(-.3-200./2500),1e-14,"external y kick");closeValue(s.puz[0],vz+.01*(.4+300./2500),1e-14,"external z kick");
const double after[]={s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]};ck(std::memcmp(gas,after,sizeof(gas))==0,"external work reversed as drag");ck(s.puxOld[0]==old,"kick erased pre-source snapshot");}
''')


def test_scan_and_external_kick_ignore_stale_active_slots_beyond_count(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);s.pStatus[3]=1;const double old[]={s.pux[3],s.puy[3],s.puz[3],s.pTheta[3],s.pT[3]};
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,.1,true,false);}
threadIdx.x=3;mixtureParticleExternalKickKernel(&s,.1);
const double now[]={s.pux[3],s.puy[3],s.puz[3],s.pTheta[3],s.pT[3]};ck(std::memcmp(old,now,sizeof(old))==0,"stale compacted slot consumed");}
''')


def test_zero_time_does_not_change_conserved_gas_or_parcel_state(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);const auto before=budget(s);
double gas[10],particles[15];for(int c=0;c<2;++c){gas[c*5]=s.rho[c];gas[c*5+1]=s.rhoUx[c];gas[c*5+2]=s.rhoUy[c];gas[c*5+3]=s.rhoUz[c];gas[c*5+4]=s.rhoE[c];}
for(int i=0;i<3;++i){particles[i*5]=s.pux[i];particles[i*5+1]=s.puy[i];particles[i*5+2]=s.puz[i];particles[i*5+3]=s.pTheta[i];particles[i*5+4]=s.pT[i];}
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,0.,true,false);}
for(int c=0;c<2;++c){ck(gas[c*5]==s.rho[c]&&gas[c*5+1]==s.rhoUx[c]&&gas[c*5+2]==s.rhoUy[c]&&gas[c*5+3]==s.rhoUz[c]&&gas[c*5+4]==s.rhoE[c],"zero time gas change");}
for(int i=0;i<3;++i){ck(particles[i*5]==s.pux[i]&&particles[i*5+1]==s.puy[i]&&particles[i*5+2]==s.puz[i]&&particles[i*5+3]==s.pTheta[i]&&particles[i*5+4]==s.pT[i],"zero time parcel change");}
conserved(before,budget(s));}
''')


def test_external_kick_grid_stride_covers_more_parcels_than_launch_width(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);double old[3];for(int i=0;i<3;++i)old[i]=s.pux[i];
gridDim.x=1;blockDim.x=1;threadIdx.x=0;mixtureParticleExternalKickKernel(&s,.1);
for(int i=0;i<3;++i)closeValue(s.pux[i],old[i]+.1*s.gravityX,1e-14,"grid stride external kick missed parcel");}
''')


def test_heat_exchange_remains_resolved_under_large_negative_formation_offset(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState a,b;setup(a);setup(b);auto*coeff=const_cast<Real*>(b.gasSpecies.thermo.coefficients);coeff[2]-=1e12;coeff[5]-=1e12;
for(int c=0;c<2;++c){b.rhoE[c]-=b.rho[c]*1e12;a.gasSpecies.thermoControls.relativeEnergyTolerance=b.gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;a.gasSpecies.thermoControls.relativeTemperatureTolerance=b.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;}
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&a,.3,false);mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&b,.3,false);
closeValue(a.Tgas[c],b.Tgas[c],2e-6,"large formation offset changed finite bath gas T");ck(b.rhoE[c]<-1e11,"formation offset floored");}
for(int i=0;i<3;++i)closeValue(a.pT[i],b.pT[i],2e-6,"large formation offset changed parcel heat");
}
''')


def test_drag_exchange_is_galilean_covariant_at_large_bulk_velocity(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState a,b;setup(a);setup(b);a.solveParticleTemperature=b.solveParticleTemperature=0;const double boost=1e5;
for(int c=0;c<2;++c){b.rhoE[c]+=boost*b.rhoUx[c]+.5*b.rho[c]*boost*boost;b.rhoUx[c]+=b.rho[c]*boost;b.Ux[c]+=boost;
a.gasSpecies.thermoControls.relativeEnergyTolerance=b.gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;a.gasSpecies.thermoControls.relativeTemperatureTolerance=b.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;}
for(int i=0;i<3;++i)b.pux[i]+=boost;
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&a,.03,true);mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&b,.03,true);closeValue(a.Ux[c],b.Ux[c]-boost,3e-11,"boosted gas velocity");closeValue(a.Tgas[c],b.Tgas[c],2e-6,"boosted gas internal temperature");}
for(int i=0;i<3;++i){closeValue(a.pux[i],b.pux[i]-boost,3e-11,"boosted parcel velocity");closeValue(a.pTheta[i],b.pTheta[i],1e-12,"boosted unresolved energy");}
}
''')


@pytest.mark.parametrize('injection_only',[False,True])
def test_split_directory_full_scan_includes_retained_base(tmp_path,injection_only):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState full,split;setup(full);setup(split);
// Split directory contains no retained base. Its CSR is either empty or only
// one injected parcel; caller selects the complete active-parcel traversal.
split.cellParticleOffset[0]=0;split.cellParticleOffset[1]=INJECT?1:0;split.cellParticleOffset[2]=INJECT?1:0;split.sortedParticleIndex[0]=1;
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&full,.03,true,true);mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&split,.03,true,false);}
for(int c=0;c<2;++c){closeValue(full.rhoUx[c],split.rhoUx[c],1e-14,"split base gas momentum missing");closeValue(full.rhoE[c],split.rhoE[c],1e-10,"split base gas energy missing");}
for(int i=0;i<3;++i){closeValue(full.pux[i],split.pux[i],1e-14,"split retained parcel skipped");closeValue(full.pT[i],split.pT[i],1e-12,"split retained heat skipped");}
}
'''.replace('INJECT','true' if injection_only else 'false'))



@pytest.mark.parametrize('injected',[False,True])
def test_indexed_split_directory_visits_base_and_injection_once(tmp_path,injected):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState full,split;setup(full);setup(split);
if(INJECT){
 full.pCellId[1]=split.pCellId[1]=1;full.pCellId[2]=split.pCellId[2]=0;
 full.sortedParticleIndex[0]=0;full.sortedParticleIndex[1]=2;full.sortedParticleIndex[2]=1;
 split.preBaseCellOffset[0]=0;split.preBaseCellOffset[1]=1;split.preBaseCellOffset[2]=2;
 split.cellParticleOffset[0]=0;split.cellParticleOffset[1]=1;split.cellParticleOffset[2]=1;split.sortedParticleIndex[0]=2;
}else{split.cellParticleOffset[0]=split.cellParticleOffset[1]=split.cellParticleOffset[2]=0;}
const auto before=budget(split);
for(int c=0;c<2;++c){threadIdx.x=c;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&full,.03,true,true,false);mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&split,.03,true,true,true);}
conserved(before,budget(split));
for(int c=0;c<2;++c){closeValue(full.rhoUx[c],split.rhoUx[c],1e-14,"indexed split gas momentum");closeValue(full.rhoE[c],split.rhoE[c],1e-10,"indexed split gas energy");}
for(int i=0;i<3;++i){closeValue(full.pux[i],split.pux[i],1e-14,"indexed split parcel once");closeValue(full.pTheta[i],split.pTheta[i],1e-14,"indexed split theta once");closeValue(full.pT[i],split.pT[i],1e-12,"indexed split heat once");}
}
'''.replace('INJECT','true' if injected else 'false'))


@pytest.mark.parametrize('directory',['scan','csr','split'])
def test_heat_failure_publishes_current_step_fatal(tmp_path,directory):
    compile_probe(tmp_path,paired_fixture()+r"""
#include "DiagnosticsHost.H"
int main(){PairedState s;setup(s);
auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
coeff[0]=coeff[3]=1000;coeff[1]=coeff[2]=coeff[4]=coeff[5]=0;
s.count=1;s.pStatus[1]=s.pStatus[2]=0;
s.cellParticleOffset[1]=s.cellParticleOffset[2]=1;
s.preBaseCellOffset[1]=s.preBaseCellOffset[2]=1;
SPLIT_OFFSETS
s.pm[0]=1;s.pd[0]=1e-6;s.pT[0]=10000;s.TpMax=10000;
s.pux[0]=s.puy[0]=s.puz[0]=0;s.epsGPrev[0]=1-s.pm[0]/(s.rhoSolid*s.V[0]);
s.rhoUx[0]=s.rhoUy[0]=s.rhoUz[0]=s.Ux[0]=s.Uy[0]=s.Uz[0]=0;
Real species[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
s.rhoE[0]=ugkwp::mixtureEnergy(species,Real(1000),s.gasSpecies.thermo);s.Tgas[0]=1000;
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;s.gasSpecies.thermo.diagnostics=&diagnostics;
const double oldEnergy=s.rhoE[0];threadIdx.x=0;
mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,2e-5,false,CSR,SPLIT);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::ThermoStatus::TemperatureOutOfRange),"expected invalid gas root");
ck(s.rhoE[0]==oldEnergy && s.pT[0]==10000,"failure partially published heat");
recoverGasPrimitivesKernel(&s);
const auto snapshot=ugkp::diagnostics::takeSnapshot(diagnostics,{},"paired-heat-test");
ck(snapshot.hasFatal,"current completion lacks fatal without a subsequent flux");
const auto& fatal=diagnostics.firstFatal;
ck(fatal.code==ugkp::diagnostics::Code::InvalidPhysicalState,"wrong failure code");
ck(fatal.entityKind==ugkp::diagnostics::EntityKind::Cell && fatal.entityIndex==0,"lost failing cell");
ck(fatal.metadata==unsigned(ugkwp::ThermoStatus::TemperatureOutOfRange),"lost thermo status");
}
""".replace('SPLIT_OFFSETS','s.cellParticleOffset[1]=s.cellParticleOffset[2]=0;' if directory=='split' else '')
       .replace('CSR','false' if directory=='scan' else 'true').replace('SPLIT','true' if directory=='split' else 'false'))
