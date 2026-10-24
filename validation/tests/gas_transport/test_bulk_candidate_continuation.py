"""Production-header finite publication; required arithmetic remains observable."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

CASES = {
'divergence_closure':r'''
s.gasSpecies.rho[0]=Real(.91);s.gasSpecies.rho[2]=Real(.1);
s.gasPhiRho[0]=Real(.2);s.gasSpecies.flux[0]=Real(.3);s.gasSpecies.flux[3]=Real(-.02);
s.gasPhiRhoUx[0]=2;s.gasPhiRhoUy[0]=-3;s.gasPhiRhoUz[0]=4;s.gasPhiRhoE[0]=5;
const Real E=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0,"finite divergence rejected");
ck(s.rho[0]==Real(.9)&&s.rhoUx[0]==-1&&s.rhoUy[0]==Real(1.5)&&s.rhoUz[0]==-2&&s.rhoE[0]==E-Real(2.5),"bulk candidate incomplete");
ck(s.gasSpecies.rho[0]==Real(.91)-Real(.5)*Real(.3)&&s.gasSpecies.rho[2]==Real(.1)-Real(.5)*Real(-.02),"raw species candidate lost");''',
'divergence_signed':r'''
s.gasPhiRho[0]=4;s.gasSpecies.flux[0]=3;s.gasSpecies.flux[3]=1;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==-1,"finite signed inventory rejected");
ck(s.gasSpecies.rho[0]==Real(.9)-Real(1.5)&&s.gasSpecies.rho[2]==Real(.1)-Real(.5),"signed candidate incomplete");''',
'rk_closure':r'''
threadIdx.x=0;saveGasConservativeStateKernel(&s);s.gasSpecies.rho[0]=Real(-.1);s.gasSpecies.rho[2]=Real(.4);s.rho[0]=Real(.2);
s.rhoUx[0]=4;s.rhoUy[0]=8;s.rhoUz[0]=12;s.rhoE[0]=123;
const Real expected[7]={Real(.75)+Real(.25)*Real(.2),1,2,3,Real(.75)*s.rhoENext[0]+Real(.25)*s.rhoE[0],Real(.75)*Real(.9)+Real(.25)*Real(-.1),Real(.75)*Real(.1)+Real(.25)*Real(.4)};
blendGasConservativeStateKernel(&s,.75,.25);
Real out[7]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
ck(s.gasSpecies.cellStatus[0]==0,"RK finite defect rejected");for(int k=0;k<7;++k)ck(out[k]==expected[k],"RK complete raw candidate differs");''',
'periodic_defect':r'''
s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;
s.gasPhiRho[0]=2;s.gasPhiRho[1]=0;s.gasSpecies.flux[0]=4;s.gasSpecies.flux[1]=-2;s.gasSpecies.flux[2]=3;s.gasSpecies.flux[3]=-1;
threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
ck(s.gasSpecies.faceStatus[0]==0&&s.gasSpecies.faceStatus[1]==0,"finite periodic defect rejected");
ck(s.gasPhiRho[0]==1&&s.gasPhiRho[1]==-1&&s.gasSpecies.flux[0]==3&&s.gasSpecies.flux[1]==-3&&s.gasSpecies.flux[2]==2&&s.gasSpecies.flux[3]==-2,"periodic raw average incomplete");''',
'recovery_closure':r'''
s.gasSpecies.rho[0]=Real(-.01);s.gasSpecies.rho[2]=Real(1.02);Real partial[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
s.rhoE[0]=ugkwp::mixtureEnergy(partial,Real(700),s.gasSpecies.thermo);s.Ux[0]=s.Uy[0]=s.Uz[0]=s.p[0]=s.Tgas[0]=-77;
s.gasSpecies.soundSpeed[0]=s.gasSpecies.heatCapacity[0]=s.gasSpecies.gasConstant[0]=-77;threadIdx.x=0;recoverGasPrimitivesKernel(&s);
ck(s.gasSpecies.cellStatus[0]==0,"finite recovery defect rejected");ck(s.Ux[0]==0&&s.Uy[0]==0&&s.Uz[0]==0&&std::abs(s.Tgas[0]-700)<Real(.01),"primitive output incomplete");
ck(s.p[0]>0&&s.gasSpecies.soundSpeed[0]>0&&s.gasSpecies.heatCapacity[0]>0&&s.gasSpecies.gasConstant[0]>0,"EOS caches incomplete");''',
'reconstruction_exhausted':r'''
s.gasReconstruction=1;s.rhoMin=2;GasPrimDevice out{-77,-77,-77,-77,-77,-77};Real Y[2]={-77,-77};
ck(reconstructGasMixtureFace(s,0,0,out,Y),"finite final MUSCL candidate refused");
ck(out.rho==s.rho[0]&&out.ux==s.Ux[0]&&out.uy==s.Uy[0]&&out.uz==s.Uz[0]&&out.p==s.p[0]&&out.T>0,"MUSCL candidate incomplete");
ck(Y[0]==s.gasSpecies.rho[0]/s.rho[0]&&Y[1]==1-Y[0],"MUSCL composition incomplete");''',
'boundary_redundant':r'''
s.riemannBoundaryRhoFix[1]=s.riemannBoundaryPFix[1]=s.riemannBoundaryTFix[1]=1;s.riemannBoundaryRho[1]=100;s.riemannBoundaryP[1]=101325;s.riemannBoundaryT[1]=500;
const auto out=riemannBoundaryState(s,1,gasCellPrimitive(s,0));Real y[2]={Real(.9),Real(.1)};
ck(s.gasSpecies.faceStatus[1]==0,"redundant finite boundary refused");ck(out.rho==Real(101325)/(ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo)*Real(500))&&out.p==101325&&out.T==500,"boundary precedence changed");''',
'geometry_certificate':r'''
Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={Real(.1),0,0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;
Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,.002,a,b)&&a==1&&b==2,"finite geometry certificate blocked volume output");
s.magSf[0]=3;ugkwp::GasFaceFrame<Real> frame;ck(ugkwp::gasGeometryFaceFrame(s,0,.002,frame)&&frame.area==1&&frame.meshVolumeRate==100,"frame output blocked by finite certificate");
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.002);ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==Real(.5),"actual volume ratio not published");''',
'riemann_below_floor':r'''
ugkpriemann::Primitive p{Real(1e-15),2,-3,4,Real(1e-15)};
auto f=ugkpriemann::fluxUnitArea(p,p,Real(1),Real(0),Real(0),Real(1.4),ugkpriemann::Scheme::RusanovTadmor);
ck(f.valid,"finite below-floor Riemann state refused");auto q=ugkpriemann::conservative(p,Real(1.4));
ck(f.flux[0]==2*p.rho&&f.flux[1]==2*q.q[1]+p.p&&f.flux[2]==2*q.q[2]&&f.flux[3]==2*q.q[3]&&f.flux[4]==2*(q.q[4]+p.p),"Riemann complete flux differs");''',
}

@pytest.mark.parametrize('scenario',CASES)
def test_complete_bulk_candidates(tmp_path,scenario):
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);'+CASES[scenario]+'\n}')

@pytest.mark.parametrize('strict',[False,True])
def test_required_nan_flux_finishes_terminal_but_retains_strict_retry(tmp_path,strict):
    compile_probe(tmp_path,fixture()+r"""
int main(){State s;initialise(s);s.gasSpecies.flux[0]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRho[1]=-.2;s.gasPhiRhoE[1]=-7;s.gasSpecies.flux[1]=-.18;s.gasSpecies.flux[4]=-.02;s.gasPhiRhoUy[1]=-3;
const Real before[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.1,STRICT);
ck(s.gasSpecies.cellStatus[0]!=0,"required NaN certificate lost");
if(STRICT){const Real after[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};ck(std::memcmp(before,after,sizeof before)==0,"strict NaN retry published state");}
else{ck(s.rho[0]==before[0]+Real(.1)*Real(.2)&&s.rhoUx[0]==0&&s.rhoUy[0]==Real(.1)*Real(3)&&s.rhoUz[0]==0&&s.rhoE[0]==before[4]+Real(.1)*Real(7),"terminal NaN bulk formula incomplete");ck(std::isnan(s.gasSpecies.rho[0])&&s.gasSpecies.rho[2]==before[6]+Real(.1)*Real(.02),"terminal NaN species formula incomplete");}}
""".replace('STRICT','true' if strict else 'false'))

@pytest.mark.parametrize('case',['source_cap','volume_residual','omega_sign','sst_flag'])
def test_sst_finite_diagnostics_do_not_block(tmp_path,case):
    from test_legacy_ale_transport import fixture as legacy_fixture
    from test_boundary_layer_operators import SETUP
    bodies={
        'source_cap':r'''s.k[0]=Real(.001);s.rhoK[0]=Real(.002);s.omega[0]=1000;Real source,unused;sstSourcesForCell(s,0,Real(0),source,unused);const Real expected=clampMin(s.rhoK[0]+Real(.1)*source,s.rho[0]*s.sstKMin);threadIdx.x=0;applySstFluxAndSourceKernel(&s,.1);ck(s.gasSpecies.cellStatus[0]==0,"finite source activity refused");near(s.rhoK[0],expected,"source candidate not published");''',
        'volume_residual':r'''s.gasBoundaryLayer.sst[0].volume=3;Real source,unused;sstSourcesForCell(s,0,Real(0),source,unused);Real expected=clampMin(s.rhoK[0]+Real(.01)*source,s.rho[0]*s.sstKMin);threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);ck(s.gasSpecies.cellStatus[0]==0,"wall volume certificate refused");near(s.rhoK[0],expected,"wall volume changed actual source");''',
        'omega_sign':r'''s.gasBoundaryLayer.sst[0].ownerOmega=-1;threadIdx.x=0;applySstWallFunctionStateKernel(&s);ck(s.gasSpecies.cellStatus[0]==0,"finite omega rejected before existing floor");near(s.omega[0],s.sstOmegaMin,"existing omega floor not used");''',
        'sst_flag':r'''s.sstConfigured=0;threadIdx.x=0;computeGasEddyViscosityKernel(&s);ck(std::isfinite(s.nut[0])&&s.nut[0]>=0,"available SST arrays not computed");'''}
    compile_probe(tmp_path,legacy_fixture()+SETUP+'\nint main(){State s;initialise(s);wall(s);'+bodies[case]+'}')

def test_volume_source_publishes_finite_below_floor(tmp_path):
    from test_authoritative_energy_sources import source_fixture
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);s.rhoE[0]=-10;s.cellPlaneCount[0]=0;threadIdx.x=0;applyGasVolumeFractionSourceKernel(&s,.25);ck(s.rhoE[0]==-10,"finite volume source energy rejected");}
''')

def test_prescribed_wall_flux_remains_complete_without_refusal(tmp_path):
    from test_boundary_layer_operators import SETUP
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasPhiRho[1]=0;s.gasPhiRhoE[1]=17;s.gasSpecies.flux[1]=.5;s.gasSpecies.flux[4]=-.5;s.gasSpecies.positivityScale[0]=.2;threadIdx.x=1;applyGasFluxPositivityScaleKernel(&s);ck(s.gasSpecies.faceStatus[1]==0,"finite prescribed wall flux refused");ck(s.gasSpecies.flux[1]==.5&&s.gasSpecies.flux[4]==-.5&&s.gasPhiRhoE[1]==17,"prescribed wall exchange changed");}
''')

@pytest.mark.parametrize('case',['rk','divergence','periodic'])
def test_numeric_candidates_publish_actual_terminal_fields(tmp_path,case):
    bodies={
        'rk':r"""threadIdx.x=0;saveGasConservativeStateKernel(&s);s.rhoUx[0]=s.rhoUxNext[0]=std::numeric_limits<Real>::max();const Real energy=s.rhoE[0];blendGasConservativeStateKernel(&s,1.,1.);ck(s.gasSpecies.cellStatus[0]==0,"original normal-return static RK status changed");ck(s.rho[0]==2&&std::isinf(s.rhoUx[0])&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==2*energy&&s.gasSpecies.rho[0]==2*Real(.9)&&s.gasSpecies.rho[2]==2*Real(.1),"actual RK candidate fields incomplete");""",
        'divergence':r"""s.rhoUx[0]=std::numeric_limits<Real>::max();s.gasPhiRhoUx[0]=-std::numeric_limits<Real>::max();const Real energy=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,1.);ck(s.gasSpecies.cellStatus[0]==0,"original normal-return static divergence status changed");ck(s.rho[0]==1&&std::isinf(s.rhoUx[0])&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==energy&&s.gasSpecies.rho[0]==Real(.9)&&s.gasSpecies.rho[2]==Real(.1),"actual divergence candidate fields incomplete");""",
        'periodic':r"""s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;s.gasSpecies.flux[0]=std::numeric_limits<Real>::quiet_NaN();threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);ck(s.gasSpecies.faceStatus[0]!=0&&s.gasSpecies.faceStatus[1]!=0,"periodic numerical certificate lost");ck(std::isnan(s.gasSpecies.flux[0])&&std::isnan(s.gasSpecies.flux[1])&&s.gasSpecies.flux[2]==0&&s.gasSpecies.flux[3]==0,"actual two-face periodic species average incomplete");for(auto field:{s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE})ck(field[0]==0&&field[1]==0,"actual two-face periodic bulk average incomplete");"""}
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);'+bodies[case]+'}')

@pytest.mark.parametrize('both_nan',[False,True])
@pytest.mark.parametrize('strict',[False,True])
def test_courant_wave_nonfinite_is_reported_and_strict_fallback_retained(tmp_path,both_nan,strict):
    compile_probe(tmp_path,fixture()+r"""
int main(){State s;initialise(s);s.gasSpecies.soundSpeed[0]=std::numeric_limits<Real>::quiet_NaN();if(BOTH)s.gasSpecies.soundSpeed[1]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoE[0]=-77;
const Real expected=s.gasSpecies.soundSpeed[1]*clampMin(s.magSf[0],OfSmall);threadIdx.x=0;computeGasCourantFieldKernel(&s,.01,STRICT);if(BOTH){ck(s.gasSpecies.faceStatus[0]!=0,"nonfinite spectral status was hidden");ck(STRICT?s.gasPhiRhoE[0]==OfGreat:std::isnan(s.gasPhiRhoE[0]),"strict fallback or terminal evaluated spectral diagnostic lost");}else{ck(s.gasPhiRhoE[0]==expected&&s.gasSpecies.faceStatus[0]==0,"finite fmax spectral result changed");}}
""".replace('BOTH','true' if both_nan else 'false').replace('STRICT','true' if strict else 'false'))

def test_large_courant_and_subsmall_time_keep_actual_values(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasPhiRhoE[0]=s.gasPhiRhoE[1]=OfGreat;threadIdx.x=0;computeGasConvectiveCourantByCellKernel(&s,4.);ck(s.gasFluxPositivityScale[0]==Real(4)*OfGreat,"large finite Courant refused");
computeGasConvectiveCourantByCellKernel(&s,1e-30);ck(s.gasFluxPositivityScale[0]>0&&s.gasSpecies.cellStatus[0]==0,"subSMALL step refused");}
''')

def test_sst_source_needs_only_physical_storage(tmp_path):
    from test_legacy_ale_transport import fixture as legacy_fixture
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;threadIdx.x=0;Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);const Real k=s.rhoK[0],o=s.rhoOmega[0];applySstFluxAndSourceKernel(&s,.001);ck(s.gasSpecies.cellStatus[0]==0,"physical source status");near(s.rhoK[0],k+Real(.001)*sk,"actual k source missing");near(s.rhoOmega[0],o+Real(.001)*so,"actual omega source missing");}
''')

def test_wall_subgrid_finite_composition_defect_keeps_real_heat(tmp_path):
    from test_mixture_wall_function import wall_fixture
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){State s;wall(s);s.gasSpecies.rho[0]+=Real(.01);Real mu=-77,k=-77,q=-77;int active=0;
gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active);
ck(s.gasSpecies.faceStatus[1]==0,"finite wall composition defect refused");ck(std::isfinite(mu)&&std::isfinite(k)&&std::isfinite(q)&&active&&q!=-77&&q!=0,"actual wall transport incomplete");}
''')

@pytest.mark.parametrize('negative',['mu','cp'])
def test_jayatilleke_evaluable_signed_transport_publishes_heat(tmp_path,negative):
    compile_probe(tmp_path,fixture()+r'''
int main(){Real mu=Real(.001),cp=1000;NEGATIVE=-NEGATIVE;
const Real rho=1,Pr=Real(.7),Prt=Real(.9),Cmu=Real(.09),kappa=Real(.41),E=Real(9.8),P=Real(2),yt=12,k=Real(.1),y=Real(.5),du=2,grad=10;
auto v=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,Pr,Prt,Cmu,kappa,E,P,yt,k,y,du,Real(0),grad);
const Real star=sqrt(sqrt(Cmu))*sqrt(k),yp=star*y*rho/mu;const bool viscous=yp<yt;
const Real tp=viscous?Pr*yp:Prt*(log(E*yp)/kappa+P),uc=viscous?Real(0):star/kappa*log(E*yt);
const Real C=Real(.5)*rho*star*(viscous?Pr*du*du:Prt*du*du+(Pr-Prt)*uc*uc);
const Real expectedK=fmax(mu*cp/Pr,cp*rho*star*y/tp),expectedQ=-expectedK*grad+C/tp;
ck(v.valid&&v.conductivity==expectedK&&v.heatFlux==expectedQ,"finite signed Jayatilleke candidate refused or changed");}
'''.replace('NEGATIVE',negative))

def test_unused_sweep_sum_overflow_cannot_refuse_finite_geometry_outputs(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={Real(.75)*std::numeric_limits<Real>::max(),Real(.75)*std::numeric_limits<Real>::max(),0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=std::numeric_limits<Real>::max();
Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,double(s.gasGeometry.interval),a,b)&&a==1&&b==1,"unused sweep certificate overflow blocked actual volumes");ugkwp::GasFaceFrame<Real> f;ck(ugkwp::gasGeometryFaceFrame(s,0,double(s.gasGeometry.interval),f)&&std::isfinite(f.wx),"actual finite frame lost");}
''')

def test_zero_volume_is_only_an_error_for_an_actual_division(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){Real q[5]={1,2,3,4,5},out[5]={-77,-77,-77,-77,-77};
ck(ugkwp::convertGasConservativeStorage<Real,0>(q,Real(0),ugkwp::GasConservativeStorage::Density,ugkwp::GasConservativeStorage::Integral,out),"zero-volume multiplication refused");for(int k=0;k<5;++k)ck(out[k]==0,"zero-volume integral incomplete");
ck(ugkwp::convertGasConservativeStorage<Real,0>(q,Real(0),ugkwp::GasConservativeStorage::Integral,ugkwp::GasConservativeStorage::Integral,out),"unused volume refused identity conversion");for(int k=0;k<5;++k)ck(out[k]==q[k],"identity output incomplete");
ck(!ugkwp::convertGasConservativeStorage<Real,0>(q,Real(0),ugkwp::GasConservativeStorage::Integral,ugkwp::GasConservativeStorage::Density,out),"zero-volume division reported success");}
''')

def test_exhausted_reconstruction_retains_its_last_complete_finite_candidate(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);auto* meta=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);meta[1].molarMass=Real(2)*meta[0].molarMass;
s.gasSpecies.rho[0]=-1;s.gasSpecies.rho[2]=2;s.gasReconstruction=1;s.gasSpecies.limiter[0]=1;s.gasSpecies.gradX[0]=1;
GasPrimDevice out{-77,-77,-77,-77,-77,-77};Real Y[2]={-77,-77};
ck(reconstructGasMixtureFace(s,0,0,out,Y),"last complete finite reconstruction lost to undefined centre EOS");
ck(Y[0]>Real(-1)&&Y[0]<0&&Y[1]==1-Y[0],"fallback did not retain an evaluated slope candidate");
const Real R=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);ck(std::isfinite(out.T)&&out.T==out.p/(out.rho*R)&&out.rho==s.rho[0]&&out.ux==s.Ux[0]&&out.uy==s.Uy[0]&&out.uz==s.Uz[0],"fallback fields are incomplete or have wrong lineage");}
''')

@pytest.mark.parametrize('field',['gasPhiRhoUx','gasSpecies.flux'])
@pytest.mark.parametrize('strict',[False,True])
def test_periodic_numeric_average_retains_strict_prefix_and_terminal_writes(tmp_path,field,strict):
    body=f"""s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;
s.{field}[0]=std::numeric_limits<Real>::max();s.{field}[1]=-std::numeric_limits<Real>::max();
threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s,{'true' if strict else 'false'});
for(auto bulk:{{s.gasPhiRho,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE}})ck(bulk[0]==0&&bulk[1]==0,"periodic unchanged channels lost their actual average");
"""
    if strict and field=='gasPhiRhoUx':
        body+='ck(s.gasSpecies.faceStatus[0]==0&&s.gasSpecies.faceStatus[1]==0,"original strict bulk normal-return status changed");'
    else:
        body+='ck(s.gasSpecies.faceStatus[0]!=0&&s.gasSpecies.faceStatus[1]!=0,"numeric periodic certificate lost");'
    if field=='gasPhiRhoUx':
        body+='ck(std::isinf(s.gasPhiRhoUx[0])&&s.gasPhiRhoUx[0]>0&&std::isinf(s.gasPhiRhoUx[1])&&s.gasPhiRhoUx[1]<0,"actual periodic bulk overflow not published");'
    elif strict:
        body+='ck(s.gasSpecies.flux[0]==std::numeric_limits<Real>::max()&&s.gasSpecies.flux[1]==-std::numeric_limits<Real>::max(),"strict failed species prefix unexpectedly published");'
    else:
        body+='ck(std::isinf(s.gasSpecies.flux[0])&&s.gasSpecies.flux[0]>0&&std::isinf(s.gasSpecies.flux[1])&&s.gasSpecies.flux[1]<0&&s.gasSpecies.flux[2]==0&&s.gasSpecies.flux[3]==0,"actual terminal periodic species overflow incomplete");'
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);'+body+'}')
