"""B2 complete actual gas face producers; serial CPU bodies, not GPU execution."""
from pathlib import Path
import os
import subprocess
import pytest
from test_mixture_transport import fixture
from test_mixture_state import ROOT, HERE

PREAMBLE = (HERE / 'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
MAIN = r'''
#include <limits>
#include <cfenv>
bool same(Real a,Real b){return (std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0;}
void field(Real a,Real b,const char*m){ck(same(a,b),m);}
int main(int argc,char**argv){ck(argc==2,"scenario");State s;initialise(s);const char*test=argv[1];const Real inf=std::numeric_limits<Real>::infinity(),nan=std::numeric_limits<Real>::quiet_NaN();
 if(!std::strcmp(test,"frame")){
  ugkwp::GasFaceFrame<Real> f;f.nx=-11;f.ny=-12;f.nz=-13;f.area=-14;f.wx=-15;f.wy=-16;f.wz=-17;f.meshVolumeRate=-18;
  ck(!ugkwp::makeGasFaceFrame(Real(1),Real(0),Real(0),Real(2),Real(0),f),"old frame certificate");
  field(f.nx,1,"nx");field(f.ny,0,"ny");field(f.nz,0,"nz");field(f.area,1,"area");field(f.meshVolumeRate,inf,"rate");field(f.wx,inf,"wx");ck(std::isnan(f.wy)&&std::isnan(f.wz),"normal speed multiplication missing");
  std::feclearexcept(FE_ALL_EXCEPT);ck(!ugkwp::makeGasFaceFrame(Real(0),Real(0),Real(0),Real(0),Real(1),f),"zero-area certificate");ck(std::isnan(f.nx)&&std::isnan(f.area)&&std::fetestexcept(FE_INVALID),"zero-area actual division missing");
 }else if(!std::strcmp(test,"volume")){
  Real oldV[2]={inf,1},newV[2]={nan,2};s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;Real a=-11,b=-12;
  ck(!ugkwp::gasGeometryCellVolumes(s,0,.1,a,b),"old volume certificate");field(a,inf,"old actual volume");field(b,nan,"new actual volume");s.gasGeometry.oldVolume=nullptr;a=-11;b=-12;ck(!ugkwp::gasGeometryCellVolumes(s,0,.1,a,b)&&a==-11&&b==-12,"missing backing");
 }else if(!std::strcmp(test,"shift_restore")){
  ugkwp::GasFaceFrame<Real> f;ck(ugkwp::makeGasFaceFrame(Real(1),Real(0),Real(0),Real(1),Real(1),f),"frame");
  ugkpriemann::Primitive p{2,inf,3,4,5},out{-11,-12,-13,-14,-15};ugkpriemann::Conservative q{{2,inf,6,8,99}},qo{{-21,-22,-23,-24,-25}};
  ck(!ugkwp::shiftGasStateToFaceFrame(p,q,f,out,qo),"shift certificate");field(out.rho,2,"shift rho");field(out.ux,p.ux-f.wx,"shift ux");field(out.uy,p.uy-f.wy,"shift uy");field(out.uz,p.uz-f.wz,"shift uz");field(out.p,5,"shift p");field(qo.q[0],2,"shift mass");for(int k=1;k<4;++k)field(qo.q[k],q.q[k]-q.q[0]*(&f.wx)[k-1],"shift momentum");field(qo.q[4],q.q[4]+(.5*q.q[0]*f.wx-q.q[1])*f.wx+(.5*q.q[0]*f.wy-q.q[2])*f.wy+(.5*q.q[0]*f.wz-q.q[3])*f.wz,"shift energy");
  ugkpriemann::FluxResult moving{{inf,2,3,4,5},7,ugkpriemann::Scheme::HllKurganov,false,true},rest{{-31,-32,-33,-34,-35},-36,ugkpriemann::Scheme::Roe,true,false};
  ck(!ugkwp::restoreGasFluxFromFaceFrame(moving,f,rest),"restore certificate");field(rest.flux[0],inf,"restore mass");for(int k=1;k<4;++k)field(rest.flux[k],moving.flux[k]+(&f.wx)[k-1]*moving.flux[0],"restore momentum");field(rest.flux[4],moving.flux[4]+f.wx*moving.flux[1]+f.wy*moving.flux[2]+f.wz*moving.flux[3]+.5*(f.wx*f.wx+f.wy*f.wy+f.wz*f.wz)*moving.flux[0],"restore energy");ck(rest.maxSignalSpeed==7&&rest.evaluatedScheme==moving.evaluatedScheme&&rest.usedFallback&&!rest.valid,"restore metadata");
 }else if(!std::strcmp(test,"boundary")){
  s.riemannBoundaryKind[1]=0;s.riemannBoundaryPFix[1]=s.riemannBoundaryTFix[1]=1;s.riemannBoundaryP[1]=101325;s.riemannBoundaryT[1]=0;s.riemannBoundaryUFix[1]=1;s.riemannBoundaryUx[1]=2;s.riemannBoundaryUy[1]=3;s.riemannBoundaryUz[1]=inf;
  const auto out=riemannBoundaryState(s,1,gasCellPrimitive(s,0));ck(s.gasSpecies.faceStatus[1]!=0,"boundary status");ck(std::isinf(out.rho)&&out.p==101325&&out.T==0&&out.ux==2&&out.uy==3&&std::isinf(out.uz),"boundary EOS/velocity incomplete");
 }else if(!std::strcmp(test,"boundary_total")){
  s.riemannBoundaryKind[1]=0;s.riemannBoundaryUFix[1]=3;s.riemannBoundaryP[1]=101325;s.riemannBoundaryT[1]=900;
  const auto owner=gasCellPrimitive(s,0);Real y[2]={Real(.9),Real(.1)},R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo),cv=ugkwp::mixtureHeatCapacity(y,owner.T,s.gasSpecies.thermo);
  const auto expected=ugkpboundary::totalConditionInletState(ugkpboundary::Primitive{owner.rho,owner.ux,owner.uy,owner.uz,owner.p,owner.T},Real(-1),Real(0),Real(0),Real(101325),Real(900),(cv+R)/cv,R,s.rhoMin,s.TgasMin);
  const auto out=riemannBoundaryState(s,1,owner);field(out.rho,expected.rho,"total density");field(out.ux,expected.ux,"total velocity");field(out.p,expected.p,"total pressure");field(out.T,expected.T,"total temperature");
 }else if(!std::strcmp(test,"legacy_flux")){
  ugkpriemann::Primitive p{0,2,3,4,5};auto result=ugkpriemann::fluxUnitArea(p,p,Real(1),Real(0),Real(0),Real(1.4),ugkpriemann::Scheme::RusanovTadmor);
  auto expected=ugkpriemann::rusanovTadmorFluxUnitNormal(p,p,Real(1),Real(0),Real(0),Real(1.4),false);for(int k=0;k<5;++k)field(result.flux[k],expected.flux[k],"legacy selected actual flux");field(result.maxSignalSpeed,expected.maxSignalSpeed,"legacy signal");
  std::feclearexcept(FE_ALL_EXCEPT);result=ugkpriemann::fluxUnitArea(ugkpriemann::Primitive{1,2,3,4,5},ugkpriemann::Primitive{1,2,3,4,5},Real(0),Real(0),Real(0),Real(1.4),ugkpriemann::Scheme::RusanovTadmor);ck(std::fetestexcept(FE_INVALID)&&std::isnan(result.flux[0]),"actual zero normal arithmetic");
 }else if(!std::strcmp(test,"legacy_moving_nan")){
  ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients> plain=static_cast<const ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>&>(s);Real sweep[3]={0,0,0};plain.gasGeometry.enabled=true;plain.gasGeometry.faceSweptVolume=sweep;plain.gasGeometry.interval=0;Real out[5]={-11,-12,-13,-14,-15};
  ck(!computeRiemannGasFaceFluxDevice<false>(plain,0,out[0],out[1],out[2],out[3],out[4]),"legacy retains invalid certificate");for(int k=0;k<5;++k)ck(std::isnan(out[k]),"legacy moving actual flux tail");
 }else if(!std::strcmp(test,"invalid_selector")){
  s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.gasFluxScheme=999;Real out[5]={-11,-12,-13,-14,-15};ck(!computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"selector retains status");ck(out[0]==0&&out[1]==s.p[0]&&out[2]==0&&out[3]==0&&out[4]==0,"actual schemeFromCreateCode selected fallback math");
 }else if(!std::strcmp(test,"nonfinite_coefficients")){
  State expected;initialise(expected);s.gasMu=nan;expected.gasMu=0;s.gasThermalConductivity=expected.gasThermalConductivity=0;s.Ux[0]=expected.Ux[0]=0;s.Ux[1]=expected.Ux[1]=2;Real out[5]={-11,-12,-13,-14,-15},ref[5]={-21,-22,-23,-24,-25};ck(computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"original nonfinite transport skip success");ck(computeRiemannGasFaceFluxDevice<false>(expected,0,ref[0],ref[1],ref[2],ref[3],ref[4]),"zero transport reference");for(int k=0;k<5;++k)field(out[k],ref[k],"original transport skip bytes");ck(s.gasSpecies.faceStatus[0]==0,"original transport skip status");
 }else if(!std::strcmp(test,"muscl")){
  s.gasReconstruction=1;s.gradUxX[0]=nan;s.gasGradientLimiterUx[0]=1;GasPrimDevice out{-11,-12,-13,-14,-15,-16};Real Y[2]={-21,-22};
  ck(!reconstructGasMixtureFace(s,0,1,out,Y),"old complete certificate");ck(out.rho==s.rho[0]&&out.p==s.p[0]&&std::isnan(out.ux),"MUSCL final evaluated candidate");field(Y[0],Real(.9),"MUSCL Y0");field(Y[1],Real(1)-Real(.9),"MUSCL Y1");Real R=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);field(out.T,out.p/(out.rho*R),"MUSCL actual EOS");
 }else if(!std::strcmp(test,"diffusion")){
  Real Y[2]={.25,.75},grad[2]={1,-1},D[2]={inf,.03},out[2]={-11,-12},energy=-13;
  ck(!ugkwp::correctedSpeciesDiffusion(Y,grad,Real(2),D,Real(.004),Real(.8),Real(700),s.gasSpecies.thermo,out,energy),"diffusion certificate");Real raw[2],sum=0;for(int k=0;k<2;++k){raw[k]=-Real(2)*(D[k]+Real(.004)/Real(.8))*grad[k];sum+=raw[k];}Real expected=raw[0]-Y[0]*sum;field(out[0],expected,"diffusion0");field(out[1],-expected,"diffusion closure");field(energy,ugkwp::speciesH(0,Real(700),s.gasSpecies.thermo)*expected+ugkwp::speciesH(1,Real(700),s.gasSpecies.thermo)*(-expected),"diffusion energy");
 }else if(!std::strcmp(test,"diffusion_metadata")){
  s.gasSpecies.thermo.coefficientCount=1;Real Y[2]={.25,.75},grad[2]={1,-1},D[2]={.01,.03},out[2]={-11,-12},energy=-13;
  ck(!ugkwp::correctedSpeciesDiffusion(Y,grad,Real(2),D,Real(0),Real(.8),Real(700),s.gasSpecies.thermo,out,energy),"metadata diffusion status");Real raw0=-Real(2)*D[0],raw1=Real(2)*D[1],j=raw0-Y[0]*(raw0+raw1);field(out[0],j,"actual metadata diffusion");field(out[1],-j,"actual closure");field(energy,Real(-3000000)*j,"actual supplied enthalpies");
 }else if(!std::strcmp(test,"face_count_labels")){
  // All cell fields still have the original eight allocated elements. Only
  // the declared cell count is changed; no pointer leaves its real backing.
  s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.nCells=1;s.faceOwner[1]=1;s.riemannBoundaryKind[1]=0;Real out[5]={-11,-12,-13,-14,-15};computeRiemannGasFaceFluxDevice<false>(s,1,out[0],out[1],out[2],out[3],out[4]);ck(out[0]==0&&out[1]==-s.p[1]&&out[2]==0&&out[3]==0&&out[4]==0,"actual allocated owner refused by count label");
  s.faceOwner[0]=0;s.faceNeighbour[0]=1;s.gasSpecies.faceStatus[0]=0;for(int k=0;k<5;++k)out[k]=-21-k;computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);ck(out[0]==0&&out[1]==s.p[0]&&out[2]==0&&out[3]==0&&out[4]==0,"actual allocated neighbour refused by count label");
 }else if(!std::strcmp(test,"face_metadata")){
  State expected;initialise(expected);s.gasReconstruction=expected.gasReconstruction=1;s.gasSpecies.thermo.coefficientCount=1;Real out[5]={-11,-12,-13,-14,-15},ref[5]={-21,-22,-23,-24,-25};
  computeRiemannGasFaceFluxDevice<false>(expected,0,ref[0],ref[1],ref[2],ref[3],ref[4]);computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);for(int k=0;k<5;++k)field(out[k],ref[k],"supplied actual face calorics");field(s.gasSpecies.flux[0],expected.gasSpecies.flux[0],"metadata species0");field(s.gasSpecies.flux[3],expected.gasSpecies.flux[3],"metadata species1");ck(s.gasSpecies.faceStatus[0]!=0,"retained metadata status");
 }else if(!std::strcmp(test,"face_status")){
  s.gasSpecies.cellStatus[0]=int(ugkwp::GasTransportCode::InvalidThermodynamics);Real out[5]={-11,-12,-13,-14,-15};
  computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);
  ck(out[1]==s.p[0]&&s.gasSpecies.flux[0]!=0&&s.gasSpecies.flux[3]==-s.gasSpecies.flux[0],"status hid actual face math");
 }else if(!std::strcmp(test,"face_diffusion")){
  Real D[2]={inf,.03};s.gasSpecies.diffusivity=D;Real out[5]={-11,-12,-13,-14,-15};s.gasSpecies.flux[0]=-21;s.gasSpecies.flux[3]=-22;
  computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);ck(out[1]==s.p[0]&&std::isnan(out[4]),"face bulk/energy diffusion tail");ck(std::isnan(s.gasSpecies.flux[0])&&std::isnan(s.gasSpecies.flux[3]),"face species diffusion tail");
 }else if(!std::strcmp(test,"face_nan")){
  s.rhoE[0]=nan;Real out[5]={-11,-12,-13,-14,-15};computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);ck(out[1]==s.p[0]&&std::isnan(out[4]),"actual Riemann NaN energy tail");
 }else if(!std::strcmp(test,"periodic")){
  s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;Real*bulk[5]={s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE};for(int k=0;k<5;++k){bulk[k][0]=k==0?inf:10+k;bulk[k][1]=1+k;}
  s.gasSpecies.flux[0]=nan;s.gasSpecies.flux[1]=-2;s.gasSpecies.flux[2]=6;s.gasSpecies.flux[3]=-4;threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
  for(int k=0;k<5;++k){Real expected=k==0?inf:Real(.5)*Real(9);field(bulk[k][0],expected,"periodic face");field(bulk[k][1],-expected,"periodic pair");}ck(std::isnan(s.gasSpecies.flux[0])&&std::isnan(s.gasSpecies.flux[1])&&s.gasSpecies.flux[2]==5&&s.gasSpecies.flux[3]==-5,"periodic all species");
 }else if(!std::strcmp(test,"positivity")){
  s.gasFluxPositivityScale[0]=-11;s.gasSpecies.positivityScale[0]=-12;s.gasSpecies.positivityScale[2]=-13;s.gasPhiRho[0]=2;s.gasSpecies.flux[0]=inf;s.gasSpecies.flux[3]=2;threadIdx.x=0;computeGasFluxPositivityScaleKernel(&s,Real(.1));
  field(s.gasFluxPositivityScale[0],1,"bulk factor");field(s.gasSpecies.positivityScale[0],0,"infinite outflow actual scale");field(s.gasSpecies.positivityScale[2],Real(.999)*Real(.1)/(Real(.1)*Real(2)),"later species factor");
 }else if(!std::strcmp(test,"subgrid")){
  s.turbulenceModel=3;s.sstWallTreatment=1;s.gasMu=1.8e-5;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;s.turbulentPrandtl=.9;s.sstWallDistance[0]=.5;s.k[0]=.1;s.Ux[0]=3;s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=0;
  Real mu=-11,k=-12,heat=-13;int active=-14;gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,heat,active);ck(s.gasSpecies.faceStatus[1]!=0,"subgrid certificate");ck(mu>=0&&std::isnan(k)&&std::isnan(heat)&&active==1,"subgrid real closure outputs");
 }else if(!std::strcmp(test,"thermal")){
  std::feclearexcept(FE_ALL_EXCEPT);auto t=ugkpwall::sstJayatillekeThermalTransport(Real(1),Real(1000),Real(0),Real(.7),Real(.9),Real(.09),Real(.41),Real(9.8),Real(0),Real(10),Real(.1),Real(.5),Real(3),Real(0),Real(2));
  ck(!t.valid&&t.heatFlux==0&&std::fetestexcept(FE_DIVBYZERO),"thermal actual zero-viscosity division");
  t=ugkpwall::sstJayatillekeThermalTransport(Real(1),Real(1000),Real(1.8e-5),Real(.7),Real(.9),Real(.09),Real(.41),Real(9.8),Real(0),Real(10),Real(.1),Real(.5),Real(3),Real(0),inf);ck(!t.valid&&t.heatFlux==-inf,"actual nonfinite thermal gradient tail");
 }else if(!std::strcmp(test,"original_status_skip")){
  s.gasSpecies.faceStatus[0]=int(ugkwp::GasTransportCode::InvalidThermodynamics);Real out[5]={-11,-12,-13,-14,-15};ck(computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"original internal computed-return blocked by old face status");ck(computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4],true),"original strict internal computed-return changed");
  s.gasSpecies.faceStatus[1]=int(ugkwp::GasTransportCode::InvalidThermodynamics);for(int k=0;k<5;++k)out[k]=-21-k;ck((computeRiemannGasFaceFluxDevice<false,true>(s,1,out[0],out[1],out[2],out[3],out[4]))&&out[0]==0,"original mass-only impermeable return changed");
 }else if(!std::strcmp(test,"legacy_periodic")){
  using Legacy=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;Legacy p=static_cast<const Legacy&>(s);p.nFaces=2;p.nInternalFaces=0;p.facePeriodicPair[0]=1;p.facePeriodicPair[1]=0;Real* b[5]={p.gasPhiRho,p.gasPhiRhoUx,p.gasPhiRhoUy,p.gasPhiRhoUz,p.gasPhiRhoE};for(int k=0;k<5;++k){b[k][0]=k==0?nan:Real(10+k);b[k][1]=1+k;}threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&p,false);ck(std::isnan(b[0][0])&&std::isnan(b[0][1]),"legacy actual nonfinite averages");for(int k=1;k<5;++k){field(b[k][0],Real(4.5),"legacy periodic face");field(b[k][1],Real(-4.5),"legacy periodic pair");}
 }else if(!std::strcmp(test,"legacy_positivity")){
  using Legacy=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;Legacy p=static_cast<const Legacy&>(s);p.V[0]=nan;p.gasPhiRho[0]=2;p.gasFluxPositivityScale[0]=-17;threadIdx.x=0;computeGasFluxPositivityScaleKernel(&p,Real(.1));field(p.gasFluxPositivityScale[0],1,"legacy actual inventory comparison");
 }else if(!std::strcmp(test,"partial_linear_enthalpy")||!std::strcmp(test,"partial_nasa_enthalpy")){
  ugkwp::SpeciesThermoData<Real> data[2]={s.gasSpecies.thermo.species[0],s.gasSpecies.thermo.species[1]};Real nasa[14]={3.5,.001,0,0,0,10,4,4.5,.002,0,0,0,20,5};auto full=s.gasSpecies.thermo;
  if(!std::strcmp(test,"partial_nasa_enthalpy")){for(int k=0;k<2;++k){data[k].model=ugkwp::SpeciesThermoModel::NASA7;data[k].coefficientOffset=7*k;data[k].midTemperature=1000;}full.species=data;full.coefficients=nasa;full.coefficientCount=14;}
  auto partial=full;partial.coefficientCount=full.coefficientCount-1;if(!std::strcmp(test,"partial_nasa_enthalpy"))partial.coefficientCount=12;
  // The original successful Cp-only/scalar and generic-bundle selectors stay.
  field(ugkwp::speciesCp(1,Real(700),partial),ugkwp::speciesCp(1,Real(700),full),"partial original Cp success");ck(std::isnan(ugkwp::speciesH(1,Real(700),partial)),"original partial scalar h changed");ugkwp::gasThermoDetail::SpeciesValues<Real> values;bool propertyEvaluated=false;ck(ugkwp::gasThermoDetail::evaluateSpecies(1,Real(700),partial,values,&propertyEvaluated)==ugkwp::ThermoStatus::Success&&propertyEvaluated&&std::isnan(values.h),"generic partial bundle changed");
  Real y[2]={.25,.75},grad[2]={1,-1},D[2]={.01,.03},flux[2]={-31,-32},heat=-33,ref[2]={-41,-42},expected=-43;bool evaluated=false;ck(ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),full,ref,expected),"full enthalpy reference");ck(!ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),partial,flux,heat,false,&evaluated)&&evaluated,"partial enthalpy certificate/availability");for(int k=0;k<2;++k)field(flux[k],ref[k],"partial actual diffusion flux");field(heat,expected,"partial actual backed enthalpy");if(!std::strcmp(test,"partial_linear_enthalpy"))field(heat,Real(90000),"linear actual 90000 heat");
  Real energy=-61;bool energyEvaluated=false;ck(ugkwp::evaluateMixtureInternalEnergy(y,Real(700),partial,energy,&energyEvaluated)!=ugkwp::ThermoStatus::Success&&energyEvaluated,"selected partial energy certificate/availability");field(energy,ugkwp::mixtureEnergy(y,Real(700),full),"selected actual partial energy");
  flux[0]=-51;flux[1]=-52;heat=-53;evaluated=true;ck(!ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),partial,flux,heat,true,&evaluated)&&!evaluated&&flux[0]==-51&&flux[1]==-52&&heat==-53,"partial strict diffusion transaction changed");
 }else if(!std::strcmp(test,"partial_face_calorics")){
  State expected;initialise(expected);s.gasReconstruction=expected.gasReconstruction=1;s.gasSpecies.thermo.coefficientCount=5;Real out[5]={-11,-12,-13,-14,-15},ref[5]={-21,-22,-23,-24,-25};computeRiemannGasFaceFluxDevice<false>(expected,0,ref[0],ref[1],ref[2],ref[3],ref[4]);ck(!computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"partial original face failure certificate");for(int k=0;k<5;++k)field(out[k],ref[k],"partial actual face calorics");for(int k=0;k<2;++k)field(s.gasSpecies.flux[k*3],expected.gasSpecies.flux[k*3],"partial actual face species");ck(s.gasSpecies.faceStatus[0]!=0,"partial face original certificate lost");
 }else if(!std::strcmp(test,"strict")){
  s.gasReconstruction=1;s.gradUxX[0]=nan;s.gasGradientLimiterUx[0]=1;GasPrimDevice out{-11,-12,-13,-14,-15,-16};Real Y[2]={-21,-22};ck(!reconstructGasMixtureFace(s,0,1,out,Y,true)&&out.rho==-11&&out.ux==-12&&out.T==-16&&Y[0]==-21&&Y[1]==-22,"strict MUSCL changed");
  Real y[2]={.25,.75},grad[2]={1,-1},D[2]={inf,.03},flux[2]={-31,-32},energy=-33;ck(!ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),s.gasSpecies.thermo,flux,energy,true)&&flux[0]==-31&&flux[1]==-32&&energy==-33,"strict diffusion changed");
 }else return 2;
}
'''

@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    folder=tmp_path_factory.mktemp('b2-face-completion')
    source=folder/'probe.cpp';source.write_text(PREAMBLE+fixture()+MAIN)
    exe=folder/'probe'
    result=subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    (folder/'build.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    return exe

@pytest.mark.parametrize('scenario',['frame','volume','shift_restore','boundary','boundary_total','legacy_flux','legacy_moving_nan','invalid_selector','nonfinite_coefficients','muscl','diffusion','diffusion_metadata','face_count_labels','face_metadata','face_status','face_diffusion','face_nan','periodic','positivity','subgrid','thermal','original_status_skip','legacy_periodic','legacy_positivity','partial_linear_enthalpy','partial_nasa_enthalpy','partial_face_calorics','strict'])
def test_actual_face_producer_completion(probe,scenario):
    result=subprocess.run([str(probe),scenario],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

SUCCESS = r'''
void dump(Real x){std::printf("%a ",double(x));}
int main(){
 for(int strict:{0,1})for(int scheme:{1,2})for(int recon:{0,1})for(int limiter:{0,1,2})for(int turbulence:{0,1,3}){
  State s;initialise(s);s.gasFluxScheme=scheme;s.gasReconstruction=recon;s.gasLimiter=limiter;s.turbulenceModel=turbulence;s.sstWallTreatment=1;s.gasMu=1.8e-5;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;s.turbulentPrandtl=.9;s.gasSpecies.turbulentSchmidt=.8;s.sstJayatillekeP=ugkpwall::jayatillekeSmoothP(s.gasPrClamped/s.turbulentPrandtl);s.sstThermalYPlus=ugkpwall::jayatillekeThermalYPlus(s.gasPrClamped/s.turbulentPrandtl,s.sstWallKappa,s.sstWallE);Real D[2]={.01,.03};s.gasSpecies.diffusivity=D;
  for(int c=0;c<2;++c){s.Ux[c]=.25+.125*c;s.Uy[c]=.0625*c;s.Uz[c]=.03;s.rhoUx[c]=s.rho[c]*s.Ux[c];s.rhoUy[c]=s.rho[c]*s.Uy[c];s.rhoUz[c]=s.rho[c]*s.Uz[c];s.sstWallDistance[c]=.5;s.k[c]=.1;s.nut[c]=.02;}
  s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=650;
  Real* gradients[]={s.gradRhoX,s.gradRhoY,s.gradRhoZ,s.gradUxX,s.gradUxY,s.gradUxZ,s.gradUyX,s.gradUyY,s.gradUyZ,s.gradUzX,s.gradUzY,s.gradUzZ,s.gradPx,s.gradPy,s.gradPz,s.gradTX,s.gradTY,s.gradTZ};Real*limiters[]={s.gasGradientLimiterRho,s.gasGradientLimiterUx,s.gasGradientLimiterUy,s.gasGradientLimiterUz,s.gasGradientLimiterP,s.gasGradientLimiterT};
  for(int c=0;c<2;++c){for(int k=0;k<18;++k)gradients[k][c]=-11-k;for(int k=0;k<6;++k)limiters[k][c]=-31-k;s.gasSpecies.limiter[c]=-41;for(int k=0;k<2;++k){s.gasSpecies.gradX[k*2+c]=-51-k;s.gasSpecies.gradY[k*2+c]=-61-k;s.gasSpecies.gradZ[k*2+c]=-71-k;}threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s,false,bool(strict));computeGasGradientLimiterKernel(&s,bool(strict));for(auto*p:gradients)dump(p[c]);for(auto*p:limiters)dump(p[c]);dump(s.gasSpecies.limiter[c]);for(int k=0;k<2;++k){dump(s.gasSpecies.gradX[k*2+c]);dump(s.gasSpecies.gradY[k*2+c]);dump(s.gasSpecies.gradZ[k*2+c]);}}
  for(int c=0;c<2;++c){GasPrimDevice p;Real y[2];ck(reconstructGasMixtureFace(s,c,0,p,y,bool(strict)),"normal MUSCL");for(auto x:{p.rho,p.ux,p.uy,p.uz,p.p,p.T,y[0],y[1]})dump(x);}
  for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,.0001,bool(strict));for(auto*p:{s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE})dump(p[f]);for(int k=0;k<2;++k)dump(s.gasSpecies.flux[k*3+f]);std::printf("%d ",s.gasSpecies.faceStatus[f]);ck(s.gasSpecies.faceStatus[f]==0,"original-success fixture has failed face");}
  Real mu=-11,k=-12,heat=-13;int active=-14;gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,heat,active,bool(strict));dump(mu);dump(k);dump(heat);std::printf("%d ",active);
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,.0001);dump(s.gasFluxPositivityScale[c]);for(int k=0;k<2;++k)dump(s.gasSpecies.positivityScale[k*2+c]);}
  for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s,bool(strict));for(auto*p:{s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE})dump(p[f]);for(int k=0;k<2;++k)dump(s.gasSpecies.flux[k*3+f]);}
  const Real secondF=s.gasSpecies.flux[3],secondPair=s.gasSpecies.flux[4];s.nFaces=2;s.gasSpecies.flux[2]=secondF;s.gasSpecies.flux[3]=secondPair;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s,bool(strict));ck(s.gasSpecies.faceStatus[0]==0&&s.gasSpecies.faceStatus[1]==0,"original-success periodic fixture failed");for(int f=0;f<2;++f){for(auto*p:{s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE})dump(p[f]);for(int k=0;k<2;++k)dump(s.gasSpecies.flux[k*2+f]);}std::puts("");
 }
 for(int fixed:{0,1}){ugkwp::GasFaceFrame<Real> frame;ck(ugkwp::makeGasFaceFrame(Real(2),Real(3),Real(4),fixed?Real(0):Real(.125),Real(.5),frame),"normal frame");for(Real x:{frame.nx,frame.ny,frame.nz,frame.area,frame.wx,frame.wy,frame.wz,frame.meshVolumeRate})dump(x);ugkpriemann::Primitive p{1,2,-0.,4,5},out;auto q=ugkpriemann::conservative(p,Real(1.4));ugkpriemann::Conservative qo;ck(ugkwp::shiftGasStateToFaceFrame(p,q,frame,out,qo),"normal shift");for(Real x:{out.rho,out.ux,out.uy,out.uz,out.p})dump(x);for(Real x:qo.q)dump(x);auto moving=ugkpriemann::rusanovTadmorFluxUnitNormal(p,p,frame.nx,frame.ny,frame.nz,Real(1.4),false),rest=moving;ck(ugkwp::restoreGasFluxFromFaceFrame(moving,frame,rest),"normal restore");for(Real x:rest.flux)dump(x);dump(rest.maxSignalSpeed);std::printf("%d %d %d\n",int(rest.evaluatedScheme),rest.valid,rest.usedFallback);}
}
'''

def compile_source(folder, root, body, label):
    folder.mkdir(parents=True,exist_ok=True)
    src=folder/(label+'.cpp');exe=folder/label;src.write_text(PREAMBLE+fixture()+body)
    cmd=['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(root/'common'),'-I'+str(root/'common/gasNumerics'),'-I'+str(HERE),str(src),'-o',str(exe)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (folder/(label+'-build.log')).write_text(' '.join(cmd)+'\n'+result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    (folder/(label+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout

def test_b2_original_success_bits(tmp_path):
    import io,tarfile
    base=tmp_path/'base';base.mkdir()
    revision='b236292f4c0b5e7e19ce04dd73e7fa0b54186bef'
    raw=subprocess.check_output(['git','archive',revision,'common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(base,filter='data')
    old=compile_source(tmp_path,base,SUCCESS,'original')
    new=compile_source(tmp_path,ROOT,SUCCESS,'current')
    assert old==new

def test_muscl_keeps_all_65_attempts_and_last_complete_fallback(tmp_path):
    header=(ROOT/'common/operators/reconstructGasMixtureFace.cuh').read_text()
    header=header.replace('    for(int attempt=0;attempt<=64;++attempt)\n    {','    for(int attempt=0;attempt<=64;++attempt)\n    {\n        ++attempts;')
    instrumented=tmp_path/'counted-reconstruction.cuh';instrumented.write_text(header)
    pre=PREAMBLE.replace('#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"','#define reconstructGasMixtureFace originalReconstructGasMixtureFace\n#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"\n#undef reconstructGasMixtureFace\nstatic int attempts=0;\n#include "'+str(instrumented)+'"')
    body=r'''
int main(){State s;initialise(s);s.gasReconstruction=1;s.rhoMin=2;
 GasPrimDevice out{-11,-12,-13,-14,-15,-16};Real y[2]={-21,-22};
 ck(reconstructGasMixtureFace(s,0,1,out,y)&&attempts==65,"direct 65-attempt complete fallback");
 ck(out.rho==1&&out.p==s.p[0]&&out.T==s.p[0]/(out.rho*ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo)),"final centre EOS evaluated");
 attempts=0;out={-11,-12,-13,-14,-15,-16};y[0]=-21;y[1]=-22;ck(!reconstructGasMixtureFace(s,0,1,out,y,true)&&attempts==65&&out.rho==-11&&y[0]==-21,"strict 65 attempts/transaction changed");
 // Same original last-complete selection as the independent continuation test:
 // the centre EOS is undefined, and a complete slope candidate precedes it.
 initialise(s);auto* meta=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);meta[1].molarMass=Real(2)*meta[0].molarMass;s.gasSpecies.rho[0]=-1;s.gasSpecies.rho[2]=2;s.gasReconstruction=1;s.gasSpecies.limiter[0]=1;s.gasSpecies.gradX[0]=1;
 attempts=0;out={-11,-12,-13,-14,-15,-16};y[0]=-21;y[1]=-22;ck(reconstructGasMixtureFace(s,0,0,out,y)&&attempts==65,"old fallback iteration budget changed");ck(y[0]>Real(-1)&&y[0]<0&&y[1]==1-y[0]&&std::isfinite(out.T)&&out.T==out.p/(out.rho*ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo)),"last complete fallback lineage changed");
}
'''
    src=tmp_path/'counted.cpp';src.write_text(pre+fixture()+body);exe=tmp_path/'counted'
    r=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr

def test_small_actual_cuda_face_compilation_when_available(tmp_path):
    import shutil
    nvcc=os.environ.get('NVCC')or shutil.which('nvcc')
    if not nvcc:pytest.skip('nvcc unavailable; host body tests do not prove CUDA compilation')
    pre=PREAMBLE
    for line in ['#define __host__\n','#define __global__\n','#define __device__\n','#define __forceinline__ inline\n','#define asm(...) std::abort()\n','struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;\n','#include "operators/computeSstFaceFluxKernel.cuh"\n']:
        pre=pre.replace(line,'')
    source=tmp_path/'b2-face.cu'
    source.write_text(pre+r'''
template<int Ns>struct CudaState:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,Ns> gasSpecies;};
using CudaLegacy=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;
static_assert(std::is_same<Real,double>::value,"actual face physics is FP64");
template<class S>__device__ void faces(S&s){
 Real out[5];computeRiemannGasFaceFluxDevice<true>(s,0,out[0],out[1],out[2],out[3],out[4],false);
 computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4],true);
 Real a,b;bool evaluated;ugkwp::gasGeometryCellVolumes(s,0,Time(.1),a,b,&evaluated);
}
__global__ void face2(CudaState<2>*s){faces(*s);}
__global__ void face10(CudaState<10>*s){faces(*s);}
__global__ void faceLegacy(CudaLegacy*s){faces(*s);}
template __global__ void computeGasPrimitiveGradientsKernel<CudaState<2>>(CudaState<2>*,const bool,const bool);
template __global__ void computeGasGradientLimiterKernel<CudaState<2>>(CudaState<2>*,const bool);
template __global__ void computeGasInternalFaceFluxKernel<true,CudaState<2>>(CudaState<2>*,const GPU_OPERATOR_TIME,const bool);
template __global__ void enforcePeriodicGasFluxAntisymmetryKernel<CudaState<2>>(CudaState<2>*,const bool);
template __global__ void computeGasFluxPositivityScaleKernel<CudaState<2>>(CudaState<2>*,const GPU_OPERATOR_TIME);
template __global__ void applyGasFluxPositivityScaleKernel<CudaState<2>>(CudaState<2>*,const bool);
''')
    output=tmp_path/'b2-face.o'
    cmd=[nvcc,'-std=c++17','-arch=sm_80','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-c',str(source),'-o',str(output)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'cuda-build.log').write_text(' '.join(cmd)+'\n'+result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert output.is_file()and output.stat().st_size>0

TRANSPORT_SKIP_SUCCESS = r'''
int main(){for(int strict:{0,1}){State s;initialise(s);s.gasMu=std::numeric_limits<Real>::quiet_NaN();s.gasThermalConductivity=0;s.Ux[0]=0;s.Ux[1]=2;Real out[5]={-11,-12,-13,-14,-15};bool result=computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4],bool(strict));std::printf("success=%d status=%d",result,s.gasSpecies.faceStatus[0]);for(auto v:out)std::printf(" %a",double(v));std::puts("");}}
'''

def test_original_nonfinite_transport_skip_bytes(tmp_path):
    import io,tarfile
    base=tmp_path/'base';base.mkdir()
    raw=subprocess.check_output(['git','archive','b236292f4c0b5e7e19ce04dd73e7fa0b54186bef','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(base,filter='data')
    assert compile_source(tmp_path,base,TRANSPORT_SKIP_SUCCESS,'original')==compile_source(tmp_path,ROOT,TRANSPORT_SKIP_SUCCESS,'current')

CV_ONLY_OVERFLOW_SUCCESS = r'''
int main(){State s;initialise(s);s.gasReconstruction=1;s.gasMu=0;s.gasThermalConductivity=0;s.rhoMin=1e-120;auto*coef=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coef[2]=coef[5]=std::numeric_limits<Real>::max()*.9;for(int c=0;c<2;++c){s.rho[c]=1e-100;s.gasSpecies.rho[c]=-s.rho[c];s.gasSpecies.rho[2+c]=2*s.rho[c];Real partial[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(700),s.gasSpecies.thermo);s.Tgas[c]=700;s.p[c]=s.rho[c]*ugkwp::mixtureGasConstant(partial,s.gasSpecies.thermo)*s.Tgas[c];}
Real out[5]={-11,-12,-13,-14,-15};bool ok=computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);std::printf("success=%d status=%d",ok,s.gasSpecies.faceStatus[0]);for(auto v:out)std::printf(" %a",double(v));std::puts("");}
'''

def test_original_cv_only_overflow_success_bytes(tmp_path):
    import io,tarfile
    base=tmp_path/'base';base.mkdir()
    raw=subprocess.check_output(['git','archive','b236292f4c0b5e7e19ce04dd73e7fa0b54186bef','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(base,filter='data')
    old=compile_source(tmp_path,base,CV_ONLY_OVERFLOW_SUCCESS,'original')
    assert old.startswith('success=1 status=0'),old
    assert old==compile_source(tmp_path,ROOT,CV_ONLY_OVERFLOW_SUCCESS,'current')

ENERGY_ONLY_CAPACITY_OVERFLOW_SUCCESS = r'''
int main(){State s;initialise(s);s.gasReconstruction=1;s.gasMu=0;s.gasThermalConductivity=0;auto*coef=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coef[2]=coef[5]=1;for(int c=0;c<2;++c){s.rho[c]=1e307;s.gasSpecies.rho[c]=.9*s.rho[c];s.gasSpecies.rho[2+c]=.1*s.rho[c];s.rhoE[c]=s.rho[c];s.p[c]=1;}Real out[5]={-11,-12,-13,-14,-15};bool ok=computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]);std::printf("success=%d status=%d",ok,s.gasSpecies.faceStatus[0]);for(auto v:out)std::printf(" %a",double(v));std::puts("");}
'''

def test_original_energy_only_capacity_overflow_success_bytes(tmp_path):
    import io,tarfile
    base=tmp_path/'base';base.mkdir()
    raw=subprocess.check_output(['git','archive','b236292f4c0b5e7e19ce04dd73e7fa0b54186bef','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(base,filter='data')
    old=compile_source(tmp_path,base,ENERGY_ONLY_CAPACITY_OVERFLOW_SUCCESS,'original')
    assert old.startswith('success=1 status=0'),old
    assert old==compile_source(tmp_path,ROOT,ENERGY_ONLY_CAPACITY_OVERFLOW_SUCCESS,'current')
