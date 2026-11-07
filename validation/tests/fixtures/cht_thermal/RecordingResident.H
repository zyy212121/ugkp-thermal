// Test-only resident boundary. All CPU thermal owners remain production code.
#ifndef T14_RECORDING_RESIDENT_H
#define T14_RECORDING_RESIDENT_H
#include "fvCFD.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include <fstream>
#include <stdexcept>
#include "ResidentRadiationFixture.H"
namespace Foam {
struct GpuThermalRestartMirrorMetadata {label particleCount{1};};
class GpuParticleResidentSolver {
 const fvMesh& mesh_;
 mutable ResidentRadiationFixture radiationDevice_;
public:
 scalarField gas,deposited,reflected,uploadedTemperature,radiationEnergy,radiationMean;label radiationApplications{0},enthalpyRefreshes{0};
 labelList faces;label gasResets{0},particleResets{0},gasDownloads{0},particleDownloads{0};
 bool mismatch{false},failConsume{false},zeroAtConfigure{false};label restoreCalls{0};
 explicit GpuParticleResidentSolver(const fvMesh& mesh):mesh_(mesh),gas(mesh.nFaces(),0),deposited(mesh.nFaces(),0),reflected(mesh.nFaces(),0),uploadedTemperature(mesh.nFaces(),0){}
 scalar rhoSolid()const{return 1000;}scalar particleHeatFactorJkgK()const{return 1;}scalar particleTemperatureMin()const{return 1;}scalar particleTemperatureMax()const{return 4000;}label particleCapacity()const{return 10;}label particleCount()const{return 1;}
 bool particleWallHeatTransferEnabled()const{return true;}
 bool particleWallContactEnabledOnPatch(const fvMesh&,label p)const{return mesh_.boundary()[p].name()=="left";}
 const word& particleStuckWallHeatFluxFieldName()const{static word n("particleStuckWallHeatFlux");return n;}
 const word& particleReflectedWallHeatFluxFieldName()const{static word n("particleReflectedWallHeatFlux");return n;}
 void configureGasWallEnergyLedger(const fvMesh& m,const labelList& patches){label n=0;forAll(patches,p)n+=m.boundary()[patches[p]].size();faces.setSize(n);label j=0;forAll(patches,p){const auto& patch=m.boundaryMesh()[patches[p]];forAll(patch,f){faces[j++]=patch.start()+f;gas[patch.start()+f]=zeroAtConfigure?0:50;deposited[patch.start()+f]=zeroAtConfigure?0:20;reflected[patch.start()+f]=zeroAtConfigure?0:-5;}}}
 #include "InterfaceExchangeResident.inc"
 // This fixture deliberately records a host-owned gas ledger. Resident solid
 // graph coverage uses solid_contact_owner and the actual CHT executable.
 ugkp::diagnostics::DeviceContext* deviceDiagnostics()const{return nullptr;}
 ugkp::gpu::ThermalInterfaceEnergyDeviceView interfaceEnergyDeviceState()const{return {};}
 void prepareInterfaceExchangeDevice(scalar,const ugkp::gpu::RadiationMaterialDeviceView*)const{
  throw std::logic_error("recording host ledger cannot advertise a resident interval");
 }
 void collectInterfaceExchange(List<scalarField>&,List<scalarField>&,List<scalarField>&)const{
  throw std::logic_error("recording host ledger has no deferred resident preview");
 }

 const labelList& configuredGasWallEnergyFaceIds()const{return faces;}
 void restorePendingWallEnergyLedgers(const scalarField& g,const scalarField& d,const scalarField& r){++restoreCalls;forAll(faces,i){gas[faces[i]]=g[i];deposited[faces[i]]=d[i];reflected[faces[i]]=r[i];}}
 void peekPendingWallEnergyLedgers(scalarField& g,scalarField& d,scalarField& r)const{g.setSize(faces.size());d.setSize(faces.size());r.setSize(faces.size());forAll(faces,i){g[i]=gas[faces[i]];d[i]=deposited[faces[i]];r[i]=reflected[faces[i]];}}
 List<scalarField> peekGasWallEnergy()const{List<scalarField> result(1);result[0].setSize(faces.size());forAll(faces,i)result[0][i]=gas[faces[i]];return result;}
 void peekParticleWallHeatLedgers(scalarField& d,scalarField& r)const{d=deposited;r=reflected;}
 void resetGasWallEnergy(){if(failConsume)throw std::runtime_error("injected actual consume error");gas=0;++gasResets;}
 void resetParticleWallHeatLedgers(){deposited=0;reflected=0;++particleResets;}
 void downloadAndResetGasWallEnergy(scalarField& g){if(failConsume)throw std::runtime_error("injected actual consume error");++gasDownloads;g.setSize(faces.size());forAll(faces,i)g[i]=gas[faces[i]]+(mismatch?7:0);gas=0;++gasResets;}
 void downloadAndResetParticleWallHeatLedgers(scalarField& d,scalarField& r){particleDownloads+=2;d=deposited;r=reflected;if(mismatch){d+=3;r-=2;}deposited=0;reflected=0;++particleResets;}
 void updateSolidInterfaceState(const ugkp::gpu::ThermalInterfaceStateDeviceView& view){
  ResidentRadiationFixture::check(cudaStreamSynchronize(static_cast<cudaStream_t>(view.producerStream)));
  scalarField values(view.faces),effusivity(view.faces);labelList contact(view.faces);
  if(view.faces){ResidentRadiationFixture::check(cudaMemcpy(values.begin(),view.temperature,view.faces*sizeof(double),cudaMemcpyDeviceToHost));
   ResidentRadiationFixture::check(cudaMemcpy(effusivity.begin(),view.effusivity,view.faces*sizeof(double),cudaMemcpyDeviceToHost));
   ResidentRadiationFixture::check(cudaMemcpy(contact.begin(),view.contactEnabled,view.faces*sizeof(int),cudaMemcpyDeviceToHost));}
  label solidOffset=0,fluidOffset=0;
  forAll(interfaceDestinations,p){forAll(interfaceDestinations[p],f){const label i=solidOffset+interfaceDestinations[p][f];
   uploadedTemperature[faces[fluidOffset+f]]=values[i];if(contact[i]&&!std::isfinite(effusivity[i]))throw std::runtime_error("nonfinite required effusivity");}
   solidOffset+=interfaceSolidSizes[p];fluidOffset+=interfaceDestinations[p].size();}
 }
 void updateSolidWallTemperatures(const fvMesh& m,const volScalarField& T,const labelList& patches){forAll(patches,p){const auto& patch=m.boundaryMesh()[patches[p]];forAll(patch,f)uploadedTemperature[patch.start()+f]=T.boundaryField()[patches[p]][f];}}
 void updateParticleWallEffusivities(const fvMesh&,const List<scalarField>& e,const labelList& patches){forAll(patches,p)forAll(e[patches[p]],f)if(!std::isfinite(e[patches[p]][f]))throw std::runtime_error("nonfinite required effusivity");}
 void downloadMobileParticleRadiationMaterial(const fvMesh& mesh,scalarField& mass,scalarField& mean,scalarField& diameter)const{ugkwp::ParticleCaloric law;ugkwp::ParticleThermalCarry<double> carried{4000.f,law.enthalpy(4100)-law.enthalpy(4000)};mass.setSize(mesh.nCells(),.001);mean.setSize(mesh.nCells(),ugkwp::effectiveParticleTemperature(carried,law));diameter.setSize(mesh.nCells(),1.5e-4);}
 scalarField particleWallOccupiedContactArea()const{return scalarField(mesh_.nFaces(),0);}
 void applyParticleRadiationEnergy(const fvMesh&,const scalarField& energy,const scalarField&,const scalarField& mean,scalar){radiationEnergy=energy;radiationMean=mean;++radiationApplications;}
 ugkp::gpu::ParticleRadiationResponseView particleRadiationResponse()const{return {};}
 ugkp::gpu::RadiationParticleDeviceView radiationParticleDeviceState()const{
  scalarField mass,mean,diameter;downloadMobileParticleRadiationMaterial(mesh_,mass,mean,diameter);
  return radiationDevice_.stage(mesh_,mass,mean,diameter,rhoSolid(),particleTemperatureMin());
 }
 void applyParticleRadiationDeviceEnergy(const ugkp::gpu::RadiationMaterialDeviceView& result){
  radiationDevice_.readEnergy(result,radiationEnergy);radiationMean=radiationDevice_.mean;++radiationApplications;
 }
 void refreshParticleEnthalpyAfterRadiation(){++enthalpyRefreshes;}
 void downloadSharedGasSpecies(const volScalarField&)const{}
 void downloadToHostMirror(Time&,volScalarField&,volVectorField&,volScalarField&,volVectorField&,volScalarField&,volScalarField&,volScalarField&,volVectorField&,volScalarField&,volScalarField&,volScalarField&,volVectorField&,volScalarField&,volScalarField&,volScalarField&,bool)const{}
 GpuThermalRestartMirrorMetadata stageThermalRestartMirrors(const Time& t,bool)const{mkDir(t.timePath());GpuThermalRestartMirrorMetadata m;for(const fileName& n:{fileName("recording-particle.dat"),fileName("recording-eps.dat"),fileName("recording-source.dat")}){std::ofstream out((t.timePath()/n).c_str());out<<"test resident payload\n";}return m;}
};}
#endif
