"""Actual native interval owner and policies, with host CUDA execution."""
from test_native_original_success import source, build_run, ROOT

def test_numeric_nonretry_diagnostic_reaches_existing_terminal_operator(tmp_path):
    base=source(ROOT).split('int main(){for(double dt:')[0]
    body=base+r'''
int main(){DeviceState actual,direct;setupNative(actual,0,0);setupNative(direct,0,0);
 double oldVolume[2]={1,1},newVolume[2]={1,1},sweep[3]={.001,0,0};
 for(auto* s:{&actual,&direct}){s->gasGeometry.enabled=true;s->gasGeometry.oldVolume=oldVolume;s->gasGeometry.newVolume=newVolume;s->gasGeometry.faceSweptVolume=sweep;s->gasGeometry.interval=0;}
 GasRequestedIntervalControls<double> controls;
 assert(advanceGasRequestedInterval<SharedGasTrialPolicy>(&actual,.0001,2.,controls)==0);
 assert(advanceGasTrial<SharedGasTrialPolicy>(&direct,.0001,2.,false)==0);
 const double* a[]={actual.rho,actual.rhoUx,actual.rhoUy,actual.rhoUz,actual.rhoE,actual.gasSpecies.rho};
 const double* b[]={direct.rho,direct.rhoUx,direct.rhoUy,direct.rhoUz,direct.rhoE,direct.gasSpecies.rho};
 for(int k=0;k<6;++k)assert(std::memcmp(a[k],b[k],(k==5?4:2)*sizeof(double))==0);
 releaseSharedGasTrialStorage(actual.gasTrial);releaseSharedGasTrialStorage(direct.gasTrial);releaseSharedGasSpecies(actual.gasSpecies);releaseSharedGasSpecies(direct.gasSpecies);assert(allocations==0);
}
'''
    output=build_run(tmp_path,'nonretry-terminal',ROOT,body)
    assert output.count('TRIAL ')==3


def test_public_actual_missing_is_sticky_and_numeric_fields_commit(tmp_path):
    base=source(ROOT).split('int main(){for(double dt:')[0]
    body=base+r'''
int main(){DeviceState s;setupNative(s,0,0);s.rhoE[0]=NAN;
 assert(advanceGasTrial<SharedGasTrialPolicy>(&s,.0001,2.,false)==0);
 assert(std::isnan(s.rhoE[0]));
 assert(s.gasSpecies.cellStatus[0]!=0);
 auto* backing=s.gasSpecies.thermo.coefficients;s.gasSpecies.thermo.coefficients=nullptr;
 assert(SharedGasTrialPolicy::resetErrors(&s)==0);threadIdx.x=0;recoverGasPrimitiveCell(s,0,false);
 assert(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage));
 ugkwp::gasRecordCellFailure(s,0,ugkwp::GasTransportCode::InvalidThermodynamics);
 assert(SharedGasTrialPolicy::validate(&s,false)!=0&&!SharedGasTrialPolicy::retryable(&s));
 s.gasSpecies.thermo.coefficients=backing;
 releaseSharedGasTrialStorage(s.gasTrial);releaseSharedGasSpecies(s.gasSpecies);assert(allocations==0);
}
'''
    build_run(tmp_path,'terminal-fields',ROOT,body)



WALL_PROBE = r'''
struct WallProbe {
 int faces[1]={1},faceSlots[3]={-1,0,-1},ownerSlots[2]={0,-1},status[1]={0};
 int offsets[2]={0,1},matching[1]={1};double weights[1]={1};
 ugkwp::GasBoundaryLayerExchange<double> exchange[1];
 ugkwp::GasBoundaryLayerSstClosure<double> sst[1];double species[2]{};
 ugkwp::gaswall::WallInput<double,2> input[1];
 ugkwp::gaswall::WallOutput<double,2> output[1];
 ugkwp::gaswall::WallStatus modelStatus[1];
 void attach(DeviceState& s) {
  auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=faceSlots;w.ownerSlot=ownerSlots;
  w.status=status;w.exchange=exchange;w.sst=sst;w.speciesFlux=species;
  auto&m=s.gasBoundaryLayerModel;m.config.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
  m.useSuppliedWallState=true;m.faces=faces;m.matchingOffsets=offsets;m.matchingCells=matching;m.matchingWeights=weights;
  m.input=input;m.output=output;m.status=modelStatus;m.workspace=nullptr;m.workspaceCount=0;
  s.gasBoundaryKind[1]=s.riemannBoundaryKind[1]=2;s.gasMu=.02;s.gasThermalConductivity=.5;
  input[0].normal[0]=1;input[0].temperature=500;input[0].matchingDistance=1.5;input[0].ownerDistance=.5;
  input[0].quadrature.volume=1;input[0].massFlux[0]=input[0].massFlux[1]=std::numeric_limits<double>::max();
 }
};
'''


def test_selected_wall_numeric_diagnosis_reaches_actual_terminal_fields(tmp_path):
    base=source(ROOT).split('int main(){for(double dt:')[0]
    body=base+WALL_PROBE+r'''
int main(){for(int inputKind:{0,1,2})for(int boundaryKind:{1,2}){
 DeviceState native,direct;setupNative(native,0,0);setupNative(direct,0,0);
 WallProbe a,b;a.attach(native);b.attach(direct);
 const double flux=inputKind==0?.001:inputKind==1?std::numeric_limits<double>::max():NAN;
 for(auto* wall:{&a,&b})wall->input[0].massFlux[0]=wall->input[0].massFlux[1]=flux;
 native.gasBoundaryKind[1]=native.riemannBoundaryKind[1]=boundaryKind;
 direct.gasBoundaryKind[1]=direct.riemannBoundaryKind[1]=boundaryKind;
 assert(ugkwp::evaluateGasBoundaryLayerSlot(native,0));
 assert(a.exchange[0].ready&&a.exchange[0].massReady);
 double face[5];const bool strict=computeRiemannGasFaceFluxDevice<false,true>(native,1,face[0],face[1],face[2],face[3],face[4],true);
 if(inputKind||boundaryKind!=2){
  assert(!strict&&native.gasSpecies.faceStatus[1]!=int(ugkwp::GasTransportCode::InvalidStorage));
  assert(SharedGasTrialPolicy::validate(&native,true)!=0&&SharedGasTrialPolicy::continueTerminal(&native));
 }else assert(strict&&native.gasSpecies.faceStatus[1]==0);
 GasRequestedIntervalControls<double> controls;controls.maximumAttempts=1;
 assert(advanceGasRequestedInterval<SharedGasTrialPolicy>(&native,1e-5,2.,controls)==0);
 assert(advanceGasTrial<SharedGasTrialPolicy>(&direct,1e-5,2.,false)==0);
 const double* left[]={native.rho,native.rhoUx,native.rhoUy,native.rhoUz,native.rhoE,native.gasSpecies.rho};
 const double* right[]={direct.rho,direct.rhoUx,direct.rhoUy,direct.rhoUz,direct.rhoE,direct.gasSpecies.rho};
 for(int k=0;k<6;++k)assert(std::memcmp(left[k],right[k],(k==5?4:2)*sizeof(double))==0);
 if(inputKind)assert(std::isnan(native.rhoE[0]));
 releaseSharedGasTrialStorage(native.gasTrial);releaseSharedGasTrialStorage(direct.gasTrial);
 releaseSharedGasSpecies(native.gasSpecies);releaseSharedGasSpecies(direct.gasSpecies);assert(allocations==0);
}}
'''
    build_run(tmp_path,'wall-numeric-terminal',ROOT,body)


def test_wall_producer_only_requires_its_consumed_inputs(tmp_path):
    base=source(ROOT).split('int main(){for(double dt:')[0]
    body=base+WALL_PROBE+r'''
int main(){DeviceState actual,reference;setupNative(actual,0,0);setupNative(reference,0,0);
 WallProbe a,b;a.attach(actual);b.attach(reference);
 for(auto* wall:{&a,&b})wall->input[0].massFlux[0]=wall->input[0].massFlux[1]=.001;
 auto* owners=actual.faceOwner;actual.faceOwner=nullptr;
 assert(ugkwp::evaluateGasBoundaryLayerSlot(actual,0));
 assert(ugkwp::evaluateGasBoundaryLayerSlot(reference,0));
 assert(a.exchange[0].ready&&a.exchange[0].massReady&&a.status[0]==0);
 const double* fieldsA[]={&a.exchange[0].mass,&a.exchange[0].momentumX,&a.exchange[0].momentumY,&a.exchange[0].momentumZ,&a.exchange[0].energy,a.species};
 const double* fieldsB[]={&b.exchange[0].mass,&b.exchange[0].momentumX,&b.exchange[0].momentumY,&b.exchange[0].momentumZ,&b.exchange[0].energy,b.species};
 for(int k=0;k<6;++k)assert(std::memcmp(fieldsA[k],fieldsB[k],(k==5?2:1)*sizeof(double))==0);
 // The final face operator genuinely consumes faceOwner; supply it at that use.
 actual.faceOwner=owners;double x[5],y[5];
 assert(computeRiemannGasFaceFluxDevice<false>(actual,1,x[0],x[1],x[2],x[3],x[4]));
 assert(computeRiemannGasFaceFluxDevice<false>(reference,1,y[0],y[1],y[2],y[3],y[4]));
 assert(std::memcmp(x,y,sizeof(x))==0);
 actual.gasBoundaryLayer.exchange=nullptr;
 assert(!computeRiemannGasFaceFluxDevice<false>(actual,1,x[0],x[1],x[2],x[3],x[4],true));
 assert(actual.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidStorage));
 assert(SharedGasTrialPolicy::validate(&actual,true)!=0&&!SharedGasTrialPolicy::continueTerminal(&actual));
 actual.gasBoundaryLayer.exchange=a.exchange;assert(SharedGasTrialPolicy::resetErrors(&actual)==0);
 auto* coefficients=actual.gasSpecies.thermo.coefficients;actual.gasSpecies.thermo.coefficients=nullptr;
 assert(!ugkwp::evaluateGasBoundaryLayerSlot(actual,0));
 assert(!a.exchange[0].ready&&a.status[0]==int(ugkwp::GasTransportCode::InvalidStorage));
 assert(!computeRiemannGasFaceFluxDevice<false>(actual,1,x[0],x[1],x[2],x[3],x[4],true));
 assert(SharedGasTrialPolicy::validate(&actual,true)!=0&&!SharedGasTrialPolicy::continueTerminal(&actual));
 actual.gasSpecies.thermo.coefficients=coefficients;
 releaseSharedGasTrialStorage(actual.gasTrial);releaseSharedGasTrialStorage(reference.gasTrial);
 releaseSharedGasSpecies(actual.gasSpecies);releaseSharedGasSpecies(reference.gasSpecies);assert(allocations==0);
}
'''
    build_run(tmp_path,'wall-producer-inputs',ROOT,body)
