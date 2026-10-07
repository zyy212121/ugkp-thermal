#include "configuration/SharedGasModel.H"
#include "gpu/SharedGasStorage.H"
#include <cassert>
#include <cmath>
int main(){
    const std::string text=R"(gasMode mixtureFrozen; species (S0 S1); speciesThermo {
      S0 { model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 3000; coefficients (1040 0 700); }
      S1 { model linearCp; molarMass 0.032; minTemperature 100; maxTemperature 3000; coefficients (950 0.1 -300); }
    } diffusion { model constant; coefficients (0.001 0.002); turbulentSchmidt 0.8; })";
    auto gas=ugkwp::parseGasModelProperties(text); chmt::ModelConfig model; std::string error;
    assert(chmt::bindSharedGasModel(gas,model,error));
    assert(model.physics.gasMode==ugkwp::GasMode::MixtureFrozen);
    assert(model.physics.gasThermoFingerprint==gas.thermoHash);
    assert(model.physics.species[0].e0==700&&model.physics.gasDiffusivity[1]==0.002);
    auto bad=gas;bad.speciesNames[0]="wrong";
    const auto before=model.physics.species[0].e0;
    assert(!chmt::bindSharedGasModel(bad,model,error));
    assert(model.physics.species[0].e0==before);
    chmt::HostState host;host.gasMesh.volumes={2,4};host.gas.resize(2);
    for(auto& q:host.gas){q.mass=2;q.momentum={4,6,8};q.energy=30;q.species[0]=2;}
    chmt::SharedGasDensityStorage storage;
    assert(storage.importSingleSpecies(host,0,error));
    auto view=storage.view();assert(view.rho[0]==1&&view.rho[1]==0.5);
    assert(view.rhoUx[0]==2&&view.rhoE[1]==7.5);
    view.rho[0]=1.5;view.rhoE[0]=20;
    assert(storage.exportSingleSpecies(host.gasMesh.volumes,0,host.gas,error));
    assert(host.gas[0].mass==3&&host.gas[0].species[0]==3&&host.gas[0].energy==40);
    auto prior=host.gas;view.rho[1]=-1;
    assert(!storage.exportSingleSpecies(host.gasMesh.volumes,0,host.gas,error));
    assert(host.gas[0].mass==prior[0].mass&&host.gas[1].mass==prior[1].mass);
    host.gas[0].species[0]=2;host.gas[0].species[1]=1;
    assert(!storage.importSingleSpecies(host,0,error));
}
