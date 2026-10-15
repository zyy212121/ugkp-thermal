#include "gpu/SharedMixtureStorage.H"
#include <cassert>
#include <cstring>
int main(){
    auto shared=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen; species (A B);
    speciesThermo {
      A {model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 3000; coefficients (1040 0 700);}
      B {model linearCp; molarMass 0.032; minTemperature 100; maxTemperature 3000; coefficients (950 0.1 -300);}
    } diffusion {model constant; coefficients (0.001 0.002);})");
    chmt::ModelConfig model;std::string error;assert(chmt::bindSharedGasModel(shared,model,error));
    chmt::HostState source;source.gasMesh.volumes={2,4};source.gasMesh.owner={0,1};
    chmt::GasPrimitive w;w.rho=2;w.temperature=600;w.Y[0]=.7;w.Y[1]=.3;
    for(auto v:source.gasMesh.volumes)source.gas.push_back(chmt::conservativeGas(w,v,model.physics));
    chmt::SharedMixtureStorage storage;assert(storage.import(source,shared,model,error));
    auto view=storage.view();
    static_assert(ugkwp::GasStateTraits<decltype(view)>::speciesCount==chmt::Ns,"actual shared species trait");
    assert(view.rho[0]==2&&view.gasSpecies.rho[0]==1.4&&view.gasSpecies.rho[2]==.6);
    assert(view.nut&&view.nut[0]==0&&view.gasSpecies.initial&&view.gasSpecies.faceStatus);
    auto copy=storage;auto copied=copy.view();assert(copied.rho!=view.rho);
    assert(copied.gasSpecies.thermo.species!=view.gasSpecies.thermo.species);
    auto accepted=source.gas;assert(storage.publish(source.gasMesh.volumes,model,accepted,error));
    assert(accepted[1].species[0]==source.gas[1].species[0]);
    const auto prior=accepted;view.gasSpecies.cellStatus[1]=1;
    assert(!storage.publish(source.gasMesh.volumes,model,accepted,error));
    assert(accepted[0].mass==prior[0].mass);view.gasSpecies.cellStatus[1]=0;
    view.gasSpecies.rho[0]=-1;
    assert(!storage.publish(source.gasMesh.volumes,model,accepted,error));
    assert(accepted[0].species[0]==prior[0].species[0]);
    assert(!copy.publish({3,4},model,accepted,error));
}
