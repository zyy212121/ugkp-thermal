#ifndef CHMT_GPU_MATERIAL_THERMAL_BACKEND_H
#define CHMT_GPU_MATERIAL_THERMAL_BACKEND_H
#include "materials/ConductionCaloric.H"
#include <memory>
#include <vector>
namespace chmt {
struct MaterialThermalInput {
    MaterialConductionCaloric caloric;
    MaterialConductionInventory inventory;
    ConductionReal anchor=0,target=0,nativeVolume=0,materialVolume=0;
};
struct MaterialThermalCoefficients {
    ConductionReal capacity=0,source=0,conductivity=0,porosity=0,rawPorosity=0;
    int error=0;
};
CHMT_HD inline MaterialThermalCoefficients materialThermalCoefficients(const MaterialThermalInput& input,
    ConductionReal temperature,ConductionReal dt,const PhysicsConfig& p){
    MaterialThermalCoefficients out;
    const ConductionReal cs=input.caloric.secant<false>(temperature,input.anchor);
    if(!(cs>0)||!std::isfinite(cs)){out.error=1;return out;}
    out.capacity=cs/input.nativeVolume;
    out.source=(input.target-input.caloric.energy<false>(input.anchor)+cs*input.anchor)/(input.nativeVolume*dt);
    ConductionReal occupied=0,k=0,poreMass=0;
    for(int j=0;j<Nc;++j){if(input.inventory.condensed[j]==0)continue;
        const ConductionReal v=input.inventory.condensed[j]/ConductionReal(p.condensed[j].rho);occupied+=v;k+=v*ConductionReal(p.condensed[j].conductivity)/input.materialVolume;}
    for(int j=0;j<Ns;++j)poreMass+=input.inventory.pore[j];
    out.rawPorosity=1-occupied/input.materialVolume;out.porosity=out.rawPorosity;
    if(!std::isfinite(out.porosity)){out.error=2;return out;}
    out.porosity=fmaxf(ConductionReal(0),fminf(ConductionReal(1),out.porosity));
    out.conductivity=k+(poreMass>0?out.porosity*ConductionReal(p.gasConductivity):0);
    return out;
}
class MaterialThermalBackend {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    MaterialThermalBackend();
    ~MaterialThermalBackend();
    bool begin(const std::vector<MaterialThermalInput>&,const PhysicsConfig&,std::string& error);
    bool evaluate(const std::vector<Real>& temperature,Real dt,std::vector<MaterialThermalCoefficients>&,std::string& error);
};
}
#endif
