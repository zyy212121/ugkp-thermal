#ifndef CHMT_GPU_MATERIAL_THERMAL_BACKEND_H
#define CHMT_GPU_MATERIAL_THERMAL_BACKEND_H
#include "materials/MaterialCaloric.H"
#include <memory>
#include <vector>
namespace chmt {
struct MaterialThermalInput {
    MaterialCaloric caloric;
    SolidQ inventory;
    Real anchor=0,target=0,nativeVolume=0,materialVolume=0;
};
struct MaterialThermalCoefficients {
    Real capacity=0,source=0,conductivity=0,porosity=0,rawPorosity=0;
    int error=0;
};
CHMT_HD inline MaterialThermalCoefficients materialThermalCoefficients(const MaterialThermalInput& input,
    Real temperature,Real dt,const PhysicsConfig& p){
    MaterialThermalCoefficients out;
    const Real cs=input.caloric.secant<false>(temperature,input.anchor);
    if(!(cs>0)||!finite(cs)){out.error=1;return out;}
    out.capacity=cs/input.nativeVolume;
    out.source=(input.target-input.caloric.energy<false>(input.anchor)+cs*input.anchor)/(input.nativeVolume*dt);
    Real occupied=0,k=0,poreMass=0;
    for(int j=0;j<Nc;++j){if(input.inventory.condensed[j]==0)continue;
        const Real v=input.inventory.condensed[j]/p.condensed[j].rho;occupied+=v;k+=v*p.condensed[j].conductivity/input.materialVolume;}
    for(int j=0;j<Ns;++j)poreMass+=input.inventory.pore[j];
    out.rawPorosity=1-occupied/input.materialVolume;out.porosity=out.rawPorosity;
    if(!finite(out.porosity)){out.error=2;return out;}
    out.porosity=maxValue(Real(0),minValue(Real(1),out.porosity));
    out.conductivity=k+(poreMass>0?out.porosity*p.gasConductivity:0);
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
