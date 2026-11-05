#include "gpu/MaterialThermalBackend.H"
#include "tests/TestSupport.H"
#include <iostream>
#include <vector>
using namespace chmt;
using namespace chmt_test;
int main(){
    MaterialThermalBackend device;
    for(int variant=0;variant<4;++variant){
        auto p=physics();p.gasConductivity=.08;
        if(variant==0){for(auto& c:p.condensed)c.cp1=0;for(auto& s:p.species)s.cp1=0;}
        if(variant==2){auto& s=p.species[0];s.model=ugkwp::SpeciesThermoModel::NASA7;s.R=300;s.Tmin=200;s.Tmid=1000;s.Tmax=3000;
            double low[]={3.5,.001,0,0,0,-10000,2},high[]={4,.0005,0,0,0,-10250,2};
            for(int i=0;i<7;++i){s.nasa7[i]=low[i];s.nasa7[i+7]=high[i];}}
        std::vector<MaterialThermalInput> input(257);std::vector<Real> temperature(257);
        for(int i=0;i<257;++i){auto& x=input[i];
            x.inventory.condensed[0]=500+i;x.inventory.condensed[1]=100;
            for(int j=0;j<Ns;++j)x.inventory.pore[j]=variant==3?0:.2+.03*j;
            x.materialVolume=variant==3?.1:1.;x.nativeVolume=x.materialVolume*(1+2e-16);
            x.anchor=250+5*i;temperature[i]=i%3==0?x.anchor:1700-5*i;
            std::string error;check(materialCaloric(x.inventory,p,x.caloric,error),error);
            x.target=x.caloric.energy(x.anchor)+17;
        }
        std::string error;check(device.begin(input,p,error),error);
        for(double dt:{1e-8,.001,3.}){
            std::vector<MaterialThermalCoefficients> actual;check(device.evaluate(temperature,dt,actual,error),error);
            for(int i=0;i<257;++i){const auto& x=input[i];const auto& got=actual[i];
                const double cs=x.caloric.secant(temperature[i],x.anchor);
                const double capacity=cs/x.nativeVolume;
                const double source=(x.target-x.caloric.energy(x.anchor)+cs*x.anchor)/(x.nativeVolume*dt);
                double occupied=0,k=0,poreMass=0;for(int j=0;j<Nc;++j){if(x.inventory.condensed[j]==0)continue;double v=x.inventory.condensed[j]/p.condensed[j].rho;occupied+=v;k+=v*p.condensed[j].conductivity/x.materialVolume;}
                for(int j=0;j<Ns;++j)poreMass+=x.inventory.pore[j];double phi=1-occupied/x.materialVolume;
                phi=maxValue(Real(0),minValue(Real(1),phi));double conductivity=k+(poreMass>0?phi*p.gasConductivity:0);
                check(got.error==0,"valid thermal input");
                near(got.capacity,capacity,"secant capacity",2e-14,0);near(got.source,source,"energy source",2e-14,0);
                check(got.conductivity==conductivity&&got.porosity==phi,"original pore/condensed conductivity and clipped porosity");
            }
        }
        std::cout<<"MATERIAL_THERMAL_RESULT "<<Ns<<' '<<variant<<" 257 cells x 3 intervals\n";
    }
    std::cout<<"PASS material implicit coefficients, polynomial/NASA7 secants and native/conservative volumes\n";
}
