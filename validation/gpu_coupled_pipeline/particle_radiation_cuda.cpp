#include "gpu/ParticleRadiationMaterial.H"
#include "MieTable.H"
#include "OriginalMieTable.H"
#include "OriginalMieDoRadiationMath.H"
#include "OriginalParticleRadiationImplicitMath.H"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace ugkp::gpu;
namespace original=Foam::originalGpuThermal;
void same(double a,double b){if(std::isnan(a)&&std::isnan(b))return;if(std::memcmp(&a,&b,sizeof(double))){
    std::cerr<<std::hexfloat<<a<<" != "<<b<<'\n';throw std::runtime_error("frozen material transaction differs");}}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    Foam::gpuThermal::MieTable current(argv[1]);original::MieTable table(argv[1]);ParticleRadiationMaterial device(current.exportGpuTable());
    std::mt19937 rng(900142);int trials=0;
    for(int n:{1,2,257,4096})for(int scenario=0;scenario<8;++scenario){
        ParticleRadiationMaterialInput x;x.epsilon.resize(n);x.mass.resize(n);x.temperature.resize(n);x.diameter.resize(n);x.incident.resize(n);x.volume.resize(n);
        x.elapsed=scenario==1?0:scenario==2?-.01:.01;x.heatFactor=scenario==3?0:1000;x.minimumTemperature=300;x.maximumTemperature=3000;
        for(int i=0;i<n;++i){x.epsilon[i]=(i%5==0)?0:1e-5*(1+rng()%17);x.mass[i]=1e-5*(1+rng()%23)*(scenario==4?-1:1);
            x.temperature[i]=300+rng()%3000;x.diameter[i]=1e-5*(1+rng()%20);x.incident[i]=double(rng()%100000);x.volume[i]=.001*(1+rng()%7);
            if(scenario==5&&i%3==0)x.incident[i]=std::numeric_limits<double>::quiet_NaN();
            if(scenario==6&&i%3==0)x.diameter[i]=0;
            if(scenario==7)x.epsilon[i]=std::numeric_limits<double>::quiet_NaN();}
        const int faces=2*n+3;x.boundaryArea.resize(faces);x.boundaryFlux.resize(faces);x.coupledBoundary.resize(faces);
        for(int i=0;i<faces;++i){x.boundaryArea[i]=.01*(1+rng()%7);x.boundaryFlux[i]=double(int(rng()%10000)-5000);x.coupledBoundary[i]=(i%3)==0;}
        std::vector<double> temperature=x.temperature,energy(n,0),boundary(faces);double accepted=0,gross=0,rawTotal=0,escaped=0;
        for(int i=0;i<n;++i){if(!(x.epsilon[i]>0))continue;
            const auto balance=[&](double t){const auto optical=table.query(x.diameter[i],t);return x.incident[i]-original::fourPiConstant()*original::bandBlackBodyIntensity(t,optical.planckBandFraction);};
            const auto power=[&](double t){const auto optical=table.query(x.diameter[i],t);const double a=3*x.epsilon[i]*optical.Qabs/(2*x.diameter[i]);
                return x.volume[i]*original::particleRadiationPowerDensity(a,x.incident[i],original::bandBlackBodyIntensity(t,optical.planckBandFraction));};
            const double capacity=x.mass[i]*x.heatFactor;
            auto root=original::detail::solveBoundedImplicitMaterialTemperature(x.temperature[i],300,3000,capacity,x.elapsed,100,1e-10,1e-9,balance,power);
            temperature[i]=root.temperatureK;energy[i]=capacity*(root.temperatureK-x.temperature[i]);accepted+=energy[i];gross+=std::abs(energy[i]);
        }
        for(int i=0;i<faces;++i){boundary[i]=x.elapsed*x.boundaryArea[i]*x.boundaryFlux[i];rawTotal+=boundary[i];gross+=std::abs(boundary[i]);}
        const auto scale=original::detail::conservativeBoundaryEnergyScale(accepted,rawTotal,gross,1e-10);
        for(int i=0;i<faces;++i){const double value=scale.scale*boundary[i];boundary[i]=0;if(x.coupledBoundary[i])boundary[i]=value;else escaped+=value;}
        auto actual=device.solve(x);for(int i=0;i<n;++i){same(temperature[i],actual.temperature[i]);same(energy[i],actual.energy[i]);}
        for(int i=0;i<faces;++i)same(boundary[i],actual.boundaryEnergy[i]);same(escaped,actual.escapedEnergy);++trials;
    }
    std::cout<<"PASS "<<trials<<" frozen material-root, boundary-scaling and ordered-budget transactions\n";
}
