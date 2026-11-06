#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/ParticleRadiationMaterial.H"
#include "gpu/RadiationImplicitMath.H"
#include "Diagnostics.H"
#include <cuda_runtime.h>
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace ugkp::gpu;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    double* p=nullptr;
    explicit Buffer(const std::vector<double>& v){check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(std::size_t(1),v.size())*sizeof(double)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(double),cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(p);}
};
void same(const std::vector<double>& a,const std::vector<double>& b,const char* message){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double)))throw std::runtime_error(message);
}
RadiationGeometry geometry(){
    RadiationGeometry g;g.cells=2;g.directions=2;g.owner={0};g.neighbour={1};g.cellOffsets={0,1,2};g.cellFaces={0,0};
    g.volume={1,1};const double pi=std::acos(-1.);g.weights={2*pi,2*pi};g.directionCosines={1,-1,-1,1};
    g.boundaryOwner={0,1};g.boundaryPatch={0,1};g.boundaryFace={0,0};
    g.boundaryType={RadiationBoundary::Diffuse,RadiationBoundary::Black};
    g.boundaryArea={1,1};g.diffuseDenominator={2*pi,2*pi};g.fixedIntensity={0,0};
    g.internalFlux={1,-1};g.transportDiagonal={1,1,1,1};g.boundaryFlux={-1,1,1,-1};g.boundaryCosine=g.boundaryFlux;return g;
}
MieOpticalTable table(){MieOpticalTable t;t.temperature={100,2000};t.diameter={1e-4,1e-2};t.mu={-1,1};t.band={1,1};
    t.absorption.assign(4,1);t.scattering.assign(4,.3);t.extinction.assign(4,1.3);t.phase.assign(8,1);return t;}
__global__ void update(int sample,const double* mass,double* tm,double* boundary,double* emissivity){
    if(threadIdx.x)return;
    tm[0]=mass[0]*(600+sample);tm[1]=mass[1]*(700+sample);
    boundary[1]=400+sample;boundary[2]=500+sample;emissivity[1]=.7+.01*sample;
}
__global__ void consume(RadiationMaterialDeviceView result,double* ledger){
    if(threadIdx.x)return;
    ledger[0]+=result.energy[0];ledger[1]+=result.energy[1];ledger[2]+=result.boundaryEnergy[0];ledger[3]+=result.boundaryEnergy[1];
}
int main(){
    auto g=geometry();auto t=table();RadiationSnapshot input;
    const double m0=std::ldexp(1.,-16),m1=std::ldexp(1.,-15),d0=std::ldexp(1.,-10),d1=std::ldexp(1.,-9);
    Buffer mass({m0,m1}),tm({m0*600,m1*700}),dm({m0*d0,m1*d1}),wall({0,400,500}),emissivity({0,.7,1});
    Buffer contact({0,.25,0}),scale({1,1,1}),ledger({0,0,0,0});
    cudaStream_t producer;cudaEvent_t ready;
    check(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking));check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&diagnostics),sizeof(*diagnostics)));check(cudaMemset(diagnostics,0,sizeof(*diagnostics)));
    input.wallTemperature={0,0};input.emissivity={0,1};input.residentWallTemperature={1,1};input.contactEmissivity={1,0};
    input.resident={true,2,1,2,mass.p,tm.p,dm.p,wall.p,contact.p,scale.p,1,100,producer,emissivity.p,diagnostics};
    RadiationControls controls;controls.parallelAngles=true;controls.sourceTolerance=1e-12;controls.sourceIterations=1000;
    DiscreteOrdinateRadiation radiation(g,t,controls);ParticleRadiationMaterial material(t);
    ParticleRadiationMaterialInput transaction;transaction.coupledBoundary={1,1};transaction.heatFactor=1000;
    transaction.elapsed=.01;transaction.minimumTemperature=100;transaction.maximumTemperature=2000;
    // Allocate/configure all workspaces before the measured advancement interval.
    radiation.solve(input,false,false);transaction.resident=radiation.deviceView();material.solve(transaction,false);material.collect();
    std::vector<double> expectedLedger(4,0);ParticleRadiationMaterialResult expected;RadiationResult expectedRadiation;
    for(int sample=1;sample<=3;++sample){
        DiscreteOrdinateRadiation reference(g,t,controls);ParticleRadiationMaterial referenceMaterial(t);
        RadiationSnapshot host;host.epsilon={m0,m1};host.temperature={double(600+sample),double(700+sample)};host.diameter={d0,d1};
        host.wallTemperature={double(400+sample),double(500+sample)};host.emissivity={(.7+.01*sample)*.75,1};
        expectedRadiation=reference.solve(host,false);
        auto request=transaction;request.resident=reference.deviceView();request.mass={m0,m1};expected=referenceMaterial.solve(request);
        for(int c=0;c<2;++c){expectedLedger[c]+=expected.energy[c];expectedLedger[c+2]+=expected.boundaryEnergy[c];}
    }
    check(cudaProfilerStart());
    for(int sample=1;sample<=3;++sample){
        update<<<1,1,0,producer>>>(sample,mass.p,tm.p,wall.p,emissivity.p);
        radiation.solve(input,false,false);transaction.resident=radiation.deviceView();const auto result=material.solve(transaction,false);
        check(cudaEventRecord(ready,static_cast<cudaStream_t>(result.resident.producerStream)));
        check(cudaStreamWaitEvent(producer,ready,0));consume<<<1,1,0,producer>>>(result.resident,ledger.p);
        check(cudaEventRecord(ready,producer));check(cudaStreamWaitEvent(static_cast<cudaStream_t>(result.resident.producerStream),ready,0));
    }
    check(cudaProfilerStop());
    const auto actualRadiation=radiation.collect();const auto actual=material.collect();std::vector<double> actualLedger(4);
    check(cudaMemcpy(actualLedger.data(),ledger.p,4*sizeof(double),cudaMemcpyDeviceToHost));
    same(expectedRadiation.incident,actualRadiation.incident,"resident DO intensity mismatch");same(expectedRadiation.boundaryFlux,actualRadiation.boundaryFlux,"resident wall flux mismatch");
    same(expected.temperature,actual.temperature,"resident material temperature mismatch");same(expected.energy,actual.energy,"resident material energy mismatch");
    same(expected.boundaryEnergy,actual.boundaryEnergy,"resident boundary energy mismatch");same(expectedLedger,actualLedger,"resident consumption sequence mismatch");
    // A bounded particle update may accept different heat. Pair the final wall
    // contribution on device and independently recompute the former host rule.
    Buffer acceptedHeat({.75*actual.energy[0],.75*actual.energy[1]});
    const double particleHeat=.75*actual.energy[0]+.75*actual.energy[1];
    double wallHeat=actual.escapedEnergy,gross=fabs(actual.escapedEnergy);
    for(double q:{.75*actual.energy[0],.75*actual.energy[1]})gross+=fabs(q);
    for(double q:actual.boundaryEnergy){wallHeat+=q;gross+=fabs(q);}
    const auto settlement=radiationMath::conservativeBoundaryEnergyScale(particleHeat,wallHeat,gross,1e-10,actual.radiativeEnergyMagnitude);
    material.settleAcceptedEnergy({2,acceptedHeat.p,producer});
    const auto settled=material.collectBoundaryEnergy();auto expectedBoundary=actual.boundaryEnergy;
    for(double& q:expectedBoundary)q*=settlement.scale;
    same(expectedBoundary,settled.boundaryEnergy,"resident accepted-heat boundary settlement mismatch");
    same({particleHeat,actual.escapedEnergy*settlement.scale},{settled.acceptedParticleEnergy,settled.escapedEnergy},"resident accepted heat summary mismatch");
    // An absent particle phase is represented without uploading a zero array.
    const auto noParticleRequest=material.solve(transaction);double noParticleWall=noParticleRequest.escapedEnergy;
    double noParticleGross=fabs(noParticleWall);
    for(double q:noParticleRequest.boundaryEnergy){noParticleWall+=q;noParticleGross+=fabs(q);}
    const auto noParticleScale=radiationMath::conservativeBoundaryEnergyScale(0,noParticleWall,noParticleGross,1e-10,noParticleRequest.radiativeEnergyMagnitude);
    material.settleAcceptedEnergy({2,nullptr,producer});const auto noParticle=material.collectBoundaryEnergy();
    auto noParticleExpected=noParticleRequest.boundaryEnergy;for(double& q:noParticleExpected)q*=noParticleScale.scale;
    same(noParticleExpected,noParticle.boundaryEnergy,"absent-particle boundary settlement mismatch");
    if(noParticle.acceptedParticleEnergy!=0)throw std::runtime_error("absent phase accepted particle heat");
    ugkp::diagnostics::DeviceContext report{};check(cudaMemcpy(&report,diagnostics,sizeof(report),cudaMemcpyDeviceToHost));
    if(report.fatalClaim)throw std::runtime_error("unexpected deferred radiation failure");
    // Deferred faults remain visible and suppress dependent physical updates.
    auto bad=g;bad.boundaryType[0]=RadiationBoundary::Fixed;bad.fixedIntensity[0]=-1e8;
    DiscreteOrdinateRadiation invalid(bad,t,controls);invalid.solve(input,false,false);
    transaction.resident=invalid.deviceView();const auto rejected=material.solve(transaction);
    check(cudaMemcpy(&report,diagnostics,sizeof(report),cudaMemcpyDeviceToHost));
    if(!report.fatalClaim||report.firstFatal.sourceSiteId!=34000)throw std::runtime_error("missing deferred radiation failure");
    for(double q:rejected.energy)if(q!=0)throw std::runtime_error("failed radiation committed particle energy");
    for(double q:rejected.boundaryEnergy)if(q!=0)throw std::runtime_error("failed radiation committed boundary energy");
    material.settleAcceptedEnergy({2,acceptedHeat.p,producer});
    const auto failedSettlement=material.collectBoundaryEnergy();
    for(double q:failedSettlement.boundaryEnergy)if(q!=0)throw std::runtime_error("failed radiation committed accepted-heat settlement");
    if(failedSettlement.acceptedParticleEnergy!=0||failedSettlement.escapedEnergy!=0)throw std::runtime_error("failed settlement published nonzero heat");
    bool caught=false;try{invalid.collect();}catch(const RadiationError&){caught=true;}
    if(!caught)throw std::runtime_error("explicit radiation publication lost native failure");
    check(cudaFree(diagnostics));check(cudaEventDestroy(ready));check(cudaStreamDestroy(producer));
    std::cout<<"PASS: device radiation/material/consumer chain, dynamic wall temperature and emissivity, deferred fault publication\n";
}
