#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/ParticleRadiationMaterial.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace ugkp::gpu;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    double* p=nullptr;
    explicit Device(const std::vector<double>& v){check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(std::size_t(1),v.size())*sizeof(double)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(double),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
};
void same(const std::vector<double>& a,const std::vector<double>& b,const char* field){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double)))throw std::runtime_error(field);
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
void dump(const RadiationResult& r,const std::vector<double>& intensity,const char* mode,int sample){
    std::cout<<mode<<' '<<sample<<' '<<r.sourceIterations<<' '<<std::hexfloat<<r.maximumRelativeChange;
    for(const auto* values:{&intensity,&r.incident,&r.power,&r.boundaryFlux})for(double value:*values)std::cout<<' '<<value;
    std::cout<<'\n'<<std::defaultfloat;
}
int main(){
    std::cout<<std::unitbuf;auto g=geometry();auto t=table();RadiationSnapshot s;
    s.epsilon={std::ldexp(1.,-16),std::ldexp(1.,-15)};s.temperature={600,700};s.diameter={std::ldexp(1.,-10),std::ldexp(1.,-9)};
    s.wallTemperature={400,500};s.emissivity={.7,1};
    for(bool parallel:{false,true}){
        RadiationControls c;c.parallelAngles=parallel;c.maximumConcurrentAngles=2;c.sourceTolerance=1e-12;c.sourceIterations=1000;
        DiscreteOrdinateRadiation host(g,t,c);
#ifndef UGKP_RADIATION_LEGACY_REFERENCE
        DiscreteOrdinateRadiation resident(g,t,c);
        Device mass(s.epsilon),tm({s.epsilon[0]*600,s.epsilon[1]*700}),dm({s.epsilon[0]*s.diameter[0],s.epsilon[1]*s.diameter[1]});
        // One internal-face slot precedes the boundary face values.
        Device boundary({0,400,500}),contact({0,.25,0}),scale({1,1,1});
        auto input=s;input.epsilon.clear();input.temperature.clear();input.diameter.clear();
        input.resident={true,2,1,2,mass.p,tm.p,dm.p,boundary.p,contact.p,scale.p,1,100,nullptr};
        input.residentWallTemperature={1,1};input.contactEmissivity={1,0};
        // Host counterpart applies the same existing exposed-area law.
        auto exposed=s;exposed.emissivity[0]*=.75;
#endif
        for(int sample=0;sample<3;++sample){
            if(sample==2){auto moved=g;moved.volume={1.25,.75};moved.internalFlux={.8,-.8};
                moved.transportDiagonal={.8,1,1,.8};host.updateGeometry(moved);
#ifndef UGKP_RADIATION_LEGACY_REFERENCE
                resident.updateGeometry(moved);exposed.epsilon={s.epsilon[0]/1.25,s.epsilon[1]/.75};
#endif
            }
            auto out=host.solve(s,sample>0);std::vector<double> intensities;host.downloadIntensities(intensities);dump(out,intensities,parallel?"parallel":"serial",sample);
#ifndef UGKP_RADIATION_LEGACY_REFERENCE
            // A separate host-input workspace makes the resident source and
            // contact coverage comparison independent of the dump/oracle state.
            DiscreteOrdinateRadiation equivalent(sample==2?([&]{auto m=g;m.volume={1.25,.75};m.internalFlux={.8,-.8};m.transportDiagonal={.8,1,1,.8};return m;}()):g,t,c);
            auto expected=equivalent.solve(exposed,false);auto actual=resident.solve(input,false);
            same(expected.incident,actual.incident,"resident incident");same(expected.power,actual.power,"resident power");same(expected.boundaryFlux,actual.boundaryFlux,"resident boundary flux");
            if(expected.sourceIterations!=actual.sourceIterations||expected.maximumRelativeChange!=actual.maximumRelativeChange)throw std::runtime_error("resident convergence");
            ParticleRadiationMaterial material(t);ParticleRadiationMaterialInput a;a.epsilon=exposed.epsilon;a.mass=s.epsilon;a.temperature=s.temperature;a.diameter=s.diameter;
            a.incident=expected.incident;a.volume=sample==2?std::vector<double>{1.25,.75}:g.volume;a.boundaryArea=g.boundaryArea;a.boundaryFlux=expected.boundaryFlux;
            a.coupledBoundary={1,1};a.heatFactor=1000;a.elapsed=.01;a.minimumTemperature=100;a.maximumTemperature=2000;
            auto h=material.solve(a);a.resident=resident.deviceView();auto v=material.solve(a);
            same(h.temperature,v.temperature,"resident material temperature");same(h.energy,v.energy,"resident material energy");same(h.boundaryEnergy,v.boundaryEnergy,"resident material boundary");
            if(h.escapedEnergy!=v.escapedEnergy||h.radiativeEnergyMagnitude!=v.radiativeEnergyMagnitude)throw std::runtime_error("resident material totals");
#endif
        }
    }
    for(int budget:{0,1,2}){
        RadiationControls c;c.sourceIterations=budget;c.sourceTolerance=0;c.linearIterations=0;c.linearTolerance=0;
        DiscreteOrdinateRadiation solver(g,t,c);auto out=solver.solve(s,false);std::vector<double> intensity;solver.downloadIntensities(intensity);
        dump(out,intensity,"budget",budget);
        for(const auto& w:out.directionWarnings)std::cout<<"warning "<<w.linearIteration<<' '<<w.sourceIteration<<' '<<std::hexfloat<<w.residual<<' '<<w.limit<<'\n'<<std::defaultfloat;
    }
    {
        RadiationGeometry empty;empty.cellOffsets={0};RadiationSnapshot snapshot;
        DiscreteOrdinateRadiation solver(empty,t,{});auto out=solver.solve(snapshot,false);
        if(!out.power.empty()||!out.incident.empty()||!out.boundaryFlux.empty()||out.sourceIterations!=1||out.maximumRelativeChange!=0)
            throw std::runtime_error("empty radiation domain");
        std::cout<<"empty "<<out.sourceIterations<<' '<<out.maximumRelativeChange<<'\n';
    }
    {
        auto closed=g;closed.boundaryOwner.clear();closed.boundaryPatch.clear();closed.boundaryFace.clear();closed.boundaryType.clear();
        closed.boundaryArea.clear();closed.diffuseDenominator.clear();closed.fixedIntensity.clear();closed.boundaryFlux.clear();closed.boundaryCosine.clear();
        closed.transportDiagonal={1,0,0,1};auto snapshot=s;snapshot.wallTemperature.clear();snapshot.emissivity.clear();
        DiscreteOrdinateRadiation solver(closed,t,{});auto out=solver.solve(snapshot,false);std::vector<double> intensity;solver.downloadIntensities(intensity);
        dump(out,intensity,"empty-boundary",0);
        closed.directions=0;closed.weights.clear();closed.directionCosines.clear();closed.internalFlux.clear();closed.transportDiagonal.clear();
        DiscreteOrdinateRadiation zeroDirections(closed,t,{});out=zeroDirections.solve(snapshot,false);zeroDirections.downloadIntensities(intensity);
        dump(out,intensity,"zero-directions",0);
    }
    // A nonphysical incoming intensity must still yield an explicit location.
    auto bad=g;bad.boundaryType[0]=RadiationBoundary::Fixed;bad.fixedIntensity[0]=-1e8;
    try{DiscreteOrdinateRadiation solver(bad,t,{});solver.solve(s,false);throw std::runtime_error("missing radiation failure");}
    catch(const RadiationError& e){std::cout<<"failure "<<e.iteration<<' '<<e.cell<<' '<<e.patch<<' '<<e.face<<' '<<e.what()<<'\n';}
}
