#pragma once
#include <vector>
#include "../GpuPrecisionTypes.H"
namespace ugkp { namespace thermal {


struct ParticleInletConfiguration {
    std::vector<int> faces;
    std::vector<GpuTime> times;
    std::vector<GpuReal> volumeFraction;
    GpuReal temperature=300,theta=0;
    bool velocitySpecified=false;
    GpuReal velocity[3]{};
};
template<class Archive,class Config>
void visitParticleInletConfiguration(Archive& a,Config& c){
    a(c.faces);a(c.times);a(c.volumeFraction);a(c.temperature);a(c.theta);
    a(c.velocitySpecified);a(c.velocity);
}
}}
