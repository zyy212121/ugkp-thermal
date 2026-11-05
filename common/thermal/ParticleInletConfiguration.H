#pragma once
#include <vector>
namespace ugkp { namespace thermal {
// One resident inlet program. Faces and schedules are resolved by the mesh
// adapter once; the common injection kernel samples gas boundary velocity.
struct ParticleInletConfiguration {
    std::vector<int> faces;
    std::vector<double> times,volumeFraction;
    double temperature=300,theta=0;
    bool velocitySpecified=false;
    double velocity[3]{};
};
template<class Archive,class Config>
void visitParticleInletConfiguration(Archive& a,Config& c){
    a(c.faces);a(c.times);a(c.volumeFraction);a(c.temperature);a(c.theta);
    a(c.velocitySpecified);a(c.velocity);
}
}}
