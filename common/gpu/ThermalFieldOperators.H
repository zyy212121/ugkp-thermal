#ifndef UGKP_GPU_THERMAL_FIELD_OPERATORS_H
#define UGKP_GPU_THERMAL_FIELD_OPERATORS_H
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct ThermalVector {double x=0,y=0,z=0;};
struct ScalarFaceInterpolation {double owner=0,neighbour=0,weight=0;int kind=0;};
struct VectorFaceInterpolation {ThermalVector owner,neighbour,direction;double weight=0;int kind=0;};
struct GradientBoundaryCorrection {ThermalVector gradient,area;double magnitude=0,normalGradient=0;};
struct GaussGradientInput {
    int cells=0,internalFaces=0;
    std::vector<int> owner,neighbour;
    std::vector<double> faceValue,volume;
    std::vector<ThermalVector> area;
};
class ThermalFieldOperators {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ThermalFieldOperators();~ThermalFieldOperators();
    void interpolate(const std::vector<ScalarFaceInterpolation>&,bool harmonic,std::vector<double>&);
    void dotInterpolate(const std::vector<VectorFaceInterpolation>&,std::vector<double>&);
    void gradient(const GaussGradientInput&,std::vector<ThermalVector>&);
    void correctGradient(const std::vector<GradientBoundaryCorrection>&,std::vector<ThermalVector>&);
};
}}
#endif
