#ifndef UGKP_GPU_THERMAL_FIELD_OPERATORS_H
#define UGKP_GPU_THERMAL_FIELD_OPERATORS_H
#include "../GpuPrecisionTypes.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct ThermalVector {GpuReal x=0,y=0,z=0;};
struct ScalarFaceInterpolation {GpuReal owner=0,neighbour=0,weight=0;int kind=0;};
#ifdef __CUDACC__
#define UGKP_THERMAL_INTERPOLATION_HD __host__ __device__
#else
#define UGKP_THERMAL_INTERPOLATION_HD
#endif
UGKP_THERMAL_INTERPOLATION_HD inline GpuReal thermalLinearInterpolation(GpuReal a,GpuReal b,GpuReal w,int kind){
    return kind==0?w*(a-b)+b:kind==1?w*a+(1-w)*b:a;
}
UGKP_THERMAL_INTERPOLATION_HD inline GpuReal thermalScalarInterpolation(const ScalarFaceInterpolation& f,bool harmonic){
    return harmonic&&f.kind!=2?(f.owner>0&&f.neighbour>0?1/((1-f.weight)/f.owner+f.weight/f.neighbour):0)
        :thermalLinearInterpolation(f.owner,f.neighbour,f.weight,f.kind);
}
#undef UGKP_THERMAL_INTERPOLATION_HD
struct VectorFaceInterpolation {ThermalVector owner,neighbour,direction;GpuReal weight=0;int kind=0;};
struct GradientBoundaryCorrection {ThermalVector gradient,area;GpuReal magnitude=0,normalGradient=0;};
struct GaussGradientInput {
    int cells=0,internalFaces=0;
    std::vector<int> owner,neighbour;
    std::vector<GpuReal> faceValue,volume;
    std::vector<ThermalVector> area;
};
class ThermalFieldOperators {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ThermalFieldOperators();~ThermalFieldOperators();
    void interpolate(const std::vector<ScalarFaceInterpolation>&,bool harmonic,std::vector<GpuReal>&);
    void dotInterpolate(const std::vector<VectorFaceInterpolation>&,std::vector<GpuReal>&);
    void gradient(const GaussGradientInput&,std::vector<ThermalVector>&);
    void correctGradient(const std::vector<GradientBoundaryCorrection>&,std::vector<ThermalVector>&);
};
}}
#endif
