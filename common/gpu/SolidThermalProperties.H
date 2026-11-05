#ifndef UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#define UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#include "SolidThermalMath.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
class SolidThermalProperties {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    SolidThermalProperties(const ThermalPropertyTable* tables,const double* knots,int count);
    ~SolidThermalProperties();
    SolidThermalProperties(const SolidThermalProperties&)=delete;
    SolidThermalProperties& operator=(const SolidThermalProperties&)=delete;
    // Residual history belongs to this region; havePrevious=false starts a window.
    double relax(double* current,const double* previous,int count,bool havePrevious,
        double relaxation,double minimum,double maximum,double small);
    void evaluate(const std::vector<ThermalPropertyQuery>& queries,std::vector<ThermalPropertyResult>& results);
};
}}
#endif
