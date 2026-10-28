#ifndef UGKP_GPU_MIE_OPTICAL_PROPERTIES_H
#define UGKP_GPU_MIE_OPTICAL_PROPERTIES_H
#include "MieOpticalMath.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct MieOpticalTable {
    std::vector<double> temperature,diameter,mu,band,absorption,scattering,extinction,phase;
    MieOpticalTableView view()const{return {int(temperature.size()),int(diameter.size()),int(mu.size()),
        temperature.data(),diameter.data(),mu.data(),band.data(),absorption.data(),scattering.data(),extinction.data(),phase.data()};}
};
struct MiePhaseMetrics {double rowError=0,columnError=0,scalingCorrection=0;int iterations=0;};
class MieOpticalProperties {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit MieOpticalProperties(const MieOpticalTable&);
    ~MieOpticalProperties();
    // Quadratic angular work uses linear per-cell storage: retain balanced
    // factors and reconstruct each probability with the original product order.
    void balance(const std::vector<MieOpticalQuery>&,const std::vector<double>& weights,
        const std::vector<double>& directionCosines,double tolerance,int maximumIterations,
        std::vector<double>& scaling,std::vector<MiePhaseMetrics>&);
    MieOpticalTableView deviceTable() const noexcept;
    void evaluate(const std::vector<MieOpticalQuery>&,std::vector<MieOpticalResult>&);
};
}}
#endif
