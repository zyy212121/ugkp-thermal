#ifndef UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#define UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#include "SolidThermalMath.H"
#include "ThermalInterfaceDeviceView.H"
#include "ScaledThermalSum.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct SolidThermalDeviceFields {
    const double* capacity=nullptr;
    const double* conductivity=nullptr;
    int count=0;
    void* readyStream=nullptr;
};
struct ThermalPropertyDiagnostic {int firstError=-1,firstExtrapolation=-1;ThermalPropertyStatus error,extrapolation;};
struct SolidThermalConvergence {
    ScaledThermalSum enthalpyChange,externalEnergy;
    double temperatureChange=0,surfaceChange=0,surfaceFluxChange=0;
};
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
    double relaxResident(double* deviceCurrent,const double* devicePrevious,int count,bool havePrevious,
        double relaxation,double minimum,double maximum,double small,void* ownerStream);
    void setCellAnchors(const double* temperatures,int count);
    void setResidentCellAnchors(const double* temperatures,int count,void* ownerStream);
    const double* deviceCellAnchors() const;
    void setConvergenceBoundary(const std::vector<int>& surfacePropertyIndex,const std::vector<double>& initialTemperature);
    ThermalPropertyDiagnostic diagnostic() const;
    SolidThermalConvergence convergence(int cells,int surfaceProperties,const double* volume,
        const double* boundaryFlux,const double* boundaryTemperature,const double* gradient,
        const double* energy,const double* area,double interval,void* ownerStream);
    void evaluateResidentCells(const std::vector<ThermalPropertyQuery>& boundaryQueries,
        const double* deviceTemperature,const double* devicePrevious,int cells,unsigned fields,bool cellsFirst,
        void* ownerStream,std::vector<ThermalPropertyResult>& results,
        const double* deviceBoundaryTemperature=nullptr,const std::vector<int>* boundarySources=nullptr,bool download=true);
    SolidThermalDeviceFields deviceFields() const;
    ThermalSolidContactDeviceView contactState(const std::vector<int>& cells,const double* temperature,void* stream,
        std::vector<ThermalPropertyResult>& status);
    ThermalInterfaceStateDeviceView interfaceState(const std::vector<int>& boundaryFaces,
        const std::vector<int>& contactEnabled,const double* boundaryTemperature,void* ownerStream,
        std::vector<ThermalPropertyResult>& status);
    void evaluate(const std::vector<ThermalPropertyQuery>& queries,std::vector<ThermalPropertyResult>& results,
        const double* deviceTemperature=nullptr,int cellOffset=0,int cellCount=0,void* ownerStream=nullptr);
};
struct SolidContactConvergence {double temperatureChange=0,energyDifference=0,energyScale=1;};
class SolidThermalContact {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    SolidThermalContact();~SolidThermalContact();
    void begin(const std::vector<int>& primaryForAuxiliary,const std::vector<double>& primaryDelta,
        const std::vector<double>& auxiliaryDelta,const std::vector<double>& auxiliaryArea);
    void evaluate(const ThermalSolidContactDeviceView& primary,const ThermalSolidContactDeviceView& auxiliary,double interval,
        ThermalInterfaceEnergyDeviceView& primaryEnergy,ThermalInterfaceEnergyDeviceView& auxiliaryEnergy,
        std::vector<double>& primaryAudit,std::vector<double>& auxiliaryAudit);
    SolidContactConvergence convergence(const ThermalSolidContactDeviceView& primary,const ThermalSolidContactDeviceView& auxiliary);
};
}}
#endif
