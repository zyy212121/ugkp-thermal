#ifndef UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#define UGKP_GPU_SOLID_THERMAL_PROPERTIES_H
#include "SolidThermalMath.H"
#include "ThermalInterfaceDeviceView.H"
#include "ScaledThermalSum.H"
#include "ExecutionGraph.H"
#include <memory>
#include <vector>
namespace ugkp { namespace diagnostics { struct DeviceContext; } namespace gpu {
struct SolidThermalDeviceFields {
    const double* capacity=nullptr;
    const double* conductivity=nullptr;
    int count=0;
    void* readyStream=nullptr;
};
struct ThermalPropertyDiagnostic {int firstError=-1,firstExtrapolation=-1;ThermalPropertyStatus error,extrapolation;};
struct SolidThermalConvergence {
    ScaledThermalSum enthalpyChange,externalEnergy,interfaceEnergy;
    double temperatureChange=0,surfaceChange=0,surfaceFluxChange=0;
};
struct SolidThermalIterationControls {
    int maximumIterations=1;
    bool aitken=false;
    double relativeTolerance=0,energyAbsoluteTolerance=0,energyRelativeTolerance=0;
    double initialRelaxation=1,minimumRelaxation=0,maximumRelaxation=1,small=1e-300;
};
struct SolidThermalIteration {
    int iteration=0;
    bool converged=false,valid=true;
    double relaxation=1,interval=0;
    SolidThermalConvergence convergence;
    ScaledThermalSum energyResidual,energyLimit;
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
    // Compile query layouts during initialization. All later temperatures,
    // caloric anchors and fixed-point history are read directly on device.
    int prepareResidentEvaluation(const std::vector<ThermalPropertyQuery>& boundaryQueries,
        const std::vector<int>& boundarySources,int cells,unsigned fields,bool cellsFirst,int plan=-1);
    void enqueueResidentEvaluation(int plan,const double* temperature,const double* previous,
        const double* boundaryTemperature,void* executionStream,const SolidThermalIteration* iteration=nullptr);
    void prepareResidentIteration(int cells);
    void enqueueResidentRelaxation(double* current,const double* previous,int count,
        SolidThermalIteration*,const SolidThermalIterationControls&,void* executionStream);
    const SolidThermalConvergence* enqueueResidentConvergence(int cells,int surfaceProperties,const double* volume,
        const double* boundaryFlux,const double* boundaryTemperature,const double* gradient,
        const double* energy,const double* area,double interval,void* executionStream,const double* deviceInterval=nullptr);
    void enqueueResidentDiagnostic(diagnostics::DeviceContext*,void* executionStream);
    void enqueueResidentAnchors(const double* temperature,const double* boundaryTemperature,int cells,void* executionStream);
    ThermalSolidContactDeviceView prepareResidentContact(const std::vector<int>& cells);
    ThermalSolidContactDeviceView residentContact() const;
    void enqueueResidentContact(const double* temperature,diagnostics::DeviceContext*,void* executionStream);
    ThermalSolidContactDeviceView contactState(const std::vector<int>& cells,const double* temperature,void* stream,
        std::vector<ThermalPropertyResult>& status);
    ThermalInterfaceStateDeviceView interfaceState(const std::vector<int>& boundaryFaces,
        const std::vector<int>& contactEnabled,const double* boundaryTemperature,void* ownerStream,
        std::vector<ThermalPropertyResult>& status,bool includeEmissivity=false,
        diagnostics::DeviceContext* diagnosticContext=nullptr,void* diagnosticStream=nullptr);
    void evaluate(const std::vector<ThermalPropertyQuery>& queries,std::vector<ThermalPropertyResult>& results,
        const double* deviceTemperature=nullptr,int cellOffset=0,int cellCount=0,void* ownerStream=nullptr);
};
class ImplicitDiffusion;
class SparseConjugateGradient;
struct SolidThermalResidentProblem {
    int cells=0,surfaceProperties=0,coefficientPlan=0,convergencePlan=0;
    double interval=0;
    const double* deviceInterval=nullptr;
    SolidThermalIterationControls controls;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
// The prepared owners outlive this execution graph. Reconfiguration of their
// topology, policies or buffer addresses requires rebuilding this graph.
class SolidThermalIterationGraph {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    SolidThermalIterationGraph(SolidThermalProperties&,ImplicitDiffusion&,SparseConjugateGradient&,const SolidThermalResidentProblem&);
    ~SolidThermalIterationGraph();
    void enqueue(void* producerStream);
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode dependency=nullptr,bool contactCandidate=false);
    void prepareContactWindow(const std::vector<int>& cells);
    void enqueueContactWindowStart(void* executionStream);
    void enqueueContactState(void* executionStream);
    ThermalSolidContactDeviceView contactState() const;
    void acceptExternalExecution(void* producerStream);
    diagnostics::DeviceContext* diagnosticsContext() const;
    const SolidThermalIteration* deviceReport()const;
    void* producerStream()const;
    SolidThermalIteration collect(diagnostics::DeviceContext* diagnosticOutput=nullptr);
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
    ThermalInterfaceEnergyDeviceView primaryEnergy(const double* interval=nullptr) const;
    ThermalInterfaceEnergyDeviceView auxiliaryEnergy(const double* interval=nullptr) const;
    void enqueueWindowStart(void* executionStream);
    void enqueueEvaluate(const ThermalSolidContactDeviceView&,const ThermalSolidContactDeviceView&,
        double interval,const double* deviceInterval,void* executionStream);
    const SolidContactConvergence* enqueueConvergence(const ThermalSolidContactDeviceView&,
        const ThermalSolidContactDeviceView&,void* executionStream);
};
struct SolidContactIterationControls {
    int maximumIterations=1;double temperatureTolerance=0,energyTolerance=0;
};
struct SolidContactIteration {
    int iteration=0;bool valid=true,converged=false;SolidContactConvergence convergence;
};
class SolidContactIterationGraph {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    SolidContactIterationGraph(SolidThermalIterationGraph& primary,SolidThermalIterationGraph& auxiliary,
        SolidThermalContact&,const SolidContactIterationControls&,double interval,const double* deviceInterval);
    ~SolidContactIterationGraph();
    void enqueue(void* producerStream);
    SolidContactIteration collect();
};
}}
#endif
