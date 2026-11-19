#ifndef UGKP_GPU_DISCRETE_ORDINATE_RADIATION_H
#define UGKP_GPU_DISCRETE_ORDINATE_RADIATION_H
#include "MieOpticalProperties.H"
#include "RadiationResidentView.H"
#include "ExecutionGraph.H"
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu {
enum class RadiationBoundary {Diffuse,Black,Zero,Fixed,Specular};
struct RadiationGeometry {
    int cells=0,directions=0;
    std::vector<int> owner,neighbour,cellOffsets,cellFaces,boundaryOwner;
    std::vector<int> boundaryPatch,boundaryFace,reflectedDirection;
    std::vector<int> reflectionOffsets,reflectionDonors;
    std::vector<double> reflectionWeights;
    std::vector<RadiationBoundary> boundaryType;
    std::vector<double> volume,weights,directionCosines;

    std::vector<double> internalFlux,transportDiagonal,boundaryFlux,boundaryCosine;
    std::vector<double> boundaryArea,diffuseDenominator,fixedIntensity;
};
struct RadiationGeometryUpdate {
    unsigned long long geometryRevision=0,executionRevision=0;
    int rebuiltDirections=0,updatedDirectionValues=0;
    bool rebuiltStructure=false,updatedReflection=false,rebuiltSourceGraph=false;
};
struct RadiationControls {
    RadiationReal activeEpsilonMinimum=0,phaseTolerance=RadiationReal(1e-12),sourceTolerance=RadiationReal(1e-10);
    RadiationReal linearTolerance=RadiationReal(1e-12),linearRelativeTolerance=0,sourceRelaxation=1,small=RadiationReal(1e-15);
    int phaseIterations=200,sourceIterations=500,linearIterations=1000;


    bool parallelAngles=false;
    int maximumConcurrentAngles=0;
    bool controlsAlreadyResolved=false;
};
struct RadiationSnapshot {
    std::vector<double> epsilon,temperature,diameter,wallTemperature,emissivity;
    RadiationParticleDeviceView resident;
    RadiationThermalCouplingDeviceView thermalCoupling;
    std::vector<unsigned char> residentWallTemperature,contactEmissivity;
};
struct RadiationDirectionWarning {
    RadiationReal residual=0,limit=0;int linearIteration=0,sourceIteration=0;
    RadiationReal publishedResidual=0,publishedTarget=0;int stopReason=0;RadiationReal linearRelativeResidual=0,linearRelativeLimit=0;
};
struct RadiationResult {
    std::vector<RadiationReal> absorption,scattering,extinction,asymmetry,incident,power,boundaryFlux;
    std::vector<MiePhaseMetrics> phaseMetrics;
    std::vector<RadiationDirectionWarning> directionWarnings;
    RadiationReal maximumRelativeChange=0;
    RadiationReal materialRelativeResidual=0,transportRelativeResidual=0;
    RadiationReal materialResidualTarget=0,transportResidualTarget=0;
    int sourceStop=0;unsigned long long matvecs=0,workLimit=0;
    double wideResidual=0,arithmeticDelta=0;
    int sourceIterations=0,lastLinearIterations=0,lastLinearStopReason=0;
    unsigned long long linearSolves=0,totalLinearIterations=0;
    RadiationReal sourceResidualTarget=0,lastLinearResidual=0;
    bool sourceConverged=false,lastLinearConverged=false;
};
struct RadiationError:std::runtime_error {
    RadiationReal value;int iteration,cell,patch,face;
    RadiationError(const std::string& reason,RadiationReal v,int i,int c,int p,int f):
        std::runtime_error(reason),value(v),iteration(i),cell(c),patch(p),face(f){}
};


struct RadiationThermalIterationDeviceView {
    int cells=0;
    const RadiationReal *incident=nullptr,*power=nullptr,*volume=nullptr;
    const RadiationReal *epsilon=nullptr,*mass=nullptr,*diameter=nullptr;
    const RadiationThermalCouplingDeviceView* coupling=nullptr;
};
using RadiationThermalIterationHook=std::function<ExecutionGraphNode(
    ExecutionGraph,ExecutionGraphNode,void*,const RadiationThermalIterationDeviceView&)>;
class RadiationWarmStart;
class DiscreteOrdinateRadiation {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    DiscreteOrdinateRadiation(const RadiationGeometry&,const MieOpticalTable&,const RadiationControls&);
    ~DiscreteOrdinateRadiation();


    void enqueue(const RadiationSnapshot&,bool warmStart,bool publishDiagnostics=true);

    void awaitCompletion() const;


    void prepareParentInputs(const RadiationSnapshot&,bool warmStart);
    ExecutionGraphNode appendInputsTo(ExecutionGraph,ExecutionGraphNode dependency=nullptr);


    ExecutionGraphNode appendSourceTo(ExecutionGraph,ExecutionGraphNode dependency=nullptr);
    void resetContinuation(void* stream);
    const int* continuationStage()const noexcept;
    const int* capacityPendingDevice()const noexcept;
    bool servicePendingCapacity();
    void acceptExternalCompletion(const std::shared_future<void>&);
    void acceptExternalExecution(void* producerStream);
    ExecutionGraphNode appendDiagnosticsTo(ExecutionGraph,ExecutionGraphNode dependency);
    void* producerStream()const noexcept;
    RadiationResult solve(const RadiationSnapshot&,bool warmStart,bool exportFields=true);

    RadiationResult collect(bool exportFields=true);



    void updateGeometry(const RadiationGeometry&);
    RadiationGeometryUpdate lastGeometryUpdate()const noexcept;
    unsigned long long executionRevision()const noexcept;

    void freezeIntensities(RadiationWarmStart&, void* nextWriter=nullptr) const;
    void downloadIntensities(std::vector<double>&)const;
    void uploadIntensities(const std::vector<double>&);



    RadiationDeviceView deviceView() const noexcept;
    void setThermalIterationHook(RadiationThermalIterationHook, unsigned long long owner=0, unsigned long long revision=0);
};
}}
#endif
