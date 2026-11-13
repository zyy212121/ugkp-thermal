#ifndef UGKP_GPU_DISCRETE_ORDINATE_RADIATION_H
#define UGKP_GPU_DISCRETE_ORDINATE_RADIATION_H
#include "MieOpticalProperties.H"
#include "RadiationResidentView.H"
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
    // Direction-major arrays preserve the native geometry and face order.
    std::vector<double> internalFlux,transportDiagonal,boundaryFlux,boundaryCosine;
    std::vector<double> boundaryArea,diffuseDenominator,fixedIntensity;
};
struct RadiationControls {
    RadiationReal activeEpsilonMinimum=0,phaseTolerance=RadiationReal(1e-12),sourceTolerance=RadiationReal(1e-10);
    RadiationReal linearTolerance=RadiationReal(1e-12),linearRelativeTolerance=0,sourceRelaxation=1,small=RadiationReal(1e-15);
    int phaseIterations=200,sourceIterations=300,linearIterations=1000;
    // This retains the original distinction between frozen angular sources
    // and sequential angular Gauss-Seidel updates; it is not a case selector.
    bool parallelAngles=false;
    int maximumConcurrentAngles=0; // 0 retains all-direction concurrency for API callers.
    bool controlsAlreadyResolved=false; // Direct API callers retain precision defaults.
};
struct RadiationSnapshot {
    std::vector<double> epsilon,temperature,diameter,wallTemperature,emissivity;
    RadiationParticleDeviceView resident;
    std::vector<unsigned char> residentWallTemperature,contactEmissivity;
};
struct RadiationDirectionWarning {
    RadiationReal residual=0,limit=0;int linearIteration=0,sourceIteration=0;
    RadiationReal publishedResidual=0,publishedTarget=0;
};
struct RadiationResult {
    std::vector<RadiationReal> absorption,scattering,extinction,asymmetry,incident,power,boundaryFlux;
    std::vector<MiePhaseMetrics> phaseMetrics;
    std::vector<RadiationDirectionWarning> directionWarnings;
    RadiationReal maximumRelativeChange=0;
    int sourceIterations=0;
};
struct RadiationError:std::runtime_error {
    RadiationReal value;int iteration,cell,patch,face;
    RadiationError(const std::string& reason,RadiationReal v,int i,int c,int p,int f):
        std::runtime_error(reason),value(v),iteration(i),cell(c),patch(p),face(f){}
};
class DiscreteOrdinateRadiation {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    DiscreteOrdinateRadiation(const RadiationGeometry&,const MieOpticalTable&,const RadiationControls&);
    ~DiscreteOrdinateRadiation();
    RadiationResult solve(const RadiationSnapshot&,bool warmStart,bool exportFields=true);
    // Explicit output publication. Device consumers use deviceView() directly.
    RadiationResult collect(bool exportFields=true);
    // Same topology/angular quadrature: refresh metric/transport graphs while
    // retaining optical tables and angular intensities on the device.
    void updateGeometry(const RadiationGeometry&);
    // Explicit diagnostic export; ordinary solves retain angular fields on GPU.
    void downloadIntensities(std::vector<double>&)const;
    void uploadIntensities(const std::vector<double>&);
    RadiationDeviceView deviceView() const noexcept;
};
}}
#endif
