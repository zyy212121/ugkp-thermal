#include "gpu/SolidThermalProperties.H"
#include "gpu/ImplicitDiffusion.H"
#include "Diagnostics.H"
#include <cuda_runtime.h>
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ugkp::gpu;
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
__global__ void setInterval(double* interval,int window){if(!threadIdx.x)*interval=.1*(window+1);}
__global__ void saveTemperatures(const double* primary,const double* auxiliary,double* result,int window){
    if(!threadIdx.x){result[2*window]=*primary;result[2*window+1]=*auxiliary;}
}
struct Region {
    std::unique_ptr<SolidThermalProperties> properties;
    ImplicitDiffusion diffusion;SparseConjugateGradient sparse;
    std::unique_ptr<SolidThermalIterationGraph> graph;
    Region(double initial,double capacity,double conductivity,const std::vector<double>& area,
        const ThermalInterfaceEnergyDeviceView& energy,ugkp::diagnostics::DeviceContext* context,bool aitken){
        ThermalPropertyTable tables[4];for(auto& t:tables){t.constant=true;t.constantValue=1;}
        tables[1].constantValue=capacity;tables[2].constantValue=conductivity;
        properties.reset(new SolidThermalProperties(tables,nullptr,0));
        diffusion.uploadSolution(&initial,1);auto stream=diffusion.deviceValues().readyStream;const int nb=area.size();
        properties->setResidentCellAnchors(diffusion.deviceTemperature(),1,stream);
        properties->setConvergenceBoundary(std::vector<int>(nb,-1),std::vector<double>(nb,initial));
        std::vector<ThermalPropertyQuery> queries(nb,{initial,initial});std::vector<int> sources(nb,0),faces(nb);
        for(int i=0;i<nb;++i)faces[i]=i;
        const int coefficients=properties->prepareResidentEvaluation(queries,sources,1,ThermalCapacity|ThermalConductivity,true);
        const int convergence=properties->prepareResidentEvaluation({}, {},1,ThermalEnergy|ThermalTemperatureChange,false);
        properties->prepareResidentIteration(1);std::vector<ThermalPropertyResult> ignored;
        properties->evaluateResidentCells(queries,diffusion.deviceTemperature(),nullptr,1,ThermalCapacity|ThermalConductivity,true,stream,ignored);
        const auto fields=properties->deviceFields();DiffusionInput in;in.cells=1;in.owner.assign(nb,0);in.volume={1};in.area=area;in.delta.assign(nb,1);
        in.deviceTemperature=diffusion.deviceTemperature();in.deviceConductivity=fields.conductivity;in.deviceBoundaryConductivity=fields.conductivity+1;in.conductivityStream=fields.readyStream;
        in.conductivityWeights.assign(nb,1);in.conductivityBoundary.assign(nb,conductivity);in.conductivityNeighbour.assign(nb,0);
        in.conductivityCoupled.assign(nb,0);in.conductivityAdjacentCells.assign(nb,-1);in.conductivityJump.assign(nb,0);
        in.gradientInternal.assign(nb,0);in.gradientBoundary.assign(nb,0);in.gradientWeights.assign(nb,1);in.gradientNeighbour.assign(nb,-1);
        in.valueInternal.assign(nb,1);in.valueBoundary.assign(nb,0);in.normalInternal.assign(nb,0);in.normalBoundary.assign(nb,0);
        in.jump.assign(nb,0);in.boundaryValueDelta.assign(nb,1);
        diffusion.setInterfaceEnergy(faces,energy);diffusion.prescribeBoundaryEnergy(std::vector<double>(nb,0),std::vector<int>(nb,1),.1,fields.conductivity+1,fields.readyStream);
        DiffusionTransientInput transient;transient.dt=.1;transient.sourceFromCapacity=true;transient.deviceCapacity=fields.capacity;
        transient.deviceRhs=properties->deviceCellAnchors();transient.readyStream=fields.readyStream;diffusion.assembleResident(in,&transient);
        std::vector<double> unused;diffusion.flux(std::vector<double>(nb,0),std::vector<int>(nb,0),std::vector<int>(nb,-1),std::vector<double>(nb,0),unused,true,false);
        SparseLduView topology;topology.rows=1;topology.boundaryFaces=nb;topology.boundaryOwner=in.owner.data();topology.boundaryNeighbour=in.gradientNeighbour.data();
        ConjugateGradientControls linear;linear.tolerance=1e-13;linear.maximumIterations=100;sparse.configureResidentLocalLdu(topology,diffusion.deviceValues(),linear);
        SolidThermalResidentProblem problem;problem.cells=1;problem.coefficientPlan=coefficients;problem.convergencePlan=convergence;problem.interval=.1;problem.deviceInterval=energy.interval;problem.diagnostics=context;
        problem.controls.maximumIterations=100;problem.controls.relativeTolerance=1e-11;problem.controls.energyAbsoluteTolerance=1e-9;problem.controls.energyRelativeTolerance=1e-12;
        problem.controls.aitken=aitken;problem.controls.initialRelaxation=.5;problem.controls.minimumRelaxation=.1;problem.controls.maximumRelaxation=1;
        graph.reset(new SolidThermalIterationGraph(*properties,diffusion,sparse,problem));graph->prepareContactWindow(std::vector<int>(nb,0));
    }
};
int main(){
    for(int scenario=0;scenario<4;++scenario){
        const bool aitken=scenario==1,limited=scenario==2,reverse=scenario==3;
        double *interval,*temperatures;ugkp::diagnostics::DeviceContext* context;
        check(cudaMalloc(&interval,sizeof(double)));check(cudaMalloc(&temperatures,6*sizeof(double)));check(cudaMalloc(&context,sizeof(*context)));check(cudaMemset(context,0,sizeof(*context)));
        {
            SolidThermalContact contact;contact.begin({0,0},{1},{1,1},{.25,.75});
            Region primary(reverse?300:500,2,1,{1},contact.primaryEnergy(interval),context,aitken);
            Region auxiliary(reverse?500:300,3,2,{.25,.75},contact.auxiliaryEnergy(interval),context,aitken);
            SolidContactIterationControls controls{limited?1:100,1e-11,1e-11};
            SolidContactIterationGraph graph(*primary.graph,*auxiliary.graph,contact,controls,.1,interval);
            check(cudaDeviceSynchronize());if(!scenario)check(cudaProfilerStart());
            for(int window=0;window<3;++window){setInterval<<<1,1>>>(interval,window);graph.enqueue(nullptr);
                saveTemperatures<<<1,1>>>(primary.diffusion.deviceTemperature(),auxiliary.diffusion.deviceTemperature(),temperatures,window);}
            if(!scenario)check(cudaProfilerStop());
            const auto report=graph.collect();double observed[6];check(cudaMemcpy(observed,temperatures,sizeof(observed),cudaMemcpyDeviceToHost));
            ugkp::diagnostics::DeviceContext status{};check(cudaMemcpy(&status,context,sizeof(status),cudaMemcpyDeviceToHost));
            if(!report.valid||report.converged==limited||ugkp::diagnostics::hasFatal(&status))return 1;
            double p=reverse?300:500,a=reverse?500:300,maxError=0,maxEnergy=0;
            for(int window=0;window<3;++window){
                const double dt=.1*(window+1),conductance=2./3.;
                const double energy=conductance*dt*(p-a)/(limited?1:1+conductance*dt*(.5+1./3.));
                p-=energy/2;a+=energy/3;
                maxError=fmax(maxError,fmax(fabs(p-observed[2*window]),fabs(a-observed[2*window+1])));
                maxEnergy=fmax(maxEnergy,fabs(2*observed[2*window]+3*observed[2*window+1]-(reverse?2100:1900)));
            }
            std::cout<<"contact "<<scenario<<" valid="<<report.valid<<" converged="<<report.converged<<" iterations="<<report.iteration
                <<" maxTemperatureError="<<maxError<<" maxEnergyError="<<maxEnergy<<'\n';
            if(maxError>2e-8||maxEnergy>2e-8)return 2;
        }
        check(cudaFree(context));check(cudaFree(interval));check(cudaFree(temperatures));
    }
    std::cout<<"PASS: resident solid-contact loop, two region candidates, many-to-one contact, changing windows, Aitken, reverse heat and iteration limit\n";
}
