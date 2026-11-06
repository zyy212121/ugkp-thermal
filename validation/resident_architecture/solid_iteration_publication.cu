#include "gpu/SolidThermalProperties.H"
#include "Diagnostics.H"
#include "gpu/ImplicitDiffusion.H"
#include "gpu/ExecutionGraph.cuh"
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
__global__ void updateInterval(double* interval,int window){if(!threadIdx.x)*interval=.5*(window+1);}
__device__ void verifyWindow(const double* temperature,const SolidThermalIteration* report,int window,bool aitken,bool limited,int* failure){
    // Two insulated unit-volume cells, rho*Cp=2 and k=1.
    // The mean stays 350 K and the half difference shrinks by
    // 1/(1+dt) at every implicit interval, including changing dt.
    double difference=50;for(int i=0;i<=window;++i)difference/=1+.5*(i+1);
    if(!report->valid||report->converged==limited||fabs(temperature[0]-(350-difference))>2e-6
       ||fabs(temperature[1]-(350+difference))>2e-6||(!aitken&&report->iteration!=(limited?1:2))) *failure=window+1;
}
__global__ void verify(const double* temperature,const SolidThermalIteration* report,int window,bool aitken,bool limited,int* failure){
    if(!threadIdx.x)verifyWindow(temperature,report,window,aitken,limited,failure);
}
__global__ void clockInterval(double* interval,const int* clock){if(!threadIdx.x)*interval=.5*(*clock+1);}
__global__ void finishWindow(const double* temperature,const SolidThermalIteration* report,int* clock,int* failure,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;verifyWindow(temperature,report,*clock,false,false,failure);
    ++*clock;cudaGraphSetConditional(handle,*clock<3);
}
int main(){
    for(int scenario=0;scenario<4;++scenario){const bool aitken=scenario==1,limited=scenario==2;
        ThermalPropertyTable tables[4];for(auto& table:tables){table.constant=true;table.constantValue=1;}tables[1].constantValue=2;
        SolidThermalProperties properties(tables,nullptr,0);ImplicitDiffusion diffusion;SparseConjugateGradient sparse;
        const double initial[]={300,400};diffusion.uploadSolution(initial,2);
        const auto stream=diffusion.deviceValues().readyStream;
        properties.setResidentCellAnchors(diffusion.deviceTemperature(),2,stream);
        properties.setConvergenceBoundary({-2,-2},{300,400});
        const int coefficients=properties.prepareResidentEvaluation({{300,300},{400,400}},{-1,-2},2,ThermalCapacity|ThermalConductivity,true);
        const int convergence=properties.prepareResidentEvaluation({}, {},2,ThermalEnergy|ThermalTemperatureChange,false);
        properties.prepareResidentIteration(2);
        // Initialization only: produce valid coefficients before preparing the
        // discretization and all graph-owned buffer addresses.
        std::vector<ThermalPropertyResult> ignored;
        properties.evaluateResidentCells({{300,300},{400,400}},diffusion.deviceTemperature(),nullptr,2,
            ThermalCapacity|ThermalConductivity,true,stream,ignored);
        const auto fields=properties.deviceFields();
        DiffusionInput input;input.cells=2;input.internalFaces=1;input.owner={0,0,1};input.neighbour={1};
        input.volume={1,1};input.area={1,1,1};input.delta={1,1,1};input.corrected=false;
        input.deviceTemperature=diffusion.deviceTemperature();input.deviceConductivity=fields.conductivity;input.deviceBoundaryConductivity=fields.conductivity+2;input.conductivityStream=fields.readyStream;
        input.conductivityWeights={.5,1,1};input.conductivityBoundary={1,1};input.conductivityNeighbour={0,0};input.conductivityCoupled={0,0};input.conductivityAdjacentCells={-1,-1};input.conductivityJump={0,0};
        input.gradientInternal={0,0};input.gradientBoundary={0,0};input.gradientWeights={.5,1,1};input.gradientNeighbour={-1,-1};
        input.valueInternal={1,1};input.valueBoundary={0,0};input.normalInternal={0,0};input.normalBoundary={0,0};input.jump={0,0};input.boundaryValueDelta={1,1};
        diffusion.prescribeBoundaryEnergy({0,0},{0,0},.5,fields.conductivity+2,fields.readyStream);
        DiffusionTransientInput transient;transient.dt=.5;transient.sourceFromCapacity=true;transient.deviceCapacity=fields.capacity;transient.deviceRhs=properties.deviceCellAnchors();transient.readyStream=fields.readyStream;
        diffusion.assembleResident(input,&transient);
        std::vector<double> unusedFlux;diffusion.flux({0,0},{0,0},{-1,-1},{0,0},unusedFlux,true,false);
        const int owner[]={0},neighbour[]={1},boundaryOwner[]={0,1},boundaryNeighbour[]={-1,-1};
        SparseLduView topology;topology.rows=2;topology.faces=1;topology.boundaryFaces=2;
        topology.owner=owner;topology.neighbour=neighbour;topology.boundaryOwner=boundaryOwner;topology.boundaryNeighbour=boundaryNeighbour;
        ConjugateGradientControls linear;linear.tolerance=1e-13;linear.maximumIterations=100;
        sparse.configureResidentLocalLdu(topology,diffusion.deviceValues(),linear);
        SolidThermalResidentProblem problem;problem.cells=2;problem.coefficientPlan=coefficients;problem.convergencePlan=convergence;problem.interval=.5;
        double* interval;check(cudaMalloc(&interval,sizeof(double)));problem.deviceInterval=interval;
        problem.controls.maximumIterations=limited?1:100;problem.controls.relativeTolerance=1e-9;problem.controls.energyAbsoluteTolerance=1e-9;problem.controls.energyRelativeTolerance=1e-12;
        problem.controls.aitken=aitken;problem.controls.initialRelaxation=.5;problem.controls.minimumRelaxation=.1;problem.controls.maximumRelaxation=1;
        SolidThermalIterationGraph graph(properties,diffusion,sparse,problem);
        int* failure;check(cudaMalloc(&failure,sizeof(int)));check(cudaMemset(failure,0,sizeof(int)));
        cudaGraph_t schedule=nullptr;cudaGraphExec_t executable=nullptr;int* clock=nullptr;
        if(scenario==3){
            check(cudaMalloc(&clock,sizeof(int)));check(cudaMemset(clock,0,sizeof(int)));
            check(cudaGraphCreate(&schedule,0));cudaGraphConditionalHandle handle;
            check(cudaGraphConditionalHandleCreate(&handle,schedule,1,cudaGraphCondAssignDefault));
            cudaGraph_t body=nullptr;executionGraphDetail::addWhile(schedule,handle,{},body);
            const auto begin=executionGraphDetail::appendCaptured(body,{},static_cast<cudaStream_t>(stream),[&]{clockInterval<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(interval,clock);});
            const auto solve=graph.appendTo(body,begin);
            executionGraphDetail::appendCaptured(body,{solve},static_cast<cudaStream_t>(stream),[&]{
                finishWindow<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(diffusion.deviceTemperature(),graph.deviceReport(),clock,failure,handle);});
            check(cudaGraphInstantiateWithFlags(&executable,schedule,0));
        }
        check(cudaDeviceSynchronize());if(scenario==0)check(cudaProfilerStart());
        if(scenario==3)check(cudaGraphLaunch(executable,static_cast<cudaStream_t>(stream)));
        else for(int window=0;window<3;++window){updateInterval<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(interval,window);graph.enqueue(stream);
            verify<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(diffusion.deviceTemperature(),graph.deviceReport(),window,aitken,limited,failure);}
        if(scenario==0)check(cudaProfilerStop());
        check(cudaDeviceSynchronize());int failed;check(cudaMemcpy(&failed,failure,sizeof(int),cudaMemcpyDeviceToHost));
        ugkp::diagnostics::DeviceContext diagnostics{};SolidThermalIteration result;
        if(scenario==3)check(cudaMemcpy(&result,graph.deviceReport(),sizeof(result),cudaMemcpyDeviceToHost));
        else result=graph.collect(&diagnostics);
        if(limited&&(diagnostics.warnings[0].count||diagnostics.warnings[1].count!=3))return 3;double temperature[2];diffusion.downloadSolution(temperature,2);
        std::cout<<"solid "<<scenario<<" failure="<<failed<<" iterations="<<result.iteration<<" valid="<<result.valid<<" converged="<<result.converged
            <<" T="<<temperature[0]<<','<<temperature[1]<<" residual="<<result.energyResidual.binary64()<<'\n';
        if(failed)return 1;
        if(executable)check(cudaGraphExecDestroy(executable));if(schedule)check(cudaGraphDestroy(schedule));if(clock)check(cudaFree(clock));
        check(cudaFree(failure));check(cudaFree(interval));
    }
    ScaledThermalSum huge;huge.product(1e300,1e300);auto limit=huge.multiplied(1e-12);auto residual=huge;residual.merge(huge.negated());residual.add(1);
    if(!residual.magnitude().lessEqual(limit)||huge.lessEqual(limit))return 2;
    if(std::abs(limit.ratioTo(huge)-1e-12)>1e-27||std::abs(huge.ratioTo(limit)-1e12)>1e-3)return 4;
    std::cout<<"PASS: nonlinear implicit solid graph, energy closure, Aitken, repeated windows, extended-range device comparison\n";
}
