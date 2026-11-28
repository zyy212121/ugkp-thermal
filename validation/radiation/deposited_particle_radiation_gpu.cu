#include "common/thermal/GpuResidentStrict.cu"
#include "common/gpu/DiscreteOrdinateRadiation.H"
#include "common/gpu/ParticleRadiationMaterial.H"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <initializer_list>
#include <utility>

namespace
{
using namespace Foam::gpuThermal;
using namespace ugkp::gpu;

void require(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceBuffers
{
    std::vector<void*> allocations;

    template<class T> T* make(std::initializer_list<T> values)
    {
        T* buffer=nullptr;
        require(cudaMallocManaged(&buffer,values.size()*sizeof(T)));
        allocations.push_back(buffer);
        std::copy(values.begin(),values.end(),buffer);
        return buffer;
    }

    template<class T> T* make(std::size_t size)
    {
        T* buffer=nullptr;
        require(cudaMallocManaged(&buffer,size*sizeof(T)));
        allocations.push_back(buffer);
        std::fill_n(buffer,size,T{});
        return buffer;
    }

    ~DeviceBuffers()
    {
        for (void* allocation : allocations) cudaFree(allocation);
    }
};

template<class State>
auto initialiseCarry(State& state,DeviceBuffers& memory,int)
    -> decltype(state.pThermalEnthalpyResidual=memory.make<GpuReal>(3))
{
    return state.pThermalEnthalpyResidual=memory.make<GpuReal>(3);
}

template<class State>
GpuReal* initialiseCarry(State&,DeviceBuffers&,long)
{
    return nullptr;
}

RadiationGeometry geometry()
{
    RadiationGeometry value;
    const double hemisphere=2*std::acos(-1.);
    value.cells=1;value.directions=2;value.cellOffsets={0,0};value.volume={1};
    value.weights={hemisphere,hemisphere};value.directionCosines={1,-1,-1,1};
    value.boundaryOwner={0,0};value.boundaryPatch={0,1};value.boundaryFace={0,0};
    value.boundaryType={RadiationBoundary::Diffuse,RadiationBoundary::Diffuse};
    value.boundaryArea={1,1};value.diffuseDenominator={hemisphere,hemisphere};value.fixedIntensity={0,0};
    value.transportDiagonal={1,1};value.boundaryFlux={-1,1,1,-1};value.boundaryCosine=value.boundaryFlux;
    return value;
}

MieOpticalTable table()
{
    MieOpticalTable result;
    result.temperature={100,5000};result.diameter={.001,.002};result.mu={-1,1};
    result.absorption.assign(4,1);result.scattering.assign(4,.3);
    result.extinction.assign(4,1.3);result.band.assign(2,1);result.phase.assign(8,1);
    return result;
}

void checkNear(GpuReal value,GpuReal expected,GpuReal tolerance)
{
    if (!std::isfinite(value)||std::abs(value-expected)>tolerance)
        throw std::runtime_error("radiation result "+std::to_string(value)
            +" differs from "+std::to_string(expected)+" by more than "+std::to_string(tolerance));
}

template<class Storage>
GpuReal meanEnthalpy(const Storage* nodes,int count)
{
    GpuReal mean=0;
    for (int node=0;node<count;++node) mean+=GpuReal(nodes[node])/GpuReal(count);
    return mean;
}

std::pair<RadiationResult,ParticleRadiationMaterialResult> solve
(
    DiscreteOrdinateRadiation& dom,ParticleRadiationMaterial& response,
    const RadiationSnapshot& transport,bool parent
)
{
    if (!parent)
    {
        auto radiation=dom.solve(transport,false);
        auto material=response.finishAnchoredWindow(dom.deviceView(),true);
        return {std::move(radiation),std::move(material)};
    }

    dom.prepareParentInputs(transport,false);
    cudaGraph_t graph=nullptr;
    require(cudaGraphCreate(&graph,0));
    auto node=response.appendWindowStartTo(graph);
    node=dom.appendInputsTo(graph,node);
    node=dom.appendSourceTo(graph,node);
    node=dom.appendDiagnosticsTo(graph,node);
    response.appendWindowFinishTo(graph,node,dom.deviceView());
    cudaGraphExec_t executable=nullptr;
    require(cudaGraphInstantiate(&executable,graph,0));
    auto stream=static_cast<cudaStream_t>(dom.producerStream());
    require(cudaGraphLaunch(executable,stream));
    require(cudaStreamSynchronize(stream));
    require(cudaGraphExecDestroy(executable));
    require(cudaGraphDestroy(graph));
    response.acceptExternalCompletion({});
    return {dom.collect(),response.collect()};
}

void run(bool packed,bool heating,bool parent)
{
    DeviceBuffers memory;
    auto& state=*memory.make<DeviceState>(1);
    const auto* carry=initialiseCarry(state,memory,0);
    state.nCells=1;state.nFaces=2;state.particleCapacity=3;
    state.particleCountDevice=memory.make<int>({3});
    state.pStatus=memory.make<int>({1,1,1});
    state.pCellId=memory.make<int>({0,0,0});
    state.pStuck=memory.make<unsigned char>({particleWallDeposited,particleWallDeposited,particleWallTransientDeposit});
    state.pStuckFaceId=memory.make<int>({0,1,0});
    state.particleStuckCandidateMask=memory.make<unsigned char>({particleWallSolidifyingDeposition,particleWallColdWall2D});
    state.pColdHandle=memory.make<int>({1,-1,0});
    state.pColdNodeSpecificEnthalpy=memory.make<GpuReal>(16);
    state.pCold2DNodeSpecificEnthalpy=memory.make<float>(3*64);
    state.pm=memory.make<GpuReal>({GpuReal(.001),GpuReal(.002),GpuReal(.004)});
    state.pd=memory.make<GpuReal>({GpuReal(.001),GpuReal(.001),GpuReal(.001)});
    state.pT=memory.make<GpuReal>({1000,1000,2500});
    state.TpMin=100;state.TpMax=4500;state.particleDiameterFallback=GpuReal(.001);state.rhoSolid=1;
    state.coldWallSolidificationParameters.meltingTemperatureK=2327;
    state.coldWallSolidificationParameters.mushyRangeK=20;
    state.coldWallSolidificationParameters.latentHeatJkg=1000000;
    state.coldWallSolidificationParameters.solidSpecificHeatJkgK=900;
    const auto law=ugkwp::coldWallCarryCaloric(state.coldWallSolidificationParameters);
    auto* axial=state.pColdNodeSpecificEnthalpy+8;
    auto* axisymmetric=state.pCold2DNodeSpecificEnthalpy+64;
    for (int n=0;n<8;++n) axial[n]=law.enthalpy(GpuReal(920+20*n));
    for (int n=0;n<64;++n) axisymmetric[n]=float(law.enthalpy(GpuReal(980+n)));
    const GpuReal initialAxial=meanEnthalpy(axial,8),initialAxisymmetric=meanEnthalpy(axisymmetric,64);
    state.pT[0]=law.temperature(initialAxial);
    state.pT[1]=law.temperature(initialAxisymmetric);
    const GpuReal initialT0=state.pT[0],initialT1=state.pT[1];
    std::vector<GpuReal> transientNodes(state.pColdNodeSpecificEnthalpy,state.pColdNodeSpecificEnthalpy+8);
    state.preBaseCellOffset=memory.make<int>({0,3});
    state.csrCellLocalPathEnabled=packed;
    state.preBaseDirectoryReady=packed;
    state.radiationMobileMass=memory.make<GpuReal>(1);
    state.radiationMobileTemperatureMass=memory.make<GpuReal>(1);
    state.radiationMobileDiameterMass=memory.make<GpuReal>(1);
    state.radiationResponseDevice=memory.make<ParticleRadiationResponseView>(1);
    state.diagnostics=memory.make<ugkp::diagnostics::DeviceContext>(1);

    clearMobileParticleRadiationSumsKernel<<<1,32>>>(&state);
    if (packed) accumulatePackedMobileParticleRadiationSumsKernel<<<1,32,3*sizeof(GpuReal)>>>(&state);
    else accumulateMobileParticleRadiationSumsAtomicKernel<<<1,32>>>(&state);
    publishParticleRadiationResponseKernel<<<1,1>>>(&state);
    require(cudaDeviceSynchronize());
    const GpuReal mass=state.pm[0]+state.pm[1];
    checkNear(state.radiationMobileMass[0],mass,mass*GpuReal(1e-6));
    checkNear(state.radiationMobileTemperatureMass[0],state.pm[0]*initialT0+state.pm[1]*initialT1,GpuReal(1e-5));
    auto* occupied=memory.make<GpuReal>({GpuReal(.25),GpuReal(.5)});
    auto* scale=memory.make<GpuReal>({GpuReal(1),GpuReal(.5)});
    auto* wallTemperature=memory.make<GpuReal>({heating?GpuReal(1800):GpuReal(350),
        heating?GpuReal(1800):GpuReal(350)});
    RadiationSnapshot transport;
    transport.wallTemperature={0,0};transport.emissivity={1,1};
    transport.residentWallTemperature={1,1};transport.contactEmissivity={1,1};
    transport.resident={true,1,0,2,state.radiationMobileMass,state.radiationMobileTemperatureMass,
        state.radiationMobileDiameterMass,wallTemperature,occupied,scale,GpuReal(1),GpuReal(100),
        nullptr,nullptr,state.diagnostics};
    ParticleRadiationMaterialInput material;
    material.particleResident=transport.resident;
    material.volume={1};material.boundaryArea={1,1};material.coupledBoundary={1,1};
    material.elapsed=GpuReal(.001);material.minimumTemperature=state.TpMin;material.maximumTemperature=state.TpMax;
    material.response.enabled=true;material.response.capacity=3;material.response.deviceView=state.radiationResponseDevice;
    const auto mie=table();
    RadiationControls controls;
    controls.sourceIterations=512;
    DiscreteOrdinateRadiation dom(geometry(),mie,controls);
    ParticleRadiationMaterial response(mie);
    transport.thermalCoupling=parent
        ? response.prepareParentWindow(material,memory.make<double>({material.elapsed}))
        : response.beginAnchoredWindow(material);
    dom.setThermalIterationHook([&](ExecutionGraph graph,ExecutionGraphNode dependency,void* stream,
        const RadiationThermalIterationDeviceView& view)
    {
        return response.appendAnchoredIteration(graph,dependency,stream,view);
    },transport.thermalCoupling.graphOwner,transport.thermalCoupling.graphRevision);
    const auto [radiation,result]=solve(dom,response,transport,parent);
    checkNear(radiation.absorption[0],GpuReal(1.5)*mass/state.pd[0],GpuReal(2e-5));
    checkNear(radiation.scattering[0],GpuReal(.3)*radiation.absorption[0],GpuReal(2e-5));
    const GpuReal tw=wallTemperature[0];
    const GpuReal wallEmission=GpuReal(2*5.670374419e-8)*tw*tw*tw*tw;
    const GpuReal exposedWallFlux=GpuReal(.75)*(GpuReal(.5)*radiation.incident[0]-wallEmission);
    for (GpuReal flux : radiation.boundaryFlux)
        checkNear(flux,exposedWallFlux,GpuReal(.0001)*std::abs(exposedWallFlux));
    auto* requests=memory.make<RadiationAffineTemperatureUpdate>({{result.energy[0],0}});
    state.radiationMobileTemperatureMass[0]=0;
    radiationEnthalpyApplyKernel<<<1,32>>>(&state,requests,result.resident.publicationFraction);
    require(cudaDeviceSynchronize());
    assert(!state.diagnostics->fatalClaim);
    assert(state.pStuck[0]==particleWallDeposited && state.pStuck[1]==particleWallDeposited);
    assert(state.pStuckFaceId[0]==0 && state.pStuckFaceId[1]==1);
    assert(state.pStuck[2]==particleWallTransientDeposit && state.pT[2]==2500);
    assert(std::equal(transientNodes.begin(),transientNodes.end(),state.pColdNodeSpecificEnthalpy));
    assert(heating ? state.pT[0]>initialT0 && state.pT[1]>initialT1
                   : state.pT[0]<initialT0 && state.pT[1]<initialT1);
    const GpuReal published=state.radiationMobileTemperatureMass[0];
    checkNear(published,state.pm[0]*(meanEnthalpy(axial,8)-initialAxial)
        +state.pm[1]*(meanEnthalpy(axisymmetric,64)+(carry?carry[1]:GpuReal(0))-initialAxisymmetric),
        GpuReal(2e-3));
    checkNear((state.pm[0]*state.pT[0]+state.pm[1]*state.pT[1])/mass,result.temperature[0],GpuReal(.002));
    checkNear(published+result.boundaryEnergy[0]+result.boundaryEnergy[1],GpuReal(0),
        GpuReal(.003)*std::max(std::abs(published),GpuReal(1)));
    std::cout<<"FP"<<sizeof(GpuReal)*8<<" parent="<<parent<<" packed="<<packed<<" heating="<<heating
        <<" mass="<<mass<<" absorption="<<radiation.absorption[0]<<" heat="<<published
        <<" wall="<<result.boundaryEnergy[0]+result.boundaryEnergy[1]
        <<" deposit T="<<state.pT[0]<<','<<state.pT[1]<<'\n';
}
}

template<class Storage>
void checkProfileShift()
{
    ugkwp::ParticleCaloric law;
    law.phaseChange=true;
    law.solidCp=900;
    for (const GpuReal limit : {GpuReal(1500),GpuReal(2100),GpuReal(300),GpuReal(900)})
    {
        const bool heating=limit>=900;
        Storage nodes[]={Storage(270000),Storage(810000)};
        const GpuReal bound=law.enthalpy(limit);
        const GpuReal capacity=coldNodeSpecificCapacity(nodes,2,GpuReal(0),bound,heating);
        const GpuReal heat=heating?GpuReal(90000):GpuReal(-90000);
        const auto sample=sampleColdRadiationNodes(nodes,2,GpuReal(0),bound,heating,heat/capacity,law,nodes);
        assert(sample.valid);
        checkNear(sample.heat,heat,GpuReal(.125));
        checkNear(nodes[1]-nodes[0],GpuReal(540000),GpuReal(.125));
        checkNear(nodes[0],GpuReal(270000)+heat,GpuReal(.125));
    }
    Storage nodes[]={Storage(90000),Storage(810000)};
    const auto capacity=coldNodeSpecificCapacity(nodes,2,GpuReal(0),GpuReal(0),false);
    checkNear(capacity,GpuReal(-90000),GpuReal(.125));
    const auto sample=sampleColdRadiationNodes(nodes,2,GpuReal(0),GpuReal(0),false,GpuReal(1),law,nodes);
    assert(sample.valid);
    checkNear(nodes[0],GpuReal(0),GpuReal(.125));
    checkNear(nodes[1],GpuReal(720000),GpuReal(.125));
}

int main()
{
    checkProfileShift<GpuReal>();
    checkProfileShift<float>();
    if constexpr (sizeof(GpuReal)>sizeof(float))
    {
        const float enthalpy[]={900000.f,1000000.f};
        const auto sample=sampleColdRadiationNodes(enthalpy,2,GpuReal(0),
            GpuReal(2000000),true,GpuReal(1e-10),ugkwp::ParticleCaloric{});
        assert(sample.valid);
        checkNear(sample.heat,GpuReal(.000105),GpuReal(1e-12));
        checkNear(sample.carry,sample.heat,GpuReal(1e-12));
        assert(sample.grid<GpuReal(1e-8));
    }
    for (bool parent : {false,true})
        for (bool packed : {false,true})
            for (bool heating : {false,true})
                run(packed,heating,parent);
}
