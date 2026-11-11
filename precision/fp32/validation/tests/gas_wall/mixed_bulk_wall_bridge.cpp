// This fixture is executed only at the scheduled numerical checkpoint.
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

using namespace ugkwp;
using namespace ugkwp::gaswall;
template<class R> struct BulkState
    : GasStateView<R,double,SstCoefficientsT<R>,double,double,double> {
    GasSpeciesState<R,2,double> gasSpecies;
    GasBoundaryLayerModelState<double,2> gasBoundaryLayerModel;
};
template<class R> struct Fixture {
    CapturedSstWall captured;
    BulkState<R> state;
    R rho[1],momentum[3],rhoK[1],rhoOmega[1],temperature[1],boundaryTemperature[1];
    R speciesDensity[2],diffusivity[2],boundaryVelocity[3]{},gradients[9]{};
    double energy[1],area[1]{1},weight[1]{1};
    double quadratureDistance[2]{},quadratureWeight[2]{},fluxProfile[3*25]{};
    int face[1]{0},cell[1]{0},offset[2]{0,1},fixed[1]{1},status[1]{};
    WallInput<double,2> input;
    WallOutput<double,2> evaluation;
    WallResidentOutput<double,2> output;
    WallStatus wallStatus;
    GasBoundaryLayerExchange<double> exchange;
    GasBoundaryLayerSstClosure<double> closure;
    double speciesFlux[2]{},history[24*6]{},candidate[24*6]{};
    WallHistoryFrame<2> frame,candidateFrame;
    WallReferenceStats stats;
    WallWorkspace<double,2,24> workspace;
    WallReferenceScratch<2,24> reference;
    coherent::Scratch<24> coherentScratch;
    coherent::PersistentWallState<24> coherentState;
    ConservativeWallColumn column;
    WallMatchingState<double,2> matching;
    double pressure=0;
    Fixture(bool volumeClosure=false) {
        input=captured.input;
        auto& s=state;auto& m=s.gasBoundaryLayerModel;
        s.nCells=s.nFaces=1;s.rho=rho;s.rhoE=energy;s.Tgas=temperature;s.magSf=area;
        s.rhoUx=momentum;s.rhoUy=momentum+1;s.rhoUz=momentum+2;
        s.rhoK=rhoK;s.rhoOmega=rhoOmega;s.faceOwner=cell;
        s.gradUxX=gradients;s.gradUxY=gradients+1;s.gradUxZ=gradients+2;
        s.gradUyX=gradients+3;s.gradUyY=gradients+4;s.gradUyZ=gradients+5;
        s.gradUzX=gradients+6;s.gradUzY=gradients+7;s.gradUzZ=gradients+8;
        s.riemannBoundaryT=boundaryTemperature;s.riemannBoundaryTFix=fixed;s.riemannBoundaryUFix=fixed;
        s.riemannBoundaryUx=boundaryVelocity;s.riemannBoundaryUy=boundaryVelocity+1;s.riemannBoundaryUz=boundaryVelocity+2;
        s.gasSpecies.thermo=input.model.thermo;s.gasSpecies.mode=input.model.mode;
        s.gasSpecies.rho=speciesDensity;s.gasSpecies.diffusivity=diffusivity;
        s.gasViscosity=input.model.molecularViscosity;s.gasMu=R(input.model.viscosity);
        s.gasPrClamped=R(input.model.molecularPrandtl);s.sstCoefficients=input.model.sstCoefficients;
        double gasR=0,internal=0,speed2=0;
        for(int j=0;j<2;++j){
            gasR+=input.matching.massFraction[j]*universalGasConstant<double>()/captured.species[j].molarMass;
            internal+=input.matching.massFraction[j]*speciesE(j,input.matching.temperature,input.model.thermo);
            diffusivity[j]=R(input.model.diffusivity[j]);
        }
        rho[0]=R(input.pressure/(gasR*input.matching.temperature));
        temperature[0]=R(input.matching.temperature);boundaryTemperature[0]=R(input.temperature);
        for(int d=0;d<3;++d){momentum[d]=R(double(rho[0])*input.matching.velocity[d]);speed2+=double(momentum[d])*momentum[d]/(double(rho[0])*rho[0]);}
        energy[0]=double(rho[0])*(internal+.5*speed2);
        for(int j=0;j<2;++j)speciesDensity[j]=R(double(rho[0])*input.matching.massFraction[j]);
        rhoK[0]=R(double(rho[0])*input.matching.k);rhoOmega[0]=R(double(rho[0])*input.matching.omega);
        s.gasBoundaryLayer.enabled=true;s.gasBoundaryLayer.count=1;
        s.gasBoundaryLayer.exchange=&exchange;s.gasBoundaryLayer.sst=&closure;
        s.gasBoundaryLayer.speciesFlux=speciesFlux;s.gasBoundaryLayer.status=status;
        m.input=&input;m.output=&output;m.evaluationOutput=&evaluation;m.status=&wallStatus;
        m.config=captured.config;m.faces=face;m.matchingOffsets=offset;m.matchingCells=cell;m.matchingWeights=weight;
        m.workspace=&workspace;m.workspaceCapacity=24;m.workspaceCount=1;
        m.volumeSstClosure=volumeClosure;m.fluxProfile=fluxProfile;
        quadratureDistance[0]=input.ownerDistance*.5;quadratureDistance[1]=input.ownerDistance*1.5;
        quadratureWeight[0]=quadratureWeight[1]=input.ownerDistance;
        input.quadrature={quadratureDistance,quadratureWeight,2,2*input.ownerDistance,2*input.ownerDistance*input.ownerDistance};
        m.acceptedHistory=history;m.candidateHistory=candidate;m.acceptedFrame=&frame;m.candidateFrame=&candidateFrame;
        m.referenceScratch=&reference;m.referenceStats=&stats;
        m.coherentScratch=&coherentScratch;m.coherentState=&coherentState;m.coherentEnabled=true;
    }
    void prepare() {
        double sweep=0;
        assert(prepareGasBoundaryLayerInput(state,0,0,sweep));
        assert(std::isfinite(input.pressure)&&std::isfinite(input.matching.temperature));
        // Exercise actual float bulk matching before enabling the conservative
        // column, whose already prepared matching data are always double.
        matching=input.matching;pressure=input.pressure;
        column.cells[0]=0;column.edge[2]=input.matchingDistance;
        auto& d=state.gasBoundaryLayerModel.conservative;
        d.enabled=true;d.columns=&column;d.matching=&matching;d.pressure=&pressure;
        assert(gasBoundaryLayerUsesCoherent(state.gasBoundaryLayerModel,0));
    }
    void solve() {
        assert((evaluateGasBoundaryLayerSlot<false,24>(state,0,0,detail::SerialWallTeam{})));
        assert(exchange.ready&&candidateFrame.valid&&coherentState.cache.haveState);
        assert(std::isfinite(output.residual)&&std::isfinite(output.conductiveHeatFlux));
        assert(std::isfinite(output.traction[0])&&std::isfinite(exchange.energy));
        assert(stats.residualEvaluations>0);
    }
};
int main() {
    static_assert(std::is_same<decltype(BulkState<float>{}.rho),float*>::value,"bulk density remains float");
    static_assert(std::is_same<decltype(BulkState<float>{}.gasBoundaryLayerModel.input),WallInput<double,2>*>::value,"wall model remains double");
    for(bool volumeClosure:{false,true}) {
    auto mixed=std::make_unique<Fixture<float>>(volumeClosure);
    auto reference=std::make_unique<Fixture<double>>(volumeClosure);
    mixed->prepare();reference->prepare();mixed->solve();reference->solve();
    auto relative=[](double a,double b){return std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)});};
    const double heatError=relative(mixed->output.conductiveHeatFlux,reference->output.conductiveHeatFlux);
    const double shearError=relative(mixed->output.traction[0],reference->output.traction[0]);
    assert(heatError<.01&&shearError<.01);
    assert(mixed->output.conductiveHeatFlux*reference->output.conductiveHeatFlux>=0);
    assert(mixed->stats.converged&&reference->stats.converged);
    std::cout<<"volume closure="<<volumeClosure<<" mixed wall heat relative error="<<heatError<<" shear relative error="<<shearError
             <<" residual="<<mixed->stats.residual<<" coherent residual calls="<<mixed->stats.residualEvaluations<<'\n';
    }
}
