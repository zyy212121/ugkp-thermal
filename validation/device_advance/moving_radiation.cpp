#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/RadiationWarmStart.H"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
using namespace ugkp::gpu;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    RadiationGeometry g;
    g.cells=2;g.directions=2;g.owner={0};g.neighbour={1};g.cellOffsets={0,1,2};g.cellFaces={0,0};
    const double hemi=2*std::acos(-1.);
    g.volume={1,1};g.weights={hemi,hemi};g.directionCosines={1,-1,-1,1};
    g.boundaryOwner={0,1};g.boundaryPatch={0,1};g.boundaryFace={0,0};
    g.boundaryType={RadiationBoundary::Diffuse,RadiationBoundary::Black};
    g.boundaryArea={1,1};g.diffuseDenominator={hemi,hemi};g.fixedIntensity={0,0};
    g.internalFlux={1,-1};g.transportDiagonal={1,1,1,1};g.boundaryFlux={-1,1,1,-1};g.boundaryCosine=g.boundaryFlux;
    MieOpticalTable t;t.temperature={100,2000};t.diameter={1e-4,1e-2};t.mu={-1,1};t.band={1,1};
    t.absorption.assign(4,1);t.scattering.assign(4,.3);t.extinction.assign(4,1.3);t.phase.assign(8,1);
    RadiationControls controls;controls.parallelAngles=true;controls.sourceIterations=16;controls.linearIterations=20;
    controls.sourceTolerance=controls.linearTolerance=1e-5;
    RadiationSnapshot input;
    input.epsilon={1e-4,2e-4};input.temperature={600,700};input.diameter={1e-3,2e-3};input.wallTemperature={400,500};input.emissivity={.7,1};
    DiscreteOrdinateRadiation owner(g,t,controls);
    auto compare=[&]{
        DiscreteOrdinateRadiation fresh(g,t,controls);
        auto a=owner.solve(input,false),b=fresh.solve(input,false);
        require(a.incident==b.incident&&a.boundaryFlux==b.boundaryFlux&&a.power==b.power,"changed geometry differs from a fresh transport solve");
        for(auto n:a.incident)require(std::isfinite(n),"nonfinite radiation");
    };
    auto faceFixture=g;
    faceFixture.diffuseDenominator[0]=0;
    bool missingIncoming=false;
    try{owner.updateGeometry(faceFixture);}
    catch(const std::invalid_argument&){missingIncoming=true;}
    require(missingIncoming,"active diffuse inflow accepted a zero normalizer");
    for(int a=0;a<g.directions;++a){
        faceFixture.boundaryFlux[2*a]=0;
        faceFixture.boundaryCosine[2*a]=0;
    }
    owner.updateGeometry(faceFixture);
    owner.updateGeometry(g);
    auto rev=owner.executionRevision();g.volume[0]=1.25;g.internalFlux[0]=1.125;
    owner.updateGeometry(g);auto change=owner.lastGeometryUpdate();
    require(owner.executionRevision()==rev&&!change.rebuiltSourceGraph&&change.updatedDirectionValues==1,"metric change rebuilt the schedule");compare();
    std::vector<double> saved;owner.downloadIntensities(saved);
    RadiationWarmStart snapshot;snapshot.describe(2,2,7,true);owner.freezeIntensities(snapshot,nullptr);
    g.internalFlux[0]=-.5;owner.updateGeometry(g);change=owner.lastGeometryUpdate();
    require(change.rebuiltDirections==1&&!change.rebuiltStructure&&change.rebuiltSourceGraph,"upwind reversal did not rebuild exactly one direction");compare();
    g.boundaryType[0]=RadiationBoundary::Specular;g.reflectionOffsets={0,1,1,1,1};g.reflectionDonors={1};g.reflectionWeights={1};
    owner.updateGeometry(g);change=owner.lastGeometryUpdate();
    require(change.updatedReflection&&!change.rebuiltStructure&&change.rebuiltDirections==0&&change.rebuiltSourceGraph,"new reflection storage was not rebound");compare();
    rev=owner.executionRevision();g.reflectionWeights[0]=.875;owner.updateGeometry(g);
    require(owner.executionRevision()==rev&&owner.lastGeometryUpdate().updatedReflection,"reflection coefficient rebuilt execution");compare();
    g.owner={1};g.neighbour={0};owner.updateGeometry(g);
    require(owner.lastGeometryUpdate().rebuiltStructure,"topology did not invalidate execution");compare();
    std::string error;require(snapshot.write(argv[1],error),error.c_str());
    ReadWarmStart disk;require(readWarmStartFile(argv[1],2,2,7,disk,error),error.c_str());
    require(disk.initialized&&disk.values==saved,"old angular snapshot changed during later solves");
    std::cout<<"PASS metric, upwind, reflection address and coefficient, topology, and immutable warm start\n";
}
