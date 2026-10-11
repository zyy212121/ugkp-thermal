#include "configuration/CouplingEntry.H"
#include "io/CoupledOutput.H"
#include <sstream>
#include <cassert>
#include <string>
int main() {
    chmt::HostState state;
    std::string error;
    assert(!chmt::validateCouplingEntry("Standalone",state,error));
    assert(!chmt::validateCouplingEntry("LegacyExplicit",state,error));
    assert(!chmt::validateCouplingEntry("Multirate",state,error));
    state.gas.resize(1); state.gasMesh.volumes={1};
    assert(!chmt::validateCouplingEntry("Multirate",state,error));
    state.solid.resize(1); state.solidMesh.volumes={1};
    assert(!chmt::validateCouplingEntry("Multirate",state,error));
    state.gasMesh.owner={0}; state.gasMesh.neighbour={-1};
    state.gasMesh.boundaryKind={chmt::BoundaryKind::Interface};
    state.solidMesh.owner={0}; state.solidMesh.neighbour={-1};
    state.surface.area={1}; state.surface.gasFace={0};
    state.surface.solidFace={0}; state.surface.solidCell={0};
    state.surface.persistentId={11};
    assert(chmt::validateCouplingEntry("Multirate",state,error));
    std::ostringstream output;chmt::ModelConfig model;
    assert(chmt::io::writeAcceptedCoupledSummary(output,state,model,error));
    state.commitSequence=1;std::ostringstream rejected;
    assert(!chmt::io::writeAcceptedCoupledSummary(rejected,state,model,error));
    assert(rejected.str().empty());state.commitSequence=0;
    state.surface.gasFace[0]=-1;
    assert(!chmt::validateCouplingEntry("Multirate",state,error));
    state.surface.gasFace[0]=0; state.surface.solidCell[0]=-1;
    assert(!chmt::validateCouplingEntry("Multirate",state,error));
}
