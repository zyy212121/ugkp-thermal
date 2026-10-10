from pathlib import Path
import subprocess
APP=Path(__file__).resolve().parents[1]
def test_accepted_wall_diagnostics_mark_unavailable_without_nan(tmp_path):
    source=r'''
#include "io/CoupledOutput.H"
#include <cassert>
#include <sstream>
int main(){using namespace chmt;
 ModelConfig model;HostState h;h.time=1;h.acceptedSteps=h.commitSequence=1;
 h.gas.resize(1);h.solid.resize(1);h.gasMesh.volumes={1};h.solidMesh.volumes={1};
 h.gasMesh.owner={0};h.gasMesh.neighbour={-1};h.gasMesh.boundaryKind={BoundaryKind::Interface};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.area={1};h.surface.gasFace={0};h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={11};
 std::string error;std::ostringstream unavailable;
 assert(io::writeAcceptedWallLayerFields(unavailable,h,model,error));assert(unavailable.str().find("NOT_AVAILABLE")!=std::string::npos);
 h.gasWallDiagnosticTime=.5;h.gasWallDiagnostics.resize(1);auto& d=h.gasWallDiagnostics[0];
 d.traceTemperature=600;d.traceDensity=1;d.traceMassFraction[0]=1;d.conductiveHeatFlux=-200;
 d.wallSpeciesFlux[0]=.01;d.matchingSpeciesFlux[0]=.005;d.reactionIntegral[0]=-.005;
 d.chemistryMismatchAvailable=false;for(int s=0;s<Ns;++s)d.chemistryMismatch[s]=std::numeric_limits<double>::quiet_NaN();
 std::ostringstream available;assert(io::writeAcceptedWallLayerFields(available,h,model,error));
 assert(available.str().find("nan")==std::string::npos);assert(available.str().find("reaction_integral")!=std::string::npos);
 assert(available.str().find("AVAILABLE")!=std::string::npos);
 h.gasWallDiagnosticTime=2;std::ostringstream invalid;assert(!io::writeAcceptedWallLayerFields(invalid,h,model,error));assert(invalid.str().empty());
}
'''
    src=tmp_path/'output.cpp';src.write_text(source);exe=tmp_path/'output'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
