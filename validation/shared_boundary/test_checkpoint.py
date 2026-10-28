from pathlib import Path
import subprocess
R=Path(__file__).resolve().parents[2]
def test_program_checkpoint_without_original_case(tmp_path):
    source=tmp_path/'checkpoint.cpp'
    source.write_text(r"""
#include "restart/Checkpoint.H"
#include "gasBoundary/Definition.H"
#include <cassert>
#include <fstream>
#include <iterator>
int main(int argc,char** argv){
 assert(argc==2);chmt::ModelConfig model;chmt::HostState state;state.gasMesh.owner={0};state.time=.75;
 auto definition=std::make_shared<ugkp::boundary::Definition>();definition->faces.resize(1);
 auto& face=definition->faces[0];face.pMode=face.tMode=2;face.uMode=4;
 face.pressureGamma=face.temperatureGamma=1.4;face.value[ugkp::boundary::Pressure]=1e5;face.value[ugkp::boundary::Temperature]=300;
 face.function[ugkp::boundary::Temperature]=0;
 definition->functions={{ugkp::boundary::FunctionKind::Table,ugkp::boundary::Bounds::Clamp,0,2,0}};
 definition->terms={{0,300},{1,400}};state.gasMesh.gasBoundary=definition;
 std::string error;assert(chmt::writeCheckpoint(argv[1],model,state,error));
 chmt::HostState restored;assert(chmt::readCheckpoint(argv[1],model,restored,error));
 assert(restored.time==.75&&restored.gasMesh.gasBoundary);
 const auto& d=*restored.gasMesh.gasBoundary;
 ugkp::boundary::View v{d.faces.data(),d.functions.data(),d.terms.data(),1};
 assert(ugkp::boundary::evaluate(v,0,restored.time,0)==375);
 auto bytes=[&](){std::ifstream f(std::string(argv[1])+"/state.chmt",std::ios::binary);return std::string(std::istreambuf_iterator<char>(f),{});};
 const auto saved=bytes();assert(chmt::writeCheckpoint(argv[1],model,restored,error));assert(saved==bytes());
}
""")
    exe=tmp_path/'checkpoint'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(R/'applications/CHMT'),'-I'+str(R/'common'),str(source),str(R/'applications/CHMT/restart/Checkpoint.C'),'-o',str(exe)],check=True)
    subprocess.run([exe,tmp_path/'saved'],check=True)
