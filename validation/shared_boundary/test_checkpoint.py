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
 face.solidMode=0;face.value[ugkp::boundary::SolidFraction]=.125;
 definition->functions={{ugkp::boundary::FunctionKind::Table,ugkp::boundary::Bounds::Clamp,0,2,0}};
 definition->terms={{0,300},{1,400}};state.gasMesh.gasBoundary=definition;
 std::string error;assert(chmt::writeCheckpoint(argv[1],model,state,error));
 chmt::HostState restored;assert(chmt::readCheckpoint(argv[1],model,restored,error));
 assert(restored.time==.75&&restored.gasMesh.gasBoundary);
 const auto& d=*restored.gasMesh.gasBoundary;
 assert(d.faces[0].solidMode==0&&d.faces[0].value[ugkp::boundary::SolidFraction]==.125);
 ugkp::boundary::View v{d.faces.data(),d.functions.data(),d.terms.data(),1};
 assert(ugkp::boundary::evaluate(v,0,restored.time,0)==375);
 auto bytes=[&](){std::ifstream f(std::string(argv[1])+"/state.chmt",std::ios::binary);return std::string(std::istreambuf_iterator<char>(f),{});};
 const auto saved=bytes();assert(chmt::writeCheckpoint(argv[1],model,restored,error));assert(saved==bytes());

 auto startup=std::make_shared<ugkp::boundary::Definition>(*definition);
 startup->faces[0].solidMode=1;startup->faces[0].function[ugkp::boundary::SolidFraction]=1;
 startup->functions.push_back({ugkp::boundary::FunctionKind::Table,ugkp::boundary::Bounds::Clamp,2,2,0});
 startup->terms.push_back({0,.25});startup->terms.push_back({1,.5});
 chmt::HostState configured;configured.gasMesh.gasBoundary=startup;
 assert(chmt::readCheckpoint(argv[1],model,configured,error));
 assert(configured.gasMesh.gasBoundary==startup);
 const auto& current=*configured.gasMesh.gasBoundary;
 ugkp::boundary::View currentView{current.faces.data(),current.functions.data(),current.terms.data(),1};
 assert(current.faces[0].solidMode==1);
 assert(ugkp::boundary::evaluate(currentView,current.faces[0].function[ugkp::boundary::SolidFraction],configured.time,0)==.4375);

}
""")
    exe=tmp_path/'checkpoint'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(R/'applications/CHMT'),'-I'+str(R/'common'),str(source),str(R/'applications/CHMT/restart/Checkpoint.C'),'-o',str(exe)],check=True)
    subprocess.run([exe,tmp_path/'saved'],check=True)


def test_dual_precision_wire_preserves_solid_policy_and_accepts_old_records(tmp_path):
    code=r"""
#include "gasBoundary/Wire.H"
#include <cassert>
#include <cmath>
using namespace ugkp::boundary;
int main(){
 Definition d;d.faces.resize(2);
 d.faces[0].solidMode=1;d.faces[0].value[SolidFraction]=.125;
 d.faces[0].function[SolidFraction]=0;
 d.faces[1].solidMode=0;d.faces[1].value[SolidFraction]=.25;
 assert(needsCheckpoint(d));
 d.functions={{FunctionKind::Table,Bounds::Clamp,0,2,0}};
 d.terms={{0,.125},{1,.25}};
 std::vector<int> iv;std::vector<double> rv;
 encode(d,iv,rv);
 assert(iv.size()==3+2*FaceInts+FunctionInts&&rv.size()==2*FaceReals+1+4);
 auto restored=decode(iv,rv);
 assert(restored.faces[0].solidMode==1&&restored.faces[1].solidMode==0);
 assert(restored.faces[0].function[SolidFraction]==0);
 assert(restored.faces[1].function[SolidFraction]==-1);
 assert(restored.faces[0].value[SolidFraction]==.125);
 View view{restored.faces.data(),restored.functions.data(),restored.terms.data(),2};
 assert(std::abs(evaluate(view,0,.5,0)-.1875)<1e-6);
 const auto originalI=iv;const auto originalR=rv;encode(restored,iv,rv);
 assert(iv==originalI&&rv==originalR);

 std::vector<int> oldI={1,0,0, 0,0,0,1,1,0, -1,-1,-1,-1,-1,-1, 0,-1};
 std::vector<double> oldR={0,0,-1e30,1,1, 1,2,3,4,1234,300};
 auto old=decode(oldI,oldR);
 assert(old.faces.size()==1&&old.faces[0].value[Pressure]==1234);
 assert(old.faces[0].value[Temperature]==300&&old.faces[0].solidMode==-1);
 assert(old.faces[0].function[SolidFraction]==-1);
 encode(old,iv,rv);auto upgraded=decode(iv,rv);
 assert(upgraded.faces[0].solidMode==-1&&upgraded.faces[0].value[Pressure]==1234);
}
"""
    for precision,include in [('fp64',R/'common'),('fp32',R/'precision/fp32/common')]:


        probe=code if precision=='fp64' else code.replace(
            'd.terms={{0,.125},{1,.25}};',
            'd.terms={Term::table(0,.125f),Term::table(1,.25f)};')
        source=tmp_path/(precision+'.cpp');source.write_text(probe)
        exe=tmp_path/precision
        subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(include),str(source),'-o',str(exe)],check=True)
        subprocess.run([exe],check=True)
