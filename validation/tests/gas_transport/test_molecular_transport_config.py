"""Canonical parser, thermodynamic identity and degenerate composition limits."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def test_transport_configuration_and_composition_limits(tmp_path):
    source=tmp_path/'config.cpp'
    source.write_text(r"""
#include "gasTransport/MolecularTransportIO.H"
#include <cassert>
using namespace ugkwp;
int main(){
 const std::string text="viscosityModel sutherland; species (A B); sutherland { A {muRef 1.8e-5;TRef 300;S 110.4;molarMass .028;} B {muRef 2.1e-5;TRef 300;S 127;molarMass .032;} }";
 const auto model=parseMolecularTransport(text,{"A","B"});
 assert(parseMolecularTransport("").model==MolecularViscosityModel::Constant);
 assert(parseMolecularTransport("viscosityModel const;").hash==0);
 auto rejects=[&](const std::string& t,const std::vector<std::string>& names){try{parseMolecularTransport(t,names);}catch(const std::exception&){return true;}return false;};
 assert(rejects(text,{"B","A"}));assert(rejects(text,{}));
 auto invalid=text;invalid.replace(invalid.find("1.8e-5"),6,"-1e-5");assert(rejects(invalid,{"A","B"}));
 assert(rejects("viscosityModel unknown;",{}));
 std::vector<SpeciesThermoData<double>> thermodynamics(2);thermodynamics[0].molarMass=.028;thermodynamics[1].molarMass=.032;
 validateTransportMolarMasses(model,thermodynamics);thermodynamics[1].molarMass=.028;
 bool mismatch=false;try{validateTransportMolarMasses(model,thermodynamics);}catch(const std::exception&){mismatch=true;}assert(mismatch);
 const double pureA[]={1,0},pureB[]={0,1},mix[]={.2,.8},scaled[]={200,800},empty[]={0,0},negative[]={-.1,1.1};
 assert(molecularViscosity<2>(model.view(),0.,300.,pureA)==1.8e-5);
 assert(molecularViscosity<2>(model.view(),0.,300.,pureB)==2.1e-5);
 const double mu=molecularViscosity<2>(model.view(),0.,900.,mix);
 assert(std::abs(molecularViscosity<2>(model.view(),0.,900.,scaled)/mu-1)<5e-15);
 assert(std::isnan(molecularViscosity<2>(model.view(),0.,900.,empty)));
 assert(std::isnan(molecularViscosity<2>(model.view(),0.,900.,negative)));
 assert(std::isnan(molecularViscosity<2>(model.view(),0.,0.,mix)));
 assert(molecularViscosity<2>(MolecularViscosityView<double>{},0.,0.,empty)==0.);
}
""")
    exe=tmp_path/'config'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
