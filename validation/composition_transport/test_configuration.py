from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_models_and_identity(tmp_path):
 source=tmp_path/'configuration.cpp'
 source.write_text(r"""
#include "gasTransport/MolecularTransportIO.H"
#include <cassert>
int main(int argc,char**argv){
 assert(argc==2);std::ifstream f(argv[1]);std::string text((std::istreambuf_iterator<char>(f)),{});
 auto parse=[](const std::string& t){return ugkwp::parseMolecularTransport(t,{"Air","CO2"});};
 auto a=parse(text);assert(a.hash);double y[]={.25,.75},scaled[]={25.,75.};
 assert(ugkwp::molecularViscosity<2>(a.view(),0.,273.,y)==ugkwp::molecularViscosity<2>(a.view(),0.,273.,scaled));
 assert(ugkwp::molecularPrandtl<2>(a.view(),0.,y)==ugkwp::molecularPrandtl<2>(a.view(),0.,scaled));
 const auto change=[&](const char* old,const char* replacement){auto s=text;auto p=s.find(old);assert(p!=std::string::npos);s.replace(p,std::string(old).size(),replacement);return s;};
 for(auto pair:std::vector<std::pair<const char*,const char*>>{{"17.10e-6","17.11e-6"},{"114 220","114 221"},{".720 .064",".721 .064"},{"273","274"},{"1.81","1.82"},{"pressureExponent -1","pressureExponent -.9"},{"101325","101326"},{"DRef 5.75e-10","DRef 5.76e-10"},{"TRef 1;","TRef 2;"}})assert(parse(change(pair.first,pair.second)).hash!=a.hash);
 for(auto pair:std::vector<std::pair<const char*,const char*>>{{"species CO2;","species missing;"},{".720 .064","1 -5 5"},{"114 220","-1 220"},{"17.10e-6 -4.65e-6 1.20e-6","0"},{"TRef 273","TRef 0"},{"pRef 101325","pRef 0"},{"DRef 5.75e-10","DRef -1"},{"temperatureExponent 1.81","temperatureExponent nan"}}){bool rejected=false;try{parse(change(pair.first,pair.second));}catch(const std::exception&){rejected=true;}assert(rejected);}
 // Each optional model is independent of the viscosity selection.
 auto independent=parse(change("viscosityModel massFractionSutherland;","viscosityModel const;"));assert(independent.view().diffusion);assert(independent.view().prandtlSpecies==1);
 assert(ugkwp::molecularViscosity<2>(independent.view(),3.,273.,y)==3.);
 auto copy=a;auto moved=std::move(copy);assert(moved.view().diffusion==moved.diffusion.data());assert(moved.view().diffusion!=a.view().diffusion);
 assert(ugkwp::parseMolecularTransport("viscosityModel const;").hash==0);
}
""")
 exe=tmp_path/'configuration'
 subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
 subprocess.run([str(exe),str(Path(__file__).with_name('transport.dict'))],check=True)
