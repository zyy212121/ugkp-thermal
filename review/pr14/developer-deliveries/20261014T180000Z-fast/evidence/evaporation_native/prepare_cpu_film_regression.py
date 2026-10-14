from pathlib import Path
import difflib
S=Path('/workspace/scratch/98b6339ec81b/pr14_fast4_evap_analysis');R=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014')
p=R/'applications/CHMT/devtools/multirate/test_cpu_film.cpp';old=p.read_text();s=old
before='''    require(!advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),drive(h),c,error),"undeclared edge inflow refused");
    require(error.find("inflow")!=std::string::npos,"inflow policy diagnostic");'''
after='''    require(advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),drive(h),c,error),"signed owner-extrapolated inflow: "+error);
    require(near(c.film[0].mass,1.02),"signed inflow adds owner-extrapolated mass");
    require(near(c.film[0].enthalpy,1.02*h.film[0].enthalpy),"signed inflow thermal inventory follows owner enthalpy");
    require(near(c.report.budgetDelta.boundaryMass,-.02),"signed inflow boundary mass");
    require(near(c.report.budgetDelta.boundaryElements[0],-.02),"signed inflow boundary element");
    for(int species=0;species<Ns;++species){
        require(near(c.film[0].species[species],1.02*h.film[0].species[species]),"signed inflow species inventory");
        require(near(c.report.budgetDelta.boundarySpecies[species],-.02*h.film[0].species[species]),"signed inflow species boundary ledger");
    }
    require(c.report.edgeFlux[0].mass<0,"signed inflow has negative outward mass flux");
    require(near(c.report.edgeFlux[0].enthalpy,-.2*h.film[0].enthalpy),"signed inflow transports owner enthalpy");
    require(near(c.report.budgetDelta.boundaryEnergy,.1*(c.report.edgeFlux[0].enthalpy+c.report.rateBudget[0].edgeKineticOutflow)),"signed inflow full boundary energy ledger");
    require(near(c.report.internalEnergyAfter-c.report.internalEnergyBefore+.1*c.report.edgeFlux[0].enthalpy,c.report.budgetDelta.bodyWork+c.report.budgetDelta.supportWork+c.report.budgetDelta.radiation),"signed inflow reduced energy closure");'''
s=s.replace(before,after).replace("#include <iostream>","#include <iostream>\n#include <iomanip>")
function=r'''
void twoUlpPressureOpenEdge() {
    auto p=physics();p.liquid.cp0=1000;p.liquid.e0=100000;p.liquidViscosity=.01;p.liquidReferencePressure=171600;
    HostState h;const Vec3 n{1,0,0},t{0,1,0};
    const Real pressure0=178167.05269620317,pressure1=std::nextafter(std::nextafter(pressure0,0.0),0.0);
    h.surface.area=h.surface.oldArea={.25,.25};h.surface.normal={n,n};h.surface.baseVelocity={{},{}};h.surface.gasFace={0,1};
    h.surface.centre={{1,.25,.25},{1,.75,.25}};
    h.surface.edgeOwner={0,0};h.surface.edgeNeighbour={-1,1};h.surface.edgeLength={.5,.5};
    h.surface.edgeConormal={-t,t};h.surface.edgeOwnerOffset={t*-.25,t*.25};h.surface.edgeNeighbourOffset={{},t*-.25};h.surface.sweptEdgeArea={0,0};
    h.film.resize(2);h.filmAux.resize(2);std::vector<CpuFilmDrive> d(2);std::vector<CpuFilmForcing> forcing(2);
    for(int f=0;f<2;++f){const Real pressure=f?pressure1:pressure0;auto& q=h.film[f];q.mass=.0025;q.species[0]=q.mass*.2;q.species[1]=q.mass-q.species[0];q.enthalpy=q.mass*liquidH(p,350,pressure);h.filmAux[f].normal=n;require(recoverFilm(q,.25,pressure,p,h.filmAux[f]),"two-ulp fixture thermal inventory");d[f].pressure=d[f].endpointPressure=pressure;d[f].hasCoupledNormalTrace=true;forcing[f].topWorkIncluded=forcing[f].bottomWorkIncluded=true;}
    const Real fullDt=1.2207031249998656e-7;
    for(int halves:{0,30}){const Real requested=std::ldexp(fullDt,-halves);std::string error;std::vector<FilmQ> rates;std::vector<FilmRateBudget> budgets;std::vector<FilmAux> profiles;
        require(detail::completeCpuFilmTransport(h,p,requested,d,rates,budgets,error,&profiles,nullptr,nullptr,nullptr),"two-ulp actual profile observation: "+error);
        const Real signedLengthRate=h.surface.edgeLength[0]*dot(profiles[0].meanVelocity,h.surface.edgeConormal[0]);
        require(std::isfinite(signedLengthRate)&&signedLengthRate<0,"two-ulp pressure difference produces strictly negative finite open-edge rate");
        const bool ordinaryTransport=evaluateCpuFilmTransport(h,p,requested,d,rates,budgets,error);
        require(ordinaryTransport,"two-ulp finite signed transport: "+error);
        Real allowed=0;require(estimateCpuFilmStep(h,p,forcing,d,requested,requested,allowed,error),"two-ulp physical step estimate: "+error);require(allowed==requested,"two-ulp pressure variation does not shorten physically allowed dt");
        CpuFilmCandidate c;require(advanceCpuFilmCandidate(h,p,requested,forcing,d,c,error),"two-ulp signed candidate: "+error);
        require(c.report.edgeFlux[0].mass<0&&c.report.budgetDelta.boundaryMass<0,"two-ulp signed flux retained in boundary ledger");
        Real deltaMass=0;for(int f=0;f<2;++f)deltaMass+=c.film[f].mass-h.film[f].mass;
        require(near(deltaMass,-c.report.budgetDelta.boundaryMass),"two-ulp mass boundary closure");
        for(int species=0;species<Ns;++species){Real delta=0;for(int f=0;f<2;++f)delta+=c.film[f].species[species]-h.film[f].species[species];require(near(delta,-c.report.budgetDelta.boundarySpecies[species]),"two-ulp species boundary closure");}
        Real outwardHeat=0;for(const auto& b:c.report.rateBudget)outwardHeat+=requested*b.edgeEnergyOutflow;
        require(near(c.report.internalEnergyAfter-c.report.internalEnergyBefore+outwardHeat,c.report.budgetDelta.bodyWork+c.report.budgetDelta.supportWork+c.report.budgetDelta.radiation),"two-ulp reduced energy boundary closure");
        std::cout<<std::setprecision(17)<<"two-ulp signed edge half="<<halves<<" rate="<<signedLengthRate<<" requested="<<requested<<" allowed="<<allowed<<'\n';
    }
}
'''
s=s.replace('\n}\nint main(){','\n'+function+'\n}\nint main(){twoUlpPressureOpenEdge();')
s=s.replace('undeclaredInflowAndOutflow','signedOpenEdgeInflowAndOutflow')
(S/'test_cpu_film_signed_open_edge.cpp').write_text(s);(S/'test_cpu_film_signed_open_edge.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='production/test_cpu_film.cpp',tofile='scratch/test_cpu_film_signed_open_edge.cpp')))
