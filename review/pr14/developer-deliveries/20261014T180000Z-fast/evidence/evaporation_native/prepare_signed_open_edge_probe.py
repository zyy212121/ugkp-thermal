from pathlib import Path
import difflib
S=Path('/workspace/scratch/98b6339ec81b/pr14_fast4_evap_analysis');R=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014')
a=(R/'applications/CHMT/film/CpuFilmDriver.C').read_text();b=a.replace('        if(!Terminal&&(!finite(lengthRate)||(neighbour<0&&lengthRate<0)))return numericReject(error,"undeclared open-edge inflow",edge,numericalFailure);','        // Open edges use the existing owner extrapolation for either flow direction.\n        if(!Terminal&&!finite(lengthRate))return numericReject(error,"non-finite film edge transport rate",edge,numericalFailure);');(S/'CpuFilmDriver_signed_open_edge.C').write_text(b);(S/'signed_open_edge_scratch.patch').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='production/CpuFilmDriver.C',tofile='scratch/CpuFilmDriver_signed_open_edge.C')))
s=(S/'checkpoint_film_probe.cpp').read_text()
helper=r'''
void recordCandidate(const std::string& label,const chmt::CpuFilmCandidate& c,std::ostream& snapshots){
    using namespace chmt;Encoder e;e(c.film);e(c.filmAux);e(c.report.budgetDelta);e(c.report.internalEnergyBefore);e(c.report.internalEnergyAfter);e(c.report.reducedEnergyResidual);e(c.report.kineticDefect);e(c.report.numericalMassRoundoff);e(c.report.phaseEvent);e(c.report.edgeFlux);
    for(const auto& b:c.report.rateBudget){e(b.bodyPower);e(b.supportPower);e(b.prescribedTopPower);e(b.radiationPower);e(b.edgeEnergyOutflow);e(b.edgeKineticOutflow);e(b.interfaceKineticOutflow);e(b.dissipationPower);}
    Real mass=0,enthalpy=0;for(const auto& q:c.film){mass+=q.mass;enthalpy+=q.enthalpy;}
    const auto hash=checksum(e.data.data(),e.data.size());std::cout<<"SNAPSHOT label="<<label<<" checksum="<<hash<<" mass="<<mass<<" enthalpy="<<enthalpy<<" boundaryMass="<<c.report.budgetDelta.boundaryMass<<" boundaryEnergy="<<c.report.budgetDelta.boundaryEnergy<<" residual="<<c.report.reducedEnergyResidual<<"\n";
    snapshots<<std::setprecision(17)<<"{\"label\":\""<<label<<"\",\"checksum\":"<<hash<<",\"film\":";JsonObserver jf(snapshots);jf(c.film);snapshots<<",\"filmAux\":";JsonObserver ja(snapshots);ja(c.filmAux);snapshots<<",\"budget\":";JsonObserver jb(snapshots);jb(c.report.budgetDelta);snapshots<<",\"edgeFlux\":";JsonObserver je(snapshots);je(c.report.edgeFlux);snapshots<<",\"internalEnergyBefore\":"<<c.report.internalEnergyBefore<<",\"internalEnergyAfter\":"<<c.report.internalEnergyAfter<<",\"reducedEnergyResidual\":"<<c.report.reducedEnergyResidual<<",\"kineticDefect\":"<<c.report.kineticDefect<<",\"numericalMassRoundoff\":"<<c.report.numericalMassRoundoff<<",\"phaseEvent\":"<<(c.report.phaseEvent?"true":"false")<<",\"rateBudget\":[";
    bool first=true;for(const auto& b:c.report.rateBudget){if(!first)snapshots<<',';first=false;snapshots<<"{\"bodyPower\":"<<b.bodyPower<<",\"supportPower\":"<<b.supportPower<<",\"prescribedTopPower\":"<<b.prescribedTopPower<<",\"radiationPower\":"<<b.radiationPower<<",\"edgeEnergyOutflow\":"<<b.edgeEnergyOutflow<<",\"edgeKineticOutflow\":"<<b.edgeKineticOutflow<<",\"interfaceKineticOutflow\":"<<b.interfaceKineticOutflow<<",\"dissipationPower\":"<<b.dissipationPower<<'}';}snapshots<<"]}\n";
}
'''
s=s.replace('int main(int argc,char** argv){',helper+'\nint main(int argc,char** argv){')
s=s.replace('    const Real fullDt=', '    std::ofstream snapshots(std::string(argv[2])+".snapshots.jsonl");\n    const Real fullDt=')
s=s.replace('            numerical=false;ok=detail::completeCpuFilmCandidate', '            if(ok)recordCandidate(mode+"-advance-half"+std::to_string(power),next,snapshots);\n            numerical=false;ok=detail::completeCpuFilmCandidate')
s=s.replace('<<" op=complete ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\\n";', '<<" op=complete ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\\n";\n            if(ok)recordCandidate(mode+"-complete-half"+std::to_string(power),next,snapshots);')
controls=r'''
    for(const std::string control:{"mass_donor_exhausted","species_donor_exhausted","missing_coupled_trace","missing_film_aux","nonfinite_length_rate","ordinary_outward"}){
        HostState h=base;h.surface.sweptEdgeArea.assign(h.surface.edgeOwner.size(),0);std::vector<CpuFilmDrive> d(h.film.size());std::vector<CpuFilmForcing> f(h.film.size());
        for(std::size_t i=0;i<d.size();++i){const auto& a=h.filmAux[i];const Vec3 n=h.surface.normal[i];d[i].pressure=d[i].endpointPressure=base.filmAux[0].pressure;d[i].hasCoupledNormalTrace=true;d[i].bottomNormalVelocity=2*dot(a.meanVelocity,n)-dot(a.topVelocity,n);d[i].topNormalVelocity=dot(a.topVelocity,n);d[i].interfaceNormalVelocity=a.normalVelocity;d[i].solidNormalVelocity=a.solidNormalVelocity;f[i].topWorkIncluded=f[i].bottomWorkIncluded=true;}
        if(control=="mass_donor_exhausted"){f[0].mass=f[0].species[0]=-2*h.film[0].mass;f[0].withdrawnMass=f[0].withdrawnSpecies[0]=-f[0].mass;}
        if(control=="species_donor_exhausted"){f[0].mass=f[0].species[1]=-.5*h.film[0].mass;f[0].withdrawnMass=f[0].withdrawnSpecies[1]=-f[0].mass;}
        if(control=="missing_coupled_trace")d[0].hasCoupledNormalTrace=false;
        if(control=="missing_film_aux")h.filmAux.clear();
        if(control=="nonfinite_length_rate")h.surface.sweptEdgeArea[0]=std::numeric_limits<Real>::max();
        if(control=="ordinary_outward"){
            const Vec3 v=h.surface.edgeConormal[0]*.2;for(auto& x:h.surface.baseVelocity)x=v;
            h.surface.edgeOwner.resize(1);h.surface.edgeNeighbour.resize(1);h.surface.edgeLength.resize(1);h.surface.edgeConormal.resize(1);h.surface.edgeOwnerOffset.resize(1);h.surface.edgeNeighbourOffset.resize(1);h.surface.sweptEdgeArea.resize(1);
        }
        CpuFilmCandidate next;bool numerical=false;const bool ok=advanceCpuFilmCandidate(h,model.physics,fullDt,f,d,next,error,&numerical);
        std::cout<<"CONTROL name="<<control<<" ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\n";if(ok)recordCandidate("control-"+control,next,snapshots);
    }
'''
pos=s.rfind('\n}');s=s[:pos]+controls+s[pos:];(S/'checkpoint_film_boundary_probe.cpp').write_text(s)
