from pathlib import Path
p=Path('applications/CHMT/devtools/multirate/test_cpu_film_curved.cpp');s=p.read_text();s=s.replace('namespace {','''namespace {
template<class State> Real physicalEnergy(const State& h,const PhysicsConfig& p) {Real value=0;for(std::size_t f=0;f<h.film.size();++f)value+=h.film[f].enthalpy-h.filmAux[f].pressure*h.film[f].mass/p.liquid.rho;return value;}
''',1)
s=s.replace('c.report.internalEnergyAfter-c.report.internalEnergyBefore','physicalEnergy(c,p)-physicalEnergy(h,p)')
s=s.replace('    require(absValue(c.report.reducedEnergyResidual)<1e-8,"curved conduction energy residual");','    require(c.film[0].mass==h.film[0].mass&&c.film[1].mass==h.film[1].mass,"curved conduction preserves actual mass");')
marker='    require(ok,"nonzero curved shear transport: "+error);'
s=s.replace(marker,marker+'''
    std::vector<FilmQ> rates;std::vector<FilmRateBudget> powers;
    require(evaluateCpuFilmTransport(h,p,dt,curvedDrive(h),rates,powers,error),"actual curved transport RHS: "+error);''')
s=s.replace('c.report.edgeFlux[0].mass','(-rates[owner].mass)',1).replace('c.report.edgeFlux[0].species[0]','(-rates[owner].species[0])').replace('c.report.edgeFlux[0].enthalpy','powers[owner].edgeEnergyOutflow').replace('c.report.rateBudget[owner].edgeKineticOutflow','powers[owner].edgeKineticOutflow').replace('c.report.rateBudget[other].edgeKineticOutflow','powers[other].edgeKineticOutflow')
s=s.replace('    require(absValue(c.report.budgetDelta.supportWork-dt*shearPower)<1e-10,"curved shear work and internal pressure work close");','    require(absValue(rates[owner].mass+rates[other].mass)<1e-12,"curved actual mass RHS antisymmetric");')
s=s.replace('    require(absValue(c.report.reducedEnergyResidual)<1e-8,"curved transport and work budget residual");','    require(absValue(powers[owner].edgeEnergyOutflow+powers[other].edgeEnergyOutflow)<1e-8,"curved actual internal edge heat flux antisymmetric");')
marker='        require(ok,"angular refinement transport: "+error);'
s=s.replace(marker,marker+'''
        std::vector<FilmQ> rates;std::vector<FilmRateBudget> powers;
        require(evaluateCpuFilmTransport(h,p,.001,curvedDrive(h),rates,powers,error),"actual angular transport RHS: "+error);''')
s=s.replace('c.report.edgeFlux[0].mass','(-rates[h.surface.edgeOwner[0]].mass)')
assert '.report' not in s;p.write_text(s)
p=Path('applications/CHMT/devtools/multirate/test_cpu_phase_lifecycle.cpp');s=p.read_text()
s=s.replace('    out.phaseEvent=film.report.phaseEvent;out.filmResidual=film.report.reducedEnergyResidual;out.storage=film.report.budgetDelta.filmPressureVolume;', '''    // Test-only physical inventory oracle, independent of removed runtime reports.
    std::vector<FilmQ> filmRates;std::vector<FilmRateBudget> filmPowers;
    require(evaluateCpuFilmTransport(filmState,p,dt,drive,filmRates,filmPowers,error),"lifecycle actual film RHS: "+error);
    for(std::size_t f=0;f<film.film.size();++f){
        out.phaseEvent=out.phaseEvent||((filmState.film[f].mass==0)!=(film.film[f].mass==0));
        const Real oldPV=filmState.filmAux[f].pressure*filmState.film[f].mass/p.liquid.rho;
        const Real newPV=film.filmAux[f].pressure*film.film[f].mass/p.liquid.rho;
        out.storage+=newPV-oldPV;
        out.filmResidual+=(film.film[f].enthalpy-newPV)-(filmState.film[f].enthalpy-oldPV)
            -dt*filmRates[f].enthalpy-force[f].energy-force[f].phaseEnergy-force[f].radiationEnergy
            -force[f].interfaceKineticOutflow-force[f].phaseKineticOutflow;
    }''')
s=s.replace('    require(absValue(c.report.budgetDelta.filmPressureVolume-20000)<1e-8,"variable pressure pV storage exactly once");\n    require(absValue(c.report.reducedEnergyResidual)<1e-7,"nonlinear film physical energy residual");','''    Real storage=0,residual=0;
    for(std::size_t f=0;f<c.film.size();++f){const Real pv0=h.filmAux[f].pressure*h.film[f].mass/p.liquid.rho,pv1=c.filmAux[f].pressure*c.film[f].mass/p.liquid.rho;
        storage+=pv1-pv0;residual+=(c.film[f].enthalpy-pv1)-(h.film[f].enthalpy-pv0)-force[f].energy;}
    require(absValue(storage-20000)<1e-8,"actual variable pressure pV storage exactly once");
    require(absValue(residual)<1e-7,"independent nonlinear film physical energy residual");''')
assert '.report' not in s;p.write_text(s)
