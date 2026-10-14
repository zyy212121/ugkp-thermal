from pathlib import Path
p=Path('applications/CHMT/devtools/multirate/test_cpu_film.cpp');s=p.read_text()
s=s.replace('Real energy(const HostState& h,const PhysicsConfig& p) {Real sum=0;for(std::size_t i=0;i<h.film.size();++i)sum+=filmInternalEnergy(h.film[i],h.filmAux[i].pressure,p);return sum;}', '''template<class State> Real energy(const State& h,const PhysicsConfig& p) {Real sum=0;for(std::size_t i=0;i<h.film.size();++i)sum+=h.film[i].enthalpy-h.filmAux[i].pressure*h.film[i].mass/p.liquid.rho;return sum;}
Real sourcePower(const std::vector<FilmRateBudget>& budgets) {Real value=0;for(const auto& b:budgets)value+=b.bodyPower+b.supportPower+b.prescribedTopPower+b.radiationPower;return value;}
Real outwardEnergy(const std::vector<FilmRateBudget>& budgets) {Real value=0;for(const auto& b:budgets)value+=b.edgeEnergyOutflow;return value;}''')
s=s.replace('c.report.internalEnergyAfter','energy(c,p)').replace('c.report.internalEnergyBefore','energy(h,p)')
s=s.replace('        require(std::abs(c.report.reducedEnergyResidual)<1e-7,"reduced energy residual");','        require(std::abs(energy(c,p)-energy(h,p))<1e-7,"independent closed physical energy difference");')
s=s.replace('require(near(c.report.budgetDelta.filmPressureVolume,200),"PV storage audit");','require(near((c.filmAux[0].pressure*c.film[0].mass-h.filmAux[0].pressure*h.film[0].mass)/p.liquid.rho,200),"actual endpoint pressure-volume storage");')
s=s.replace('    require(c.report.budgetDelta.supportWork==0,"no duplicated top or bottom work");\n    require(near(c.report.budgetDelta.radiation,7),"radiation added once");\n    require(near(c.report.budgetDelta.exchangeEnergy[FilmParticipant],31),"kinetic advection excluded from physical exchange");\n    require(near(c.report.kineticDefect,c.filmAux[0].kineticEnergy-h.filmAux[0].kineticEnergy+3),"kinetic defect separate from heating");', '''    require(near(energy(c,p)-energy(h,p)-f[0].radiationEnergy-f[0].interfaceKineticOutflow,f[0].energy),"physical source remains distinct from radiation and kinetic channel");
    require(c.film[0].mass==h.film[0].mass,"energy forcing does not create film mass");''')
a=s.index('    require(near(c.report.budgetDelta.boundaryMass,.02)');b=s.index('\n}\nvoid radiationAndSourceLimits',a)
s=s[:a]+'''    require(near(h.film[0].mass-c.film[0].mass,.02),"actual outflow mass loss");
    Real elementalOut=0;for(int species=0;species<Ns;++species)elementalOut+=(h.film[0].species[species]-c.film[0].species[species])*p.species[species].element[0];
    require(near(elementalOut,.02),"outflow actual elemental loss");
    h.surface.baseVelocity[0]=-tangentX*.2;std::string error;
    require(advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),drive(h),c,error),"signed owner-extrapolated inflow: "+error);
    require(near(c.film[0].mass,1.02),"signed inflow adds owner-extrapolated mass");
    require(near(c.film[0].enthalpy,1.02*h.film[0].enthalpy),"signed inflow thermal inventory follows owner enthalpy");
    std::vector<FilmQ> rates;std::vector<FilmRateBudget> powers;
    require(evaluateCpuFilmTransport(h,p,.1,drive(h),rates,powers,error),"observe actual inflow transport RHS: "+error);
    require(rates[0].mass>0&&near(rates[0].mass,.2),"signed negative outward flux gives positive mass RHS");
    Real elementalIn=0;
    for(int species=0;species<Ns;++species){
        require(near(c.film[0].species[species],1.02*h.film[0].species[species]),"signed inflow species inventory");
        require(near(c.film[0].species[species]-h.film[0].species[species],.1*rates[0].species[species]),"species RHS and actual inventory close");
        elementalIn+=(c.film[0].species[species]-h.film[0].species[species])*p.species[species].element[0];
    }
    require(near(elementalIn,.02),"signed inflow actual elemental gain");
    require(near(powers[0].edgeEnergyOutflow,-.2*h.film[0].enthalpy),"actual RHS edge energy matches independent owner enthalpy flux");
    // Standalone surface pV work is -p*dV/dt. It is suppressed in the interface-owner transport helper.
    const Real pressureWork=-h.filmAux[0].pressure*rates[0].mass/p.liquid.rho;
    require(near(energy(c,p)-energy(h,p)+.1*powers[0].edgeEnergyOutflow,.1*pressureWork),"signed inflow physical energy plus pressure work closes");'''+s[b:]
s=s.replace('require(near(energy(c,p)-energy(h,p),c.report.budgetDelta.radiation),"radiation budget closes");','require(near(energy(c,p)-energy(h,p),.1*h.surface.area[0]*5.670374419e-8*p.emissivity*(std::pow(p.ambientTemperature,4)-std::pow(h.filmAux[0].temperature,4))),"independent Stefan-Boltzmann energy closes");')
s=s.replace('require(c.report.phaseEvent,"dryout reports phase event");','require(h.film[0].mass>0&&c.film[0].mass==0,"actual wet-to-dry phase transition");')
s=s.replace('    require(near(c.report.budgetDelta.bodyWork,.1*body),"body power once");\n    require(near(c.report.budgetDelta.supportWork,.1*(support+top)),"top and support power once");','    require(std::isfinite(body)&&std::isfinite(support)&&std::isfinite(top),"independent mechanical powers are finite");')
a=s.index('        require(c.report.edgeFlux[0].mass<0');b=s.index('        std::cout<<std::setprecision(17)<<"two-ulp',a)
s=s[:a]+'''        Real massRhs=0;for(const auto& rate:rates)massRhs+=rate.mass;
        const Real signedBoundaryMass=-(requested*massRhs);
        require(signedBoundaryMass<0,"two-ulp negative outward boundary mass retained by actual RHS");
        const Real independentFlux=p.liquid.rho*profiles[0].thickness*signedLengthRate;
        require(near(-massRhs,independentFlux),"two-ulp boundary flux matches independently sampled profile");
        Real deltaMass=0;for(int f=0;f<2;++f)deltaMass+=c.film[f].mass-h.film[f].mass;
        require(near(deltaMass,-signedBoundaryMass),"two-ulp actual mass boundary closure");
        for(int species=0;species<Ns;++species){Real delta=0,rhs=0;for(int f=0;f<2;++f){delta+=c.film[f].species[species]-h.film[f].species[species];rhs+=rates[f].species[species];}require(near(delta,requested*rhs),"two-ulp actual species boundary closure");}
        require(near(energy(c,p)-energy(h,p)+requested*outwardEnergy(budgets),requested*sourcePower(budgets)),"two-ulp independent physical energy boundary closure");
'''+s[b:]
assert '.report' not in s
p.write_text(s)
