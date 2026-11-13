#pragma once
// Shared DATA ONLY. Both historical writers use their own unmodified source
// closures and their old types. No new serializer generates the legacy input.
#define CHMT_ENERGY_FIELDS(X) \
 X(gas[0].energy,"J",103001.123456789) \
 X(solid[0].energy,"J",203002.234567891) \
 X(film[0].enthalpy,"J",3003.345678912) \
 X(sst[0].rhoK,"J",4.456789123) \
 X(gasTemperature[0],"K",705.567891234) \
 X(filmAux[0].pressure,"Pa",100006.678912345) \
 X(filmAux[0].temperature,"K",507.789123456) \
 X(normalEnthalpy[0],"J",8008.891234567) \
 X(normalAcceptedWallHeat[0],"J",-19.912345678) \
 X(surface.bottomTemperature[0],"K",510.123456789) \
 X(surface.topTemperature[0],"K",611.234567891) \
 X(surface.bottomHeatFlux[0],"W/m2",1212.345678912) \
 X(surface.topHeatFlux[0],"W/m2",-1313.456789123) \
 X(gasMesh.boundaryPrimitive[0].temperature,"K",714.567891234) \
 X(gasMesh.boundaryPrimitive[0].pressure,"Pa",100015.678912345) \
 X(gasMesh.boundaryPrimitive[0].soundSpeed,"m/s",516.789123456) \
 X(ledger[0].energy,"J",17.891234567) \
 X(ledger[0].conductive,"J",-18.912345678) \
 X(ledger[0].advective,"J",19.123456789) \
 X(ledger[0].pressureWork,"J",20.234567891) \
 X(ledger[0].viscousWork,"J",-21.345678912) \
 X(ledger[0].radiation,"J",22.456789123) \
 X(ledger[0].liquidKineticAdvection,"J",-23.567891234) \
 X(filmStorage[0].oldPV,"J",24.678912345) \
 X(filmStorage[0].newPV,"J",25.789123456) \
 X(gasSstAudit.transportK,"J",-26.891234567) \
 X(gasSstAudit.sourceK,"J",27.912345678) \
 X(gasSstAudit.constraintK,"J",-28.123456789) \
 X(budget.boundaryEnergy,"J",29.234567891) \
 X(budget.supportWork,"J",-30.345678912) \
 X(budget.bodyWork,"J",31.456789123) \
 X(budget.radiation,"J",-32.567891234) \
 X(budget.exchangeEnergy[0],"J",33.678912345) \
 X(budget.exchangeEnergy[1],"J",-34.789123456) \
 X(budget.exchangeEnergy[2],"J",35.891234567) \
 X(budget.exchangeEnergy[3],"J",-36.912345678) \
 X(budget.filmPressureVolume,"J",37.123456789) \
 X(budget.filmKineticAdvection,"J",-38.234567891) \
 X(budget.filmKineticStorage,"J",39.345678912) \
 X(budget.filmReducedResidual,"J",-40.456789123) \
 X(budget.filmKineticDefect,"J",41.567891234) \
 X(budget.numericalEnergyResidual,"J",-42.678912345) \
 X(budget.turbulenceInventory,"J",43.789123456) \
 X(budget.turbulenceProduction,"J",44.891234567) \
 X(budget.turbulenceDissipation,"J",-45.912345678) \
 X(budget.turbulenceBoundaryFlux,"J",46.123456789) \
 X(gasChemistryAudit[0].energyResidual,"J",-47.234567890999998) \
 X(particles[0].energy,"J",1401.2345678900001) \
 X(particles[0].contactMechanicalEnergy,"J",0.234567891)

#define CHMT_PHYSICAL_FIELDS(X) \
 X(gas[0].mass,"kg",1.23456789012345) \
 X(gas[0].species[0],"kg",0.723456789012345) \
 X(gas[0].species[1],"kg",0.511111101111105) \
 X(solid[0].condensed[0],"kg",2.34567890123456) \
 X(solid[0].pore[0],"kg",0.034567890123456) \
 X(solid[0].porosity,"1",0.145678901234567) \
 X(film[0].mass,"kg",0.256789012345678) \
 X(film[0].species[0],"kg",0.256789012345678) \
 X(ledger[0].mass,"kg",0.000123456789012345) \
 X(ledger[0].poreSweep[0],"kg",0.000023456789012345) \
 X(sst[0].rhoOmega,"kg/s",3.45678901234567) \
 X(gasSstAudit.sourceOmega,"kg/s",-4.56789012345678) \
 X(particleSourceResidualMass[0],"kg",0.000000123456789012345) \
 X(gasMesh.boundaryPrimitive[0].rho,"kg/m3",1.34567890123456) \
 X(gasMesh.boundaryPrimitive[0].Y[0],"1",0.712345678901234) \
 X(gasMesh.boundaryPrimitive[0].Y[1],"1",0.287654321098766) \
 X(solid[0].progress[0],"1",0.234567891) \
 X(gasVoidFraction[0],"1",0.91234567889999996) \
 X(normalLiquidFraction[0],"1",0.34567891229999997) \
 X(surface.normalLayerMass[0],"kg",0.4567891234) \
 X(surface.normalPressure[0],"Pa",100234.567891) \
 X(surface.normalPressureRate[0],"Pa/s",123.45678909999999) \
 X(ledger[0].species[0],"kg",7.2345678909999995e-05) \
 X(ledger[0].species[1],"kg",5.1111110099999998e-05) \
 X(ledger[0].condensed[0],"kg",3.2345678909999998e-05) \
 X(ledger[0].pore[0],"kg",4.2345678909999998e-05) \
 X(budget.boundaryMass,"kg",-0.01234567891) \
 X(budget.exchangeMass[0],"kg",0.023456789120000002) \
 X(budget.boundarySpecies[0],"kg",-0.034567891230000002) \
 X(gasChemistryAudit[0].speciesMassChange[0],"kg",1.234567891e-05) \
 X(gasChemistryAudit[0].massResidual,"kg",-1.2345678909999999e-07) \
 X(gasChemistryAudit[0].maximumElementResidual,"kg",2.34567891e-08) \
 X(particles[0].mass,"kg",0.00123456789012345) \
 X(particles[0].diameter,"m",0.00034567891230000001) \
 X(particles[0].contactArea,"m2",1.2345678910000001e-06) \
 X(particles[0].thermal.theta,"1",0.234567891) \
 X(particles[0].thermal.contactMaximumArea,"m2",2.345678912e-06) \
 X(particles[0].thermal.contactPeakFraction,"1",0.345678912) \
 X(particles[0].thermal.coldFrozenArea,"m2",3.4567891230000001e-06) \
 X(particles[0].thermal.cold2DFrozenArea,"m2",4.5678912340000001e-06) \
 X(filmAux[0].normalVelocity,"m/s",0.123456789) \
 X(filmAux[0].solidNormalVelocity,"m/s",-0.234567891) \
 X(gas[0].momentum.x,"kg m/s",1.234567891) \
 X(gas[0].momentum.y,"kg m/s",-2.2345678910000002) \
 X(gas[0].momentum.z,"kg m/s",3.2345678910000002) \
 X(gasMesh.boundaryPrimitive[0].velocity.x,"m/s",1.234567891) \
 X(gasMesh.boundaryPrimitive[0].velocity.y,"m/s",-2.2345678910000002) \
 X(gasMesh.boundaryPrimitive[0].velocity.z,"m/s",3.2345678910000002) \
 X(filmAux[0].baseVelocity.x,"m/s",1.234567891) \
 X(filmAux[0].baseVelocity.y,"m/s",-2.2345678910000002) \
 X(filmAux[0].baseVelocity.z,"m/s",3.2345678910000002) \
 X(filmAux[0].meanVelocity.x,"m/s",1.234567891) \
 X(filmAux[0].meanVelocity.y,"m/s",-2.2345678910000002) \
 X(filmAux[0].meanVelocity.z,"m/s",3.2345678910000002) \
 X(filmAux[0].topVelocity.x,"m/s",1.234567891) \
 X(filmAux[0].topVelocity.y,"m/s",-2.2345678910000002) \
 X(filmAux[0].topVelocity.z,"m/s",3.2345678910000002) \
 X(filmAux[0].normal.x,"1",1.234567891) \
 X(filmAux[0].normal.y,"1",-2.2345678910000002) \
 X(filmAux[0].normal.z,"1",3.2345678910000002) \
 X(surface.baseVelocity[0].x,"m/s",1.234567891) \
 X(surface.baseVelocity[0].y,"m/s",-2.2345678910000002) \
 X(surface.baseVelocity[0].z,"m/s",3.2345678910000002) \
 X(ledger[0].momentum.x,"kg m/s",1.234567891) \
 X(ledger[0].momentum.y,"kg m/s",-2.2345678910000002) \
 X(ledger[0].momentum.z,"kg m/s",3.2345678910000002) \
 X(budget.boundaryMomentum.x,"kg m/s",1.234567891) \
 X(budget.boundaryMomentum.y,"kg m/s",-2.2345678910000002) \
 X(budget.boundaryMomentum.z,"kg m/s",3.2345678910000002) \
 X(particles[0].velocity.x,"m/s",1.234567891) \
 X(particles[0].velocity.y,"m/s",-2.2345678910000002) \
 X(particles[0].velocity.z,"m/s",3.2345678910000002) \
 X(particles[0].contactRelativeVelocity.x,"m/s",1.234567891) \
 X(particles[0].contactRelativeVelocity.y,"m/s",-2.2345678910000002) \
 X(particles[0].contactRelativeVelocity.z,"m/s",3.2345678910000002) \
 X(particles[0].thermal.savedVelocity[0],"m/s",2.3456789119999999) \
 X(particles[0].thermal.savedVelocity[1],"m/s",3.3456789119999999) \
 X(particles[0].thermal.savedVelocity[2],"m/s",4.3456789120000003)

#define CHMT_INDEPENDENT_FIELDS(X) \
 X(gasMesh.points[0].x,"m",123456789.123456789) \
 X(gasMesh.points[0].y,"m",-987654321.987654321) \
 X(gasMesh.volumes[0],"m3",0.0123456789012345) \
 X(surface.area[0],"m2",0.234567890123456) \
 X(surface.gasDistance[0],"m",0.000034567890123456) \
 X(solidSweepRemainder[0],"m3",-0.00000000000000123456789012345) \
 X(time,"s",0.29500000000000004) \
 X(radiationTime,"s",0.29400000000000004) \
 X(nextDt,"s",0.0000123456789012345) \
 X(lastAcceptedDt,"s",0.0000234567890123456) \
 X(gasStages[0].interval,"s",0.0000345678901234567) \
 X(particles[0].position.x,"m",11.123456789012344) \
 X(particles[0].position.y,"m",2.234567890123456) \
 X(particles[0].position.z,"m",3.3456789012345669) \
 X(particles[0].contactAge,"s",0.123456789012345) \
 X(particles[0].contactDuration,"s",0.234567890123456) \
 X(particles[0].thermal.coldContactAge,"s",0.123456789012345)

inline void chmtSeed(chmt::ModelConfig& model,chmt::HostState& h,bool legacyCarry){
 static_assert(chmt::Ns>=2,"fixture needs two independently nonzero species");
 model.physics.enableParticles=true;
 h.gas.resize(1);h.solid.resize(1);h.film.resize(1);h.sst.resize(1);
 h.gasTemperature.resize(1);h.filmAux.resize(1);h.normalEnthalpy.resize(1);
 h.normalAcceptedWallHeat.resize(1);h.surface.bottomTemperature.resize(1);
 h.surface.topTemperature.resize(1);h.surface.bottomHeatFlux.resize(1);h.surface.topHeatFlux.resize(1);
 h.gasMesh.boundaryPrimitive.resize(1);h.gasMesh.points.resize(1);h.gasMesh.volumes.resize(1);
 h.surface.area.resize(1);h.surface.gasDistance.resize(1);h.solidSweepRemainder.resize(1);
 h.ledger.push_back(chmt::ExchangePacket{});h.filmStorage.resize(1);
 h.particleSourceFaces={0};h.particleSourceResidualMass.resize(1);
 h.gasChemistryAudit.resize(1);h.particles.resize(1);h.gasVoidFraction.resize(1);
 h.normalLiquidFraction.resize(1);h.surface.normalLayerMass.resize(1);
 h.surface.normalPressure.resize(1);h.surface.normalPressureRate.resize(1);h.surface.baseVelocity.resize(1);
 #define CHMT_SET(field,unit,value) h.field=static_cast<std::remove_reference_t<decltype(h.field)>>(value);
 CHMT_ENERGY_FIELDS(CHMT_SET)
 CHMT_PHYSICAL_FIELDS(CHMT_SET)
 CHMT_INDEPENDENT_FIELDS(CHMT_SET)
 #undef CHMT_SET
 h.acceptedSteps=17;h.rejectedSteps=3;h.commitSequence=17;
 h.particles.resize(1);auto& q=h.particles[0];q.id=90210;q.cell=0;q.state=0;

 q.thermal.initialized=true;q.thermal.temperature=701.123456789;

 for(int k=0;k<8;++k){q.thermal.coldNodeSpecificEnthalpy[k]=float(10001.25+17.5*k);
 q.thermal.coldRingSolidMass[k]=0.00123456789*(k+1);q.thermal.cold2DRingContactAge[k]=0.0023456789012345*(k+1);}
 for(int k=0;k<64;++k)q.thermal.cold2DNodeSpecificEnthalpy[k]=float(20003.5+7.25*k);
 #if defined(CHMT_LEGACY_SEED)
 q.thermal.specificEnthalpyResidual=legacyCarry?0.125:0.;
 #else
 (void)legacyCarry;
 #endif
}
