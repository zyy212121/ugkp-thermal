// Checkpoint 2 regression fixture; compile now, execute in the full batch only.
// Reuse the actual mixed-state production-kernel host harness.
#define main mixed_divergence_fixture_main
#include "mixed_divergence_ale.cpp"
#undef main

static unsigned warnings(const Fixture& f) {
    unsigned count=0;
    for(const auto& slot:f.diagnostics.warnings) count+=slot.count;
    return count;
}

static void waleRegression() {
    // Independent double oracle for this diagonal family, evaluated only when
    // the full checkpoint batch executes this fixture.
    const double ratio=std::pow(2./3.,1.5)
        /(std::pow(2.,2.5)+std::pow(2./3.,1.25));
    for(const auto magnitude:{1e-4f,1.f,1e7f,1e30f}) {
        Fixture f;auto& s=f.s;
        s.turbulenceModel=1;s.lesDeltaCoeff=1;s.waleCw=.325f;
        s.cellLength[0]=magnitude==1e30f?1e-10f:1e-5f;
        s.gradUxX[0]=magnitude;s.gradUyY[0]=-magnitude;
        const double dimensionalDenominator=std::pow(double(magnitude),5)
            *(std::pow(2.,2.5)+std::pow(2./3.,1.25));
        const double expected=dimensionalDenominator>OfSmall
            ? double(s.waleCw)*s.waleCw*double(s.cellLength[0])*s.cellLength[0]*magnitude*ratio : 0.;
        computeGasEddyViscosityKernel(&s);
        require(!f.diagnostics.fatalClaim,"finite WALE gradient triggered fatal");
        require(std::isfinite(s.nut[0]),"WALE viscosity is not finite");
        require(expected==0. ? s.nut[0]==0.f :
            std::abs(double(s.nut[0])-expected)<=32*FLT_EPSILON*expected,
            "normalized WALE differs from dimensional double oracle");
        if(magnitude==1e7f) {
            const GpuReal direct=ugkwp::waleNut(s.waleCw,GpuReal(s.cellLength[0]),
                2e14f,GpuReal((2./3.)*1e28),OfSmall);
            require(std::isfinite(direct)&&std::abs(double(direct)-expected)<=32*FLT_EPSILON*expected,
                "invariant-based WALE entry still overflows");
            require(std::abs(double(s.nut[0])-9.185591927e-6)<5e-11,
                "reported WALE overflow example is not repaired");
        }
    }
    Fixture zero;zero.s.turbulenceModel=1;zero.s.lesDeltaCoeff=1;zero.s.waleCw=.325f;
    computeGasEddyViscosityKernel(&zero.s);
    require(zero.s.nut[0]==0&&!zero.diagnostics.fatalClaim,"zero WALE gradient changed semantics");
    Fixture bad;bad.s.turbulenceModel=1;
    bad.s.gradUxX[0]=std::numeric_limits<GpuReal>::infinity();
    computeGasEddyViscosityKernel(&bad.s);
    require(bad.diagnostics.fatalClaim&&bad.diagnostics.firstFatal.sourceSiteId==30300,
        "nonfinite gradient lost its existing diagnostic");
    Fixture badCoefficient;badCoefficient.s.turbulenceModel=1;
    badCoefficient.s.waleCw=std::numeric_limits<GpuReal>::quiet_NaN();
    computeGasEddyViscosityKernel(&badCoefficient.s);
    require(badCoefficient.diagnostics.fatalClaim&&badCoefficient.diagnostics.firstFatal.sourceSiteId==30326,
        "nonfinite WALE coefficient lost its existing diagnostic");
}

static void speciesRoundoffRegression() {
    const GpuReal reference=-0.048840794712305069f;
    const GpuReal observed=-0.048840798437595367f;
    require(std::nextafter(reference,-1.f)==observed,"reported values are not adjacent floats");
    for(const bool native:{false,true}) {
        Fixture f;auto& s=f.s;
        GpuReal density[2]={1.f,1.f},flux[2]={observed,0.f};
        s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
        s.gasSpecies.rho=density;s.gasSpecies.flux=flux;
        s.gasSpecies.densityClosureTolerance=1e-5f;
        s.gasPhiRho[0]=reference;
        applyGasFluxDivergenceByCellKernel(&s,1e-3,native);
        require(!f.diagnostics.fatalClaim&&warnings(f)==0,
            "one-ULP species face sum was diagnosed as a conservation failure");

        Fixture periodic;auto& q=periodic.s;
        int status[2]={};GpuReal paired[4]={observed,-observed,0.f,0.f};
        q.nFaces=2;q.facePeriodicPair[0]=1;q.facePeriodicPair[1]=0;
        q.faceOwner[0]=q.faceOwner[1]=0;
        q.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
        q.gasSpecies.rho=density;q.gasSpecies.flux=paired;q.gasSpecies.faceStatus=status;
        q.gasPhiRho[0]=reference;q.gasPhiRho[1]=-reference;
        enforcePeriodicGasFluxAntisymmetryKernel(&q,native);
        require(!periodic.diagnostics.fatalClaim&&warnings(periodic)==0,
            "one-ULP periodic species sum was diagnosed as a conservation failure");
        require(paired[0]==-paired[1]&&paired[2]==-paired[3],"periodic antisymmetry lost");

        // Exercise the RK closure repair for multiple exactly stored species
        // states. The broad physical density tolerance is unchanged.
        for(int i=1;i<32;++i) {
            Fixture rk;auto& r=rk.s;
            GpuReal initial[2]={1.f,GpuReal(i)*.01f};
            GpuReal stage[2]={GpuReal(i)*.03f,.75f};
            r.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
            r.gasSpecies.rho=stage;r.gasSpecies.initial=initial;
            r.gasSpecies.densityClosureTolerance=1e-5f;
            r.rhoNext[0]=initial[0]+initial[1];r.rho[0]=stage[0]+stage[1];
            blendGasConservativeStateKernel(&r,1./3.,2./3.,native);
            require(!rk.diagnostics.fatalClaim&&warnings(rk)==0,
                "FP32 RK roundoff caused a conservation warning");
        }
    }
}

int main() {
    waleRegression();speciesRoundoffRegression();
    std::cout<<"PASS FP32 WALE and species roundoff regressions\n";
}
