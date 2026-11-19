#include "gas_field_storage.hpp"

struct PrimitiveState : ugkwp::GasStateView<Real,GpuTime,ugkwp::SstCoefficients,GpuEnergy,GpuGeometry,GpuWallEnergy> {
    ugkwp::GasSpeciesState<Real,2> gasSpecies;
};

void setPrimitiveInput(PrimitiveState& state, int sample)
{
    const float nominal[5]{1.f, .51f, -.125f, .01f, 250000.f};
    float* conserved[]{state.rho, state.rhoUx, state.rhoUy, state.rhoUz, state.rhoE};
    for (int field=0; field<5; ++field) *conserved[field]=nominal[field];
    state.rhoMin=1e-4f;
    state.TgasMin=.25f;
    if (sample<15) {
        const float invalid[]{NAN, INFINITY, -INFINITY};
        *conserved[sample%5]=invalid[sample/5];
        return;
    }
    switch (sample) {
        case 15: *state.rho=0.f; break;
        case 16: *state.rho=-0.f; break;
        case 17: *state.rho=state.rhoMin/2; break;
        case 18: *state.rho=-1.f; break;
        case 19: *state.rho=state.rhoMin; break;
        case 20: *state.rhoE=-1.f; break;
        case 21: *state.rhoE=0.f; break;
        case 22: *state.rhoUx=std::numeric_limits<float>::max(); break;
        case 23: state.rhoMin=0.f; *state.rho=0.f; break;
        case 24: *state.rhoUy=-0.f; break;
        case 25: *state.rhoUz=std::numeric_limits<float>::denorm_min(); break;
        case 26: *state.rhoE=std::numeric_limits<float>::max(); break;
        case 27: state.TgasMin=0.f; *state.rhoE=0.f; break;
        case 28: *state.rho=808.f; *state.rhoUx=-2032.f; break;
        default:
            *state.rho=float(sample)*.006123f;
            *state.rhoUx=87535.6f/sample;
            *state.rhoUy=-.13f*sample;
            *state.rhoUz=sample*sample;
            *state.rhoE=float(sample)*648251.5f;
    }
}

int main()
{
    static_assert(sizeof(Real)==sizeof(std::uint32_t));
    PrimitiveState state;
    allocateGasTestStorage(state,1);
    state.nCells=1;
    state.gammaGas=1.4f;
    state.Rgas=287.f;
    for (bool strict : {false,true}) {
        for (int sample=0; sample<36; ++sample) {
            ugkp::diagnostics::DeviceContext diagnostics{};
            int status=0;
            state.diagnostics=&diagnostics;
            state.gasSpecies.cellStatus=&status;
            setPrimitiveInput(state,sample);
            for (auto field : {state.Ux,state.Uy,state.Uz,state.p,state.Tgas}) *field=-777.f;
            recoverGasPrimitiveCell(state,0,strict);
            const auto& fatal=diagnostics.firstFatal;
            std::printf("%d %d %d %u %u",int(strict),sample,status,fatal.sourceSiteId,unsigned(fatal.code));
            for (const auto& warning : diagnostics.warnings) {
                if (warning.count) {
                    std::printf(" w%u:%u:%u",unsigned(warning.sample.code),warning.sample.sourceSiteId,warning.count);
                }
            }
            for (auto field : {state.rho,state.rhoUx,state.rhoUy,state.rhoUz,state.rhoE,
                               state.Ux,state.Uy,state.Uz,state.p,state.Tgas}) {
                std::uint32_t bits;
                std::memcpy(&bits,field,sizeof(bits));
                std::printf(" %08x",bits);
            }
            std::puts("");
        }
    }
}
