// Real CUDA regression: include the complete production owner and device paths.
// No CUDA API shim, extracted numerical substitute, or frozen buggy oracle.
#include "GpuResidentStrict.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

void check(bool condition, const char* message)
{
    if (!condition) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
void cudaCheck(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}
void nativeCheck(int status)
{
    if (status) { std::fprintf(stderr, "%s\n", lastError); std::exit(3); }
}
int configureCold(DeviceState* s, const unsigned char* mask, int heat = 1)
{
    return ugkwpGpuResidentStrictConfigureParticleStuckModel
    (
        s, 4, mask, 20, heat, .63, 1, 1, 1,
#if !PR14_COLD_CHT
        1800, 710, 100,
#endif
        145, 1, 12, 2327, 20, 1160000, 2500, 1273, 20, .5, 0
    );
}
DeviceState* newOwner()
{
    DeviceState* s = new DeviceState;
    s->nFaces = 4;
    s->nInternalFaces = 0;
    s->particleCapacity = 4;
    nativeCheck(allocate(s->deviceState, size_t(1), "test device state"));
    nativeCheck(syncDeviceState(s, "test initial state publication"));
    return s;
}

#if PR14_COLD_CHT
void checkEffusivity(const DeviceState* s, const double* expected)
{
    double faces[4];
    cudaCheck(cudaMemcpy(faces, s->particleWallEffusivityByFace,
                         sizeof(faces), cudaMemcpyDeviceToHost));
    check(std::memcmp(faces, expected, sizeof(faces)) == 0,
          "scalar fallback and coupled overrides retain exact face values");
    DeviceState published;
    cudaCheck(cudaMemcpy(&published, s->deviceState, sizeof(published),
                         cudaMemcpyDeviceToHost));
    check(published.particleWallEffusivityByFace == s->particleWallEffusivityByFace,
          "published face owner matches canonical allocation");
}
void effusivityLifecycle(bool uploadFirst)
{
    DeviceState* s = newOwner();
    const unsigned char mixed[] = {0, 3, 4, 0}, off[] = {0, 0, 0, 0};
    const double fallback = ::sqrt(1800.0*710.0*100.0);
    double expected[] = {fallback, fallback, fallback, fallback};
    const double firstPatch[] = {24001.25, 17003.5};
    if (uploadFirst)
    {
        nativeCheck(ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
                    (s, 1, 2, firstPatch));
        expected[1] = firstPatch[0]; expected[2] = firstPatch[1];
        checkEffusivity(s, expected);
    }
    nativeCheck(configureCold(s, mixed));
    checkEffusivity(s, expected);
    if (!uploadFirst)
    {
        nativeCheck(ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
                    (s, 1, 2, firstPatch));
        expected[1] = firstPatch[0]; expected[2] = firstPatch[1];
    }
    auto* owner = s->particleWallEffusivityByFace;
    nativeCheck(configureCold(s, mixed));
    nativeCheck(configureCold(s, off, 0));
    nativeCheck(configureCold(s, mixed));
    check(s->particleWallEffusivityByFace == owner, "reconfigure retains owner");
    checkEffusivity(s, expected);
    // Supplied patch values are copied unchanged, including zero and signed
    // values. The fix supplies absent material data; it adds no value gate.
    const double replacement[] = {-0.0, -2.0};
    nativeCheck(ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
                (s, 1, 2, replacement));
    expected[1] = replacement[0]; expected[2] = replacement[1];
    nativeCheck(configureCold(s, mixed));
    checkEffusivity(s, expected);
    check((s->particleStuckInitialisationPending & (4 | 0x800)) == 0,
          "initialization/publication work finishes once");
    releaseState(s);
}
#endif

struct ProfileResult
{
    double initial, stored, physical, carryBefore, carryAfter;
    double temperatureBefore, temperatureAfter;
    int uniform, historyCleared;
};
__global__ void initializeProfiles(DeviceState* state, ProfileResult* results)
{
    const int i = threadIdx.x;
    if (i >= 4) return;
    DeviceState& s = *state;
    const double temperatures[] = {3200.1234567, 1000.1234567, 2327.25, 2337.125};
    const double carries[] = {0., .07125, -.0625, 17.03125};
    const bool two = (i & 1) != 0;
    ProfileResult r{};
    r.initial = Foam::gpuThermal::coldWallSpecificEnthalpyJkg
                (temperatures[i], s.coldWallSolidificationParameters);
#if PR14_COLD_CHT
    s.pThermalEnthalpyResidual[i] = carries[i];
    r.carryBefore = carries[i];
#endif
    if (two)
    {
        s.pCold2DFrozenArea[i] = 3.;
        for (int n = 0; n < 8; ++n) s.pCold2DRingContactAge[8*i+n] = 7.;
        initialiseColdWall2DParticleState(s, i, temperatures[i]);
    }
    else
    {
        s.pColdFrozenArea[i] = 3.;
        s.pColdContactAge[i] = 7.;
        for (int n = 0; n < 8; ++n) s.pColdRingSolidMass[8*i+n] = 7.;
        initialiseColdWallParticleState(s, i, temperatures[i]);
    }
    const int count = two ? 64 : 8;
    const auto* nodes = two ? s.pCold2DNodeSpecificEnthalpy : s.pColdNodeSpecificEnthalpy;
    r.stored = double(nodes[i*count]);
    r.uniform = 1;
    // Offset averaging is exact for this uniform initialization witness and
    // does not add an unrelated repeated-summation error to the FSH oracle.
    r.physical = r.stored;
    for (int n = 0; n < count; ++n)
    {
        r.uniform &= (double(nodes[i*count+n]) == r.stored);
        r.physical += (double(nodes[i*count+n])-r.stored)/double(count);
    }
#if PR14_COLD_CHT
    r.carryAfter = s.pThermalEnthalpyResidual[i];
#endif
    r.physical += r.carryAfter;
    r.temperatureBefore = Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (r.initial + r.carryBefore, s.coldWallSolidificationParameters);
    r.temperatureAfter = Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (r.physical, s.coldWallSolidificationParameters);
    r.historyCleared = two ? s.pCold2DFrozenArea[i] == 0 :
        (s.pColdFrozenArea[i] == 0 && s.pColdContactAge[i] == 0);
    for (int n = 0; n < 8; ++n)
        r.historyCleared &= two ? s.pCold2DRingContactAge[8*i+n] == 0 :
                                  s.pColdRingSolidMass[8*i+n] == 0;
    results[i] = r;
}
void profileInitialization()
{
    DeviceState* s = newOwner();
    const unsigned char mixed[] = {0, 3, 4, 0};
    nativeCheck(configureCold(s, mixed));
#if PR14_COLD_CHT
    static_assert(std::is_same<decltype(s->pColdNodeSpecificEnthalpy), float*>::value,
                  "CHT retains its profile cache storage");
    nativeCheck(allocate(s->pThermalEnthalpyResidual, size_t(4), "test carry"));
#else
    static_assert(std::is_same<decltype(s->pColdNodeSpecificEnthalpy), double*>::value,
                  "FSH retains double profiles");
#endif
    nativeCheck(syncDeviceState(s, "test profile state publication"));
    ProfileResult *device, result[4];
    cudaCheck(cudaMalloc(&device, sizeof(result)));
    initializeProfiles<<<1, 32>>>(s->deviceState, device);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost));
    bool hasCastLoss = false;
    for (const auto& r : result)
    {
        check(r.uniform && r.historyCleared, "uniform profile and cleared history");
        check(std::isfinite(r.physical), "finite initialized physical enthalpy");
        check(std::fabs(r.temperatureAfter-r.temperatureBefore) <= 1e-9,
              "initial mean profile plus carry preserves physical temperature");
#if PR14_COLD_CHT
        const double loss = r.initial-r.stored;
        hasCastLoss |= loss != 0;
        check(r.carryAfter == r.carryBefore+loss,
              "initial cast loss is added to pre-existing double carry");
#else
        check(r.stored == r.initial && r.physical == r.initial,
              "FSH initialization remains exact double");
#endif
    }
#if PR14_COLD_CHT
    check(hasCastLoss, "fixture exercises a real float cache cast loss");
#endif
    cudaCheck(cudaFree(device));
    releaseState(s);
}
int main()
{
#if PR14_COLD_CHT
    effusivityLifecycle(false);
    effusivityLifecycle(true);
#endif
    profileInitialization();
    std::puts("PASS actual CUDA cold-wall initialization and owner lifetime");
}
