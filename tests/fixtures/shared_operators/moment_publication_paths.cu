// Native path fixture: compile with the application's real include directory.
// This file never calls publishParticleMomentsCell directly.
#include "GpuResidentStrict.cu"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

#ifndef MH04_THERMAL
#error "Define MH04_THERMAL=0 for gasUGKP or 1 for FSH/CHT"
#endif
#ifndef MH04_CHT
#error "Define MH04_CHT=1 for CHT or 0 otherwise"
#endif
#if MH04_CHT && UGKWP_GPU_REAL_BITS == 32
using namespace ugkwpCudaFp32;
#endif

namespace mh04_fixture
{
using Real = typename std::remove_pointer<decltype(DeviceState::momRhoP)>::type;
constexpr int cellCount = 5;
constexpr int particleCount = 419;
constexpr int maxTasks = 16;
constexpr int countPoison = -777;
constexpr int countGuard = 654321;
constexpr Real valuePoison = Real(-123.0);
constexpr Real valueGuard = Real(987654.0);
constexpr long double tolerance = sizeof(Real) == 4 ? 6e-6L : 3e-12L;
const char* activePath = "initialization";
int activeBlock = 0;
int activeVolumeCase = 0;

void require(const bool condition, const char* what)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL path=%s block=%d volume=%d: %s\n",
            activePath, activeBlock, activeVolumeCase, what);
        std::exit(2);
    }
}

void cudaOk(const cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA path=%s block=%d: %s: %s\n",
            activePath, activeBlock, what, cudaGetErrorString(result));
        std::exit(3);
    }
}

void finishLaunch()
{
    cudaOk(cudaGetLastError(), "kernel launch");
    cudaOk(cudaDeviceSynchronize(), "kernel completion");
}

struct ManagedStorage
{
    std::vector<void*> pointers;
    template<class T> void allocate(T*& pointer, const int count)
    {
        cudaOk(cudaMallocManaged(&pointer, sizeof(T)*count), "fixture allocation");
        pointers.push_back(pointer);
        for (int i = 0; i < count; ++i) ::new(static_cast<void*>(pointer + i)) T{};
    }
    ~ManagedStorage()
    {
        for (void* pointer : pointers) cudaFree(pointer);
    }
};

struct Particle
{
    int cell, status;
    bool stuck;
    Real mass, vx, vy, vz, theta, diameter, temperature;
};

struct Ledger
{
    std::array<long double, 7> extensive{};
    int count = 0;
};

using CellLedger = std::array<Ledger, cellCount>;
using MomentPointers = std::array<Real*, 7>;

MomentPointers outputFields(DeviceState& state)
{
    return {{state.momRhoP, state.momRhoUPx, state.momRhoUPy,
             state.momRhoUPz, state.momRhoEP, state.momRhoPD, state.momRhoHpP}};
}

void near(const long double got, const long double expected,
          const int cell, const char* field)
{
    const long double limit = tolerance*std::max(1.0L, std::fabs(expected));
    if (!std::isfinite(got) || std::fabs(got - expected) > limit)
    {
        std::fprintf(stderr,
            "FAIL path=%s block=%d volume=%d cell=%d field=%s got=%.21Lg expected=%.21Lg limit=%.4Lg\n",
            activePath, activeBlock, activeVolumeCase, cell, field, got, expected, limit);
        std::exit(4);
    }
}

// Independent host particle ledger: no CSR task traversal, GPU reduction or
// publisher is reused. The established host/device material law is the only
// shared oracle dependency; this fixture does not revalidate that law itself.
CellLedger referenceLedger(const std::vector<Particle>& particles,
                           const DeviceState& state)
{
    CellLedger reference{};
    for (const Particle& particle : particles)
    {
        if (particle.status != 1 || particle.cell < 0 || particle.cell >= cellCount) continue;
        Ledger& cell = reference[particle.cell];
        const long double mass = particle.mass;
        const std::array<long double, 3> velocity{{particle.vx, particle.vy, particle.vz}};
        long double speedSquared = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
            cell.extensive[axis + 1] += mass*velocity[axis];
            speedSquared += velocity[axis]*velocity[axis];
        }
        long double theta = particle.theta;
#if MH04_THERMAL
        if (particle.stuck) theta = 0;
#endif
        cell.extensive[0] += mass;
        cell.extensive[4] += mass*speedSquared/2 + mass*theta*3/2;
        cell.extensive[5] += mass*particle.diameter;
        const Real boundedTemperature = std::max(state.TpMin,
            std::min(state.TpMax, particle.temperature));
#if MH04_THERMAL
        cell.extensive[6] += mass*Foam::gpuThermal::aluminaSpecificEnthalpyJkg(boundedTemperature);
#else
        // Fixture parameters deliberately give thermal mass ratio 1/2 and Cp 4.
        cell.extensive[6] += mass*2*boundedTemperature;
#endif
        ++cell.count;
    }
    return reference;
}

void resetOutputs(DeviceState& state)
{
    for (Real* field : outputFields(state))
    {
        std::fill(field, field + cellCount, valuePoison);
        field[cellCount] = valueGuard;
    }
    std::fill(state.cellParticleCount, state.cellParticleCount + cellCount + 1, countPoison);
    state.cellParticleCount[cellCount + 1] = countGuard;
    std::fill(state.csrHeavyPartials, state.csrHeavyPartials + 8*maxTasks, valuePoison);
    state.csrHeavyPartials[8*maxTasks] = valueGuard;
}

void checkGuards(const DeviceState& state)
{
    for (Real* field : outputFields(const_cast<DeviceState&>(state)))
        require(field[cellCount] == valueGuard, "moment array guard modified");
    require(state.cellParticleCount[cellCount + 1] == countGuard, "count array guard modified");
    require(state.csrHeavyPartials[8*maxTasks] == valueGuard, "partial array guard modified");
}

void checkCell(DeviceState& state, const CellLedger& reference, const int c,
               const bool published)
{
    static const char* names[] = {"mass", "momentum-x", "momentum-y", "momentum-z",
                                 "mechanical-energy", "mass-diameter", "material-enthalpy"};
    const MomentPointers fields = outputFields(state);
    if (!published)
    {
        require(state.cellParticleCount[c] == countPoison, "unpublished count was written");
        for (Real* field : fields) require(field[c] == valuePoison, "unpublished moment was written");
        return;
    }
    require(state.cellParticleCount[c] == reference[c].count, "survivor count mismatch");
    // Compare reconstructed extensive amounts with the independent ledger.
    // No copied invV expression or sum-array-to-field publisher is the oracle.
    const long double effectiveVolume = std::max<long double>(state.V[c], state.rhoMin);
    for (int component = 0; component < 7; ++component)
        near(static_cast<long double>(fields[component][c])*effectiveVolume,
             reference[c].extensive[component], c, names[component]);
}

void initialize(DeviceState& state, ManagedStorage& storage,
                std::vector<Particle>& particles)
{
    state.deviceState = &state;
    state.nCells = cellCount;
    state.particleCapacity = particleCount;
    state.rhoMin = Real(0.125);
    state.TpMin = Real(200);
    state.TpMax = Real(4000);
    state.particleDiameterFallback = Real(0.0625);
#if !MH04_THERMAL
    state.solveParticleTemperature = 1;
    state.particleThermalRho = 2;
    state.particleCp = 4;
    state.rhoSolid = 4;
#endif
    storage.allocate(state.V, cellCount);
    storage.allocate(state.pCellId, particleCount);
    storage.allocate(state.pStatus, particleCount);
    storage.allocate(state.pm, particleCount);
    storage.allocate(state.pux, particleCount);
    storage.allocate(state.puy, particleCount);
    storage.allocate(state.puz, particleCount);
    storage.allocate(state.pTheta, particleCount);
    storage.allocate(state.pd, particleCount);
    storage.allocate(state.pT, particleCount);
#if MH04_THERMAL
    storage.allocate(state.pStuck, particleCount);
#endif
    storage.allocate(state.cellParticleOffset, cellCount + 1);
    storage.allocate(state.cellParticleCount, cellCount + 2);
    storage.allocate(state.momRhoP, cellCount + 1);
    storage.allocate(state.momRhoUPx, cellCount + 1);
    storage.allocate(state.momRhoUPy, cellCount + 1);
    storage.allocate(state.momRhoUPz, cellCount + 1);
    storage.allocate(state.momRhoEP, cellCount + 1);
    storage.allocate(state.momRhoPD, cellCount + 1);
    storage.allocate(state.momRhoHpP, cellCount + 1);
    storage.allocate(state.csrCellTaskCount, cellCount + 1);
    storage.allocate(state.csrCellTaskOffset, cellCount + 1);
    storage.allocate(state.csrReductionTasks, maxTasks);
    storage.allocate(state.csrMultiTaskCellList, cellCount);
    storage.allocate(state.csrHeavyTaskCount, 1);
    storage.allocate(state.csrHeavyCellCount, 1);
    storage.allocate(state.csrHeavyTaskCursor, 1);
    storage.allocate(state.csrHeavyPartials, 8*maxTasks + 1);

    std::array<std::vector<int>, cellCount> bins;
    particles.reserve(particleCount);
    const Real temperatures[] = {Real(100), Real(800), Real(2400), Real(3200), Real(4500)};
    for (int i = 0; i < particleCount; ++i)
    {
        const int c = i < 271 ? 0 : i < 372 ? 1 : i < 395 ? 2 : 4;
        Particle p{};
        p.cell = i % 47 == 13 ? -1 : c;
        p.status = c == 2 ? 0 : i % 17 == 3 ? 2 : i % 11 == 7 ? 0 : 1;
        p.stuck = i % 5 == 0;
        p.mass = Real(0.5 + 0.125*(i % 7));
        p.vx = Real(0.125*((i % 11) - 5));
        p.vy = Real(0.25*((i % 5) - 2));
        p.vz = Real(0.0625*((i % 13) - 6));
        p.theta = Real(0.125 + 0.03125*(i % 4));
        p.diameter = Real(0.03125 + 0.00390625*(i % 9));
        p.temperature = temperatures[i % 5];
        particles.push_back(p);
        bins[c].push_back(i);
        state.pCellId[i] = p.cell;
        state.pStatus[i] = p.status;
        state.pm[i] = p.mass;
        state.pux[i] = p.vx;
        state.puy[i] = p.vy;
        state.puz[i] = p.vz;
        state.pTheta[i] = p.theta;
        state.pd[i] = p.diameter;
        state.pT[i] = p.temperature;
#if MH04_THERMAL
        state.pStuck[i] = Foam::gpuThermal::particleWallMobile + (p.stuck ? 1 : 0);
#endif
    }
    // A shuffled fullIndexed CSR is sufficient to exercise both real consumers.
    // Invalid/wrong-cell records are extras and never enter the host ledger.
    std::vector<int> sorted;
    for (int c = 0; c < cellCount; ++c)
    {
        state.cellParticleOffset[c] = static_cast<int>(sorted.size());
        if (c == 3) continue;  // Actual empty cell, with no synthetic L2 task.
        std::reverse(bins[c].begin(), bins[c].end());
        bins[c].insert(bins[c].begin() + bins[c].size()/2, -1);
        bins[c].push_back(particleCount + 3);
        bins[c].push_back(c == 0 ? 272 : 0);
        sorted.insert(sorted.end(), bins[c].begin(), bins[c].end());
    }
    state.cellParticleOffset[cellCount] = static_cast<int>(sorted.size());
    storage.allocate(state.sortedParticleIndex, static_cast<int>(sorted.size()));
    std::copy(sorted.begin(), sorted.end(), state.sortedParticleIndex);
}

void prepareDirectory(DeviceState& state, const std::array<int, cellCount>& taskCounts)
{
    int task = 0;
    int multi = 0;
    for (int c = 0; c < cellCount; ++c)
    {
        state.csrCellTaskCount[c] = taskCounts[c];
        state.csrCellTaskOffset[c] = task;
        const int begin = state.cellParticleOffset[c];
        const int size = state.cellParticleOffset[c + 1] - begin;
        require(taskCounts[c] == 0 || size >= taskCounts[c], "fixture task range is empty");
        for (int local = 0; local < taskCounts[c]; ++local)
        {
            require(task < maxTasks, "fixture task capacity exceeded");
            state.csrReductionTasks[task++] = CsrReductionTask{
                c, begin + size*local/taskCounts[c], begin + size*(local + 1)/taskCounts[c],
                static_cast<int>(CsrReductionTaskSource::fullIndexed)};
        }
        if (taskCounts[c] > 1) state.csrMultiTaskCellList[multi++] = c;
    }
    state.csrCellTaskOffset[cellCount] = task;
    std::reverse(state.csrMultiTaskCellList, state.csrMultiTaskCellList + multi);
    *state.csrHeavyTaskCount = task;
    *state.csrHeavyCellCount = multi;
}

void runS1(DeviceState& state, const CellLedger& reference, const int block)
{
    activePath = "S1-native";
    resetOutputs(state);
    state.csrHeavyReductionEnabled = 0;
    accumulateParticleMomentsSegmentedKernel<false, false>
        <<<cellCount, block, 8*((block + 31)/32)*sizeof(Real)>>>(&state);
    finishLaunch();
    for (int c = 0; c < cellCount; ++c) checkCell(state, reference, c, true);
    require(state.cellParticleCount[cellCount] == 0, "S1 count sentinel not zero");
    checkGuards(state);
}

void runL2(DeviceState& state, const CellLedger& reference, const int block,
           const std::array<int, cellCount>& taskCounts, const char* label)
{
    activePath = label;
    resetOutputs(state);
    prepareDirectory(state, taskCounts);
    state.csrHeavyReductionEnabled = 1;
    *state.csrHeavyTaskCursor = 0;
    const size_t sharedBytes = 8*((block + 31)/32)*sizeof(Real);
    accumulateCsrSegmentedMomentTasksPersistentKernel<false><<<3, block, sharedBytes>>>(&state);
    finishLaunch();
    for (int c = 0; c < cellCount; ++c)
        checkCell(state, reference, c, taskCounts[c] == 1);
    require(state.cellParticleCount[cellCount] == (taskCounts[0] == 1 ? 0 : countPoison),
        "worker count sentinel ownership mismatch");
    checkGuards(state);
    if (*state.csrHeavyCellCount > 0)
    {
        // A separate real kernel consumes actual worker partials; do not seed
        // synthetic sums or bypass the reduction/finalizer being refactored.
        *state.csrHeavyTaskCursor = 0;
        finalizeCsrSegmentedMomentCellsKernel<<<2, block, sharedBytes>>>(&state);
        finishLaunch();
    }
    for (int c = 0; c < cellCount; ++c)
        checkCell(state, reference, c, taskCounts[c] != 0);
    require(state.cellParticleCount[cellCount] == 0, "completed L2 count sentinel not zero");
    checkGuards(state);
}
} // namespace mh04_fixture

int main()
{
    using namespace mh04_fixture;
    ManagedStorage storage;
    DeviceState* state = nullptr;
    storage.allocate(state, 1);
    std::vector<Particle> particles;
    initialize(*state, storage, particles);
    const CellLedger reference = referenceLedger(particles, *state);
    require(reference[0].count > 32 && reference[1].count > 0 && reference[4].count > 0,
            "fixture lacks active particles");
    require(reference[2].count == 0 && reference[3].count == 0,
            "fixture zero-survivor cells are malformed");
    int completedCases = 0;
    for (int volumeCase = 0; volumeCase < 2; ++volumeCase)
    {
        activeVolumeCase = volumeCase;
        const Real volumes[] = {Real(2.5), Real(0.75), Real(3.25), Real(0.5),
                               volumeCase == 0 ? Real(1.5) : Real(0.0625)};
        std::copy(volumes, volumes + cellCount, state->V);
        for (const int block : {32, 64, 128, 256})
        {
            activeBlock = block;
            runS1(*state, reference, block);
            runL2(*state, reference, block, {{1, 1, 1, 0, 1}}, "L2-all-single");
            runL2(*state, reference, block, {{3, 2, 2, 0, 2}}, "L2-all-multi");
            runL2(*state, reference, block, {{1, 3, 1, 0, 2}}, "L2-mixed-cell0-single");
            runL2(*state, reference, block, {{3, 1, 2, 0, 1}}, "L2-mixed-cell0-multi");
            completedCases += 5;
        }
    }
    std::printf("PASS native moment paths: %d cases, real-bits=%zu, thermal=%d; "
        "S1/L2-single/L2-finalize, seven physical moments, counts, survivor filters, "
        "non-unit/clamped V, empty/rejected cells, sentinel ownership, array guards\n",
        completedCases, sizeof(Real)*8, MH04_THERMAL);
    return 0;
}
