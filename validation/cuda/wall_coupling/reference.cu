#include <cuda_runtime.h>
#include <cstring>
#include "common/gasWall/WallReferenceSolver.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
using namespace ugkwp::gaswall;
using In = WallInput<double, 2>;
using Work = WallWorkspace<double, 2, 128>;
using Out = WallOutput<double, 2>;
struct History
{
    bool available = false;
    int nodes = 0;
    double state[128][6]{};
};
void ck(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(e) << '\n';
        exit(2);
    }
}
template <class T> T *managed(int n = 1)
{
    T *p;
    ck(cudaMallocManaged(&p, sizeof(T) * n));
    for (int i = 0; i < n; ++i)
        new (p + i) T{};
    return p;
}
__global__ __launch_bounds__(32) void solve(In *in, WallModelConfig<double> *c, Work *w,
                                            WallReferenceScratch<2, 128> *scratch, History *h,
                                            Out *out, WallStatus *st, WallReferenceStats *stats,
                                            int *ok)
{
    detail::CudaWallTeam team;
    bool result = evaluateWallReference(
        *in, *c, *w, *scratch, h->available ? &h->state[0][0] : nullptr, *out, *st, *stats, team);
    if (team.leader())
        *ok = result;
}
int main(int argc, char **argv)
{
    std::cout
        << std::setprecision(17)
        << "case,nodes,history,cpu_iterations,cpu_residual,gpu_ok,gpu_iterations,gpu_residual,cpu_"
           "residual_calls,gpu_residual_calls,gpu_central,state_relative,gpu_ms\n";
    auto in = managed<In>();
    auto c = managed<WallModelConfig<double>>();
    auto w = managed<Work>();
    auto scratch = managed<WallReferenceScratch<2, 128>>();
    auto h = managed<History>();
    auto out = managed<Out>();
    auto st = managed<WallStatus>();
    auto stats = managed<WallReferenceStats>();
    auto ok = managed<int>();
    auto species = managed<ugkwp::SpeciesThermoData<double>>(2);
    auto coef = managed<double>(6);
    CapturedSstWall props;
    memcpy(species, props.species, sizeof(props.species));
    memcpy(coef, props.coefficients, sizeof(props.coefficients));
    int failed = 0;
    for (int a = -6; a < argc - 1; ++a)
    {
        *h = {};
        std::string name;
        if (a < 0)
        {
            int j = a + 6;
            CapturedSingularWall small;
            *in = j >= 3 ? small.input : props.input;
            *c = props.config;
            c->nodes = (int[]){24, 48, 128}[j % 3];
            c->maxIterations = 100;
            name = j >= 3 ? "cold-small-k" : "cold-normal";
        }
        else
        {
            std::ifstream f(argv[a + 1], std::ios::binary);
            Work saved;
            f.read((char *)in, sizeof(*in));
            f.read((char *)c, sizeof(*c));
            f.read((char *)&saved, sizeof(saved));
            f.read((char *)h, sizeof(*h));
            if (!f)
                return 3;
            name = argv[a + 1];
        }
        in->model = props.input.model;
        in->diagnostics = nullptr;
        in->model.thermo.diagnostics = nullptr;
        if (in->quadrature.count)
            return 4;
        Work cpu{};
        Out cpuout;
        WallStatus cpust;
        WallReferenceStats cs;
        WallReferenceScratch<2, 128> ss;
        bool cpuok = evaluateWallReference(
            *in, *c, cpu, ss, h->available ? &h->state[0][0] : nullptr, cpuout, cpust, cs);
        in->model.thermo.species = species;
        in->model.thermo.coefficients = coef;
        solve<<<1, 32>>>(in, c, w, scratch, h, out, st, stats, ok);
        ck(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        ck(cudaEventCreate(&start));
        ck(cudaEventCreate(&end));
        ck(cudaEventRecord(start));
        for (int rep = 0; rep < 3; ++rep)
            solve<<<1, 32>>>(in, c, w, scratch, h, out, st, stats, ok);
        ck(cudaEventRecord(end));
        ck(cudaEventSynchronize(end));
        float ms;
        ck(cudaEventElapsedTime(&ms, start, end));
        double err = 0;
        for (int i = 0; i < c->nodes; ++i)
            for (int j = 0; j < 6; ++j)
                err = std::max(err, std::abs(cpu.state[i][j] - w->state[i][j]) /
                                        std::max(1e-8, std::abs(cpu.state[i][j])));
        std::cout << name << ',' << c->nodes << ',' << h->available << ',' << cs.iterations << ','
                  << cs.residual << ',' << *ok << ',' << stats->iterations << ',' << stats->residual
                  << ',' << cs.residualEvaluations << ',' << stats->residualEvaluations << ','
                  << stats->centralDirections << ',' << err << ',' << ms / 3 << std::endl;
        failed += !cpuok || !*ok || !cs.converged || !stats->converged || err > 1e-5;
        ck(cudaEventDestroy(start));
        ck(cudaEventDestroy(end));
    }
    for (void *p : {(void *)in, (void *)c, (void *)w, (void *)scratch, (void *)h, (void *)out,
                    (void *)st, (void *)stats, (void *)ok, (void *)species, (void *)coef})
        ck(cudaFree(p));
    return failed ? 5 : 0;
}
