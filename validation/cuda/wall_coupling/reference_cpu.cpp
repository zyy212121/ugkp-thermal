#include <cstring>
#include "common/gasWall/WallReferenceSolver.H"
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include "HistoryReference.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace ugkwp::gaswall;
int main(int argc, char **argv)
{
    std::cout << std::setprecision(17)
              << "file,nodes,cpu_ok,cpu_iterations,cpu_residual,ported_ok,ported_iterations,ported_"
                 "residual,residual_calls,jacobian_calls,central_calls,state_relative\n";
    for (int arg = 1; arg < argc; ++arg)
    {
        CapturedSstWall props;
        WallInput<double, 2> in;
        WallModelConfig<double> c;
        WallWorkspace<double, 2, 128> saved{}, a{}, b{};
        augmented_wall::AcceptedHistory<2, 128> h;
        std::ifstream f(argv[arg], std::ios::binary);
        f.read((char *)&in, sizeof(in));
        f.read((char *)&c, sizeof(c));
        f.read((char *)&saved, sizeof(saved));
        f.read((char *)&h, sizeof(h));
        if (!f)
            return 2;
        in.model = props.input.model;
        in.diagnostics = nullptr;
        in.model.thermo.diagnostics = nullptr;
        if (in.quadrature.count)
            return 3;
        WallOutput<double, 2> oa, ob;
        augmented_wall::Stats sa;
        WallReferenceStats sb;
        WallReferenceScratch<2, 128> scratch;
        WallStatus st;
        bool ka = solveHistoryReference(in, c, a, h, oa, sa);
        bool kb = evaluateWallReference(in, c, b, scratch, h.available ? &h.state[0][0] : nullptr,
                                        ob, st, sb);
        double error = 0;
        for (int i = 0; i < c.nodes; ++i)
            for (int j = 0; j < 6; ++j)
                error = std::max(error, std::abs(a.state[i][j] - b.state[i][j]) /
                                            std::max(1e-12, std::abs(a.state[i][j])));
        std::cout << argv[arg] << ',' << c.nodes << ',' << ka << ',' << sa.iterations << ','
                  << sa.physicalResidual << ',' << kb << ',' << sb.iterations << ',' << sb.residual
                  << ',' << sb.residualEvaluations << ',' << sb.jacobianEvaluations << ','
                  << sb.centralDirections << ',' << error << '\n';
        if (!ka || !kb || !sb.converged || error > 1e-6)
            return 4;
    }
}
