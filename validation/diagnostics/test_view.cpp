#define UGKWP_GAS_HD
#include "gasTransport/GasStateView.H"
#include <cassert>
int main() {
    ugkp::diagnostics::DeviceContext diagnostics{};
    ugkwp::GasStateView<double,double,int> state{};
    state.diagnostics=&diagnostics;
    const auto view=ugkwp::makeGasStateView(state);
    assert(view.diagnostics==&diagnostics);
}
