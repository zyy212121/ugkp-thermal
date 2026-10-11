#ifndef UGKWP_GAS_BUILD_CONFIG_H
#define UGKWP_GAS_BUILD_CONFIG_H
// Storage shape only. No physical composition or mechanism is selected here.
// Build with -DUGKWP_GAS_SPECIES=10 for the explicit H2O2 verification model.
#ifndef UGKWP_GAS_SPECIES
#define UGKWP_GAS_SPECIES 2
#endif
namespace ugkwp
{
static constexpr int compiledGasSpecies = UGKWP_GAS_SPECIES;
static_assert(compiledGasSpecies > 0, "UGKWP_GAS_SPECIES must be positive");
}
#endif
