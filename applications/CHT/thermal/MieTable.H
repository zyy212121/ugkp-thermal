#ifndef GPU_THERMAL_MIE_TABLE_H
#define GPU_THERMAL_MIE_TABLE_H

#include "fileName.H"
#include "scalar.H"
#include "scalarField.H"

#include <memory>
#include <stdexcept>

namespace Foam
{
namespace gpuThermal
{

struct MieOpticalSample
{
    scalar Qabs;
    scalar Qsca;
    scalar Qext;
    scalar planckBandFraction;
};

class MieTableError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class MieTableData;

class MieTable
{
    std::shared_ptr<const MieTableData> data_;

public:
    explicit MieTable(const fileName& tableFile);

    MieOpticalSample query(scalar diameterM, scalar temperatureK) const;
    scalar phase(scalar diameterM, scalar temperatureK, scalar mu) const;
    scalar planckBandFraction(scalar temperatureK) const;
    bool sharesDataWith(const MieTable& other) const noexcept;

    scalar minDiameterM() const;
    scalar maxDiameterM() const;
    scalar minTemperatureK() const;
    scalar maxTemperatureK() const;
    const scalarField& muValues() const;
};

}                        
}                  

#endif
