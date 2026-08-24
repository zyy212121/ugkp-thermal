#ifndef GpuBoundarySchedule_H
#define GpuBoundarySchedule_H

#include "IFstream.H"
#include "List.H"
#include "Tuple2.H"
#include "error.H"
#include "fileName.H"
#include "scalar.H"
#include "scalarField.H"

#include <cmath>

namespace Foam
{

class GpuBoundaryScheduleTable
{
    List<Tuple2<scalar, scalar>> rows_;

    void requireValid(const char* context) const
    {
        if (!validRows(rows_))
        {
            FatalErrorInFunction
                << context << ": schedule requires at least one finite row, "
                << "strictly increasing non-negative times, and finite values"
                << exit(FatalError);
        }
    }

public:
    explicit GpuBoundaryScheduleTable
    (
        const List<Tuple2<scalar, scalar>>& rows
    )
    :
        rows_(rows)
    {
        requireValid("in-memory GPU boundary schedule");
    }

    explicit GpuBoundaryScheduleTable(const fileName& path)
    :
        rows_()
    {
        IFstream input(path);
        if (!input.good())
        {
            FatalErrorInFunction
                << "cannot read GPU boundary schedule " << path
                << exit(FatalError);
        }
        input >> rows_;
        requireValid(path.c_str());
    }

    static bool validRows(const List<Tuple2<scalar, scalar>>& rows)
    {
        if (rows.empty())
        {
            return false;
        }

        scalar previousTime = -1;
        forAll(rows, i)
        {
            const scalar timeValue = rows[i].first();
            const scalar value = rows[i].second();
            if
            (
                !std::isfinite(timeValue)
             || !std::isfinite(value)
             || timeValue < 0
             || (i > 0 && timeValue <= previousTime)
            )
            {
                return false;
            }
            previousTime = timeValue;
        }
        return true;
    }

    label size() const
    {
        return rows_.size();
    }

    void copyColumns(scalarField& times, scalarField& values) const
    {
        times.setSize(rows_.size());
        values.setSize(rows_.size());
        forAll(rows_, i)
        {
            times[i] = rows_[i].first();
            values[i] = rows_[i].second();
        }
    }
};

}                      

#endif
