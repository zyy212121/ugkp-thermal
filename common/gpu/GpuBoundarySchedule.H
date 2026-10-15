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

public:
    explicit GpuBoundaryScheduleTable
    (
        const List<Tuple2<scalar, scalar>>& rows
    )
    :
        rows_(rows)
    {
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
