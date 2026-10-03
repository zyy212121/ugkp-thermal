#pragma once
__host__ __device__ constexpr int oppositeCellAcrossFace
(const int cell, const int owner, const int neighbour)
{
    return cell == owner ? neighbour : owner;
}
