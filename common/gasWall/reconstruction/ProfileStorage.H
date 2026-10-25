#pragma once
#include "../WallModel.H"
namespace ugkwp
{
namespace gaswall
{
namespace reconstruction
{
template <class T, int N> struct Array
{
    T data[N];
    UGKWP_GAS_HD T &operator[](int i) { return data[i]; }
    UGKWP_GAS_HD const T &operator[](int i) const { return data[i]; }
    UGKWP_GAS_HD T *begin() { return data; }
    UGKWP_GAS_HD T *end() { return data + N; }
    UGKWP_GAS_HD const T *begin() const { return data; }
    UGKWP_GAS_HD const T *end() const { return data + N; }
};
template <class T, int N> struct FixedVector
{
    T data[N]{};
    int count = 0;
    UGKWP_GAS_HD int size() const { return count; }
    UGKWP_GAS_HD T &operator[](int i) { return data[i]; }
    UGKWP_GAS_HD const T &operator[](int i) const { return data[i]; }
    UGKWP_GAS_HD T &front() { return data[0]; }
    UGKWP_GAS_HD const T &front() const { return data[0]; }
    UGKWP_GAS_HD T &back() { return data[count - 1]; }
    UGKWP_GAS_HD const T &back() const { return data[count - 1]; }
    UGKWP_GAS_HD T *begin() { return data; }
    UGKWP_GAS_HD T *end() { return data + count; }
    UGKWP_GAS_HD const T *begin() const { return data; }
    UGKWP_GAS_HD const T *end() const { return data + count; }
    UGKWP_GAS_HD void push_back(const T &v) { data[count++] = v; }
};
template <class T> struct BorrowedVector
{
    T *data;
    int count;
    UGKWP_GAS_HD T *begin() { return data; }
    UGKWP_GAS_HD T *end() { return data + count; }
    UGKWP_GAS_HD void push_back(const T &v) { data[count++] = v; }
};
} // namespace reconstruction
} // namespace gaswall
} // namespace ugkwp
