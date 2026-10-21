#pragma once
struct SequenceRecord {
 int step=0,stage=0,owner=0;double time=0;
 ugkwp::gaswall::WallInput<double,2> input;
 double lo[2]{},hi[2]{},target[2][9]{},normalLo[2]{},normalHi[2]{};
 bool algebraic[2]{};
 double U[2][3][3]{},T[2][3]{},k[2][3]{},omega[2][3]{},Y[2][2][3]{};
};
