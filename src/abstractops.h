#pragma once

#include "smt.h"

namespace aop {

struct UsedAbstractOps {
  bool dot;
  bool add;
  bool mul;
  bool sum;
};
UsedAbstractOps getUsedAbstractOps();

enum AbsLevelDot {
  FULLY_ABS = 0, // Dot is a fully unknown function
  SUM_MUL   = 1  // Dot is a summation of pairwisely multiplied values
};

// This resets the used abstract ops record.
void setAbstractionLevel(AbsLevelDot);

smt::expr mkZeroElemFromArr(const smt::expr &arr);
smt::expr fp_add(const smt::expr &f1, const smt::expr &f2);
smt::expr fp_mul(const smt::expr &f1, const smt::expr &f2);
smt::expr sum(const smt::expr &arr, const smt::expr &n);
smt::expr dot(const smt::expr &arr1, const smt::expr &arr2, const smt::expr &n);

};