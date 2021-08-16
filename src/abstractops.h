#pragma once

#include "smt.h"
#include <vector>

namespace aop {

struct UsedAbstractOps {
  bool dot;
  bool add;
  bool mul;
  bool sum;
};
UsedAbstractOps getUsedAbstractOps();

enum class AbsLevelDot {
  FULLY_ABS = 0, // Dot is a fully unknown function
  SUM_MUL   = 1  // Dot is a summation of pairwisely multiplied values
};

// This resets the used abstract ops record.
void setAbstractionLevel(AbsLevelDot);

smt::sort fpSort();
smt::expr fpConst(double f);
void fpEvalConstVars(smt::model mdl);
// Return the set of possible FP constants for 'e'.
std::vector<double> fpPossibleConsts(const smt::expr &e);

smt::expr mkZeroElemFromArr(const smt::expr &arr);
smt::expr fpAdd(const smt::expr &f1, const smt::expr &f2);
smt::expr fpMul(const smt::expr &f1, const smt::expr &f2);
smt::expr sum(const smt::expr &arr, const smt::expr &n);
smt::expr dot(const smt::expr &arr1, const smt::expr &arr2, const smt::expr &n);

};