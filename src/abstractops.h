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
  SUM_MUL   = 1,  // Dot is a summation of pairwisely multiplied values
  ASSOCIATIVE_SUM_MUL = 2 // Dot is summation of pairwisely multiplied values and summation is associative.
};

// This resets the used abstract ops record.
void setAbstractionLevel(AbsLevelDot);

smt::Sort fpSort();
smt::Expr fpConst(double f);
// Return the set of possible FP constants for 'e'.
std::vector<double> fpPossibleConsts(const smt::Expr &e);

smt::Expr mkZeroElemFromArr(const smt::Expr &arr);
smt::Expr fpAdd(const smt::Expr &f1, const smt::Expr &f2);
smt::Expr fpMul(const smt::Expr &f1, const smt::Expr &f2);
smt::Expr sum(const smt::Expr &arr, const smt::Expr &n);
smt::Expr dot(const smt::Expr &arr1, const smt::Expr &arr2, const smt::Expr &n);

};
