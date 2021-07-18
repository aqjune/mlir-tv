#pragma once

#include "z3++.h"

namespace aop {

enum AbsLevelDot {
  FULLY_ABS = 0, // Dot is a fully unknown function
  SUM_MUL   = 1  // Dot is a summation of pairwisely multiplied result
};

void setAbstractionLevel(AbsLevelDot);

z3::expr mkZeroElemFromArr(const z3::expr &arr);
z3::expr fp_add(const z3::expr &f1, const z3::expr &f2);
z3::expr fp_mul(const z3::expr &f1, const z3::expr &f2);
z3::expr sum(const z3::expr &arr, const z3::expr &n);
z3::expr dot(const z3::expr &arr1, const z3::expr &arr2, const z3::expr &n);

};