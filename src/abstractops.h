#pragma once

#include "z3++.h"

namespace aop {

z3::expr mkZeroElemFromArr(const z3::expr &arr);
z3::expr fp_add(const z3::expr &f1, const z3::expr &f2);
z3::expr fp_mul(const z3::expr &f1, const z3::expr &f2);
z3::expr dot(const z3::expr &arr1, const z3::expr &arr2, const z3::expr &n);
z3::expr dot2(const z3::expr &arr1, const z3::expr &arr2, const z3::expr &n);

};