#pragma once

#include "z3++.h"

namespace aop {

z3::expr mkZeroElemFromArr(const z3::expr &arr);
z3::expr mul(const z3::expr &a, const z3::expr &b);
z3::expr dot(const z3::expr &a, const z3::expr &b, const z3::expr &n);
z3::expr dot2(const z3::expr &a, const z3::expr &b, const z3::expr &n);

};