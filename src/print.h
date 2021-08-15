#pragma once

#include "smt.h"
#include "state.h"
#include "vcgen.h"

#include <vector>
#include "mlir/IR/BuiltinOps.h"
#include "z3++.h"

void printCounterEx(
    z3::model model, const std::vector<smt::expr> &params,
    mlir::FuncOp src, mlir::FuncOp tgt,
    const State &st_src, const State &st_tgt,
    VerificationStep step);