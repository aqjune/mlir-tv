#pragma once

#include "smt.h"
#include "state.h"
#include "vcgen.h"

#include <vector>
#include "mlir/IR/BuiltinOps.h"
#include "z3++.h"

void printOperations(smt::model m, mlir::FuncOp fn, const State &st);

void printCounterEx(
    smt::model model, const std::vector<smt::expr> &params,
    mlir::FuncOp src, mlir::FuncOp tgt,
    const State &st_src, const State &st_tgt,
    VerificationStep step);