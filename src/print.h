#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "smt.h"
#include "state.h"
#include "vcgen.h"

#include <vector>

void printOperations(smt::Model m, mlir::func::FuncOp fn, const State &st);

void printCounterEx(
    smt::Model model, const std::vector<smt::Expr> &params,
    mlir::func::FuncOp src, mlir::func::FuncOp tgt,
    const State &st_src, const State &st_tgt,
    VerificationStep step, unsigned retvalidx = -1,
    std::optional<mlir::Type> memElemTy = std::nullopt);