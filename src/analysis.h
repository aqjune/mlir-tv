#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

struct AnalysisResult {
    int argFpCount;
    int varFpCount;
    int constFpCount;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
