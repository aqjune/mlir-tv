#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

struct AnalysisResult {
    int argFpCount;
    int varFpCount;
    int constF32Count;
    int constF64Count;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
