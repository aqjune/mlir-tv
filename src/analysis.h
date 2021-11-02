#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include <algorithm>

struct AnalysisResult {
    int argFpCount;
    int varFpCount;
    int constFpCount;
};

AnalysisResult analysis(mlir::FuncOp &fn, bool isFullyAbstract);
