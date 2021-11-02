#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include <algorithm>

struct AnalysisResult {
    int varFpCount;
    int constFpCount;
};

// encode can throw UnsupportedException.
// {constant count, variable count}
AnalysisResult analysis(mlir::FuncOp &fn, bool isFullyAbstract);
