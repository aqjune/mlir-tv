#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

struct AnalysisResult {
    int constF32Count;
    int varF32Count;
    int argF32Count;
    
    int constF64Count;
    int varF64Count;
    int argF64Count;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
