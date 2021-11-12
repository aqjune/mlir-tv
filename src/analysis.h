#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <set>

struct SolePrecisionAnalysisResult {
  std::set<llvm::APFloat> fpConstSet;
  size_t fpVarCount;
  size_t fpArgCount;
};

struct AnalysisResult {
    SolePrecisionAnalysisResult F32;
    SolePrecisionAnalysisResult F64;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
