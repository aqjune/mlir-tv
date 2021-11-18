#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <optional>
#include <set>

struct SolePrecisionAnalysisResult {
  std::set<llvm::APFloat> fpConstSet;
  size_t fpVarCount = 0;
  size_t fpArgCount = 0;
};

struct AnalysisResult {
    SolePrecisionAnalysisResult F32;
    SolePrecisionAnalysisResult F64;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
