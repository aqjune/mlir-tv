#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <optional>
#include <set>

struct FPConstAnalysisResult {
  llvm::APFloat value;
  std::optional<bool> zero_limit_bits;
  std::optional<bool> zero_prec_bits;
};
bool operator<(const FPConstAnalysisResult&, const FPConstAnalysisResult&);

struct SolePrecisionAnalysisResult {
  std::set<FPConstAnalysisResult> fpConstSet;
  size_t fpVarCount = 0;
  size_t fpArgCount = 0;
};

struct AnalysisResult {
    SolePrecisionAnalysisResult F32;
    SolePrecisionAnalysisResult F64;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
