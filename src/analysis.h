#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "utils.h"
#include <map>
#include <optional>
#include <set>

struct FPAnalysisResult {
  std::set<llvm::APFloat> constSet;
  size_t argCount = 0;
  size_t varCount = 0;
};

struct MemRefAnalysisResult {
  TypeMap<size_t> argCount;
  TypeMap<size_t> varCount;
};

struct AnalysisResult {
  FPAnalysisResult F32;
  FPAnalysisResult F64;
  MemRefAnalysisResult memref;
};

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract);
