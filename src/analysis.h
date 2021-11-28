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

struct ShapedTypeAnalysisResult {
  TypeMap<size_t> memrefArgCount;
  TypeMap<size_t> memrefVarCount;
  TypeMap<size_t> tensorArgCount;
  TypeMap<size_t> tensorVarCount;
};

struct AnalysisResult {
  FPAnalysisResult F32;
  FPAnalysisResult F64;
  ShapedTypeAnalysisResult shapedValue;
};

AnalysisResult analyze(mlir::FuncOp &fn);
