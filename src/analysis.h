#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "utils.h"
#include <map>
#include <optional>
#include <set>
#include <string>

struct FPAnalysisResult {
  std::set<llvm::APFloat> constSet;
  bool hasInfOrNaN = false;
  size_t argCount = 0;
  size_t varCount = 0;
  size_t elemsCount = 0;
};

struct MemRefAnalysisResult {
  TypeMap<size_t> argCount;
  TypeMap<size_t> varCount;
  std::map<std::string, mlir::memref::GlobalOp> usedGlobals;
};

struct AnalysisResult {
  FPAnalysisResult F32;
  FPAnalysisResult F64;
  MemRefAnalysisResult memref;
  bool isElementwiseFPOps = true;
};

AnalysisResult analyze(mlir::FuncOp &fn);
