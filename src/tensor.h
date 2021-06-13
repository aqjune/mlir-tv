#pragma once

#include "z3++.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <string>
#include <vector>

class Tensor {
public:
  static const unsigned BITS_FLOAT = 4;
  static const unsigned BITS_INDEX = 32;

  std::vector<z3::expr> dims;
  z3::expr arr;

  Tensor();

  static Tensor newVar(mlir::TensorType tensorTy, const std::string &name);
  static z3::expr newIdxConst(uint64_t idx);
  static z3::expr newIdxVar(const std::string &name);
  static std::vector<z3::expr> newIdxVars(
      const std::vector<std::string> &names);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, Tensor &);
};