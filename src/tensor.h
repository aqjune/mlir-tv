#pragma once

#include "z3++.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <string>
#include <vector>

class Tensor {
  std::vector<z3::expr> dims;
  z3::expr arr;

public:
  static const unsigned BITS_FLOAT = 4;
  static const unsigned BITS_INDEX = 32;

  Tensor();

  z3::expr get(const std::vector<z3::expr> &indices) const;

  // Return a new tensor T2 s.t.
  //   T2[newidxvars] = this[srcidxs]
  // For example, if newidxvars = [x, y, z] and srcidxs = [x, y + z],
  //   T2[x][y][z] = this[x][y + z]
  Tensor affine(
      const std::vector<z3::expr> &newidxvars,
      std::vector<z3::expr> srcidxs,
      const std::vector<z3::expr> &newsizes) const;

  // Return a new tensor T2 s.t.
  //   T2[i1][i2]..[iN] = this[i2]..[iN][i1]
  Tensor rotateDimensions() const;

  // Return a new tensor which is convolution of this tensor and filter.
  Tensor conv(const Tensor &filter) const;

  static std::vector<z3::expr> getDims(mlir::TensorType tensorTy);
  static Tensor newVar(mlir::TensorType tensorTy, const std::string &name);
  static z3::expr newIdxConst(uint64_t idx);
  static z3::expr newIdxVar(const std::string &name);
  static std::vector<z3::expr> newIdxVars(
      const std::vector<std::string> &names);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, Tensor &);

private:
  z3::expr to1DArrayWithOfs(
      const std::vector<z3::expr> &offbegins,
      const std::vector<z3::expr> &sizes) const;
};