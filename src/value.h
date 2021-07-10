#pragma once

#include "z3++.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <string>
#include <vector>

class Index {
  z3::expr e;

public:
  static const unsigned BITS = 32;

  Index();
  Index(unsigned);
  Index(const std::string &name);
  Index(const z3::expr &e);

  operator z3::expr() const { return e; }

  static z3::sort sort();
  static Index one();
  static Index zero();

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Index &);
  Index eval(z3::model m) const;
};

class Float {
  z3::expr e;

public:
  static const unsigned BITS = 4;

  Float(const std::string &name);
  Float(const z3::expr &e): e(e) {}
  Float(const llvm::APFloat &apf);
  Float(double f);

  operator z3::expr() const { return e; }

  static z3::sort sort();

  Float add(const Float &b) const;
  Float mul(const Float &b) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Float &);
  Float eval(z3::model m) const;
};

class Integer {
  z3::expr e;

public:
  Integer(const std::string &name, unsigned bw);
  Integer(const z3::expr &e): e(e) {}

  operator z3::expr() const { return e; }

  static z3::sort sort(unsigned bw);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Integer &);
  Integer eval(z3::model m) const;
};

class Tensor {
  std::vector<z3::expr> dims;
  z3::expr arr;

public:
  Tensor();
  // A splat tensor.
  Tensor(const z3::expr &splat_elem, const std::vector<z3::expr> &dims);
  Tensor(const std::vector<z3::expr> &elems1d);
  Tensor(const std::string &name, const std::vector<z3::expr> &dims,
         const z3::sort &elemty);

  z3::expr asArray() const { return arr; }

  // Return the element at indices.
  //   z3::expr v = tensor.get(indices)
  //   useAsInt(Integer(v)) // valid only if tensor had integer elems
  //   useAsFloat(Float(v)) // valid only if tensor had float elems
  z3::expr get(const std::vector<z3::expr> &indices) const;

  Index getDim(uint64_t idx) const;
  std::vector<z3::expr> getDims() const { return dims; }

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

  Tensor reshape(const std::vector<z3::expr> &ns2) const;

  Tensor transpose() const;

  Tensor matmul(const Tensor &b) const;

  operator z3::expr() const { return arr; }

  // Returns (arr[idx] == src.arr[idx], idx var)
  std::pair<z3::expr, z3::expr> refines(const Tensor &src) const;

  static std::vector<z3::expr> getDims(mlir::TensorType tensorTy);

  // If tensorTy is unsupported, return nullopt
  static std::optional<std::pair<std::vector<z3::expr>, z3::sort>>
      getDimsAndElemTy(mlir::TensorType tensorTy);

  static Tensor mkLambda(
      std::vector<z3::expr> &&newdims,
      std::vector<z3::expr> &&indexvars, z3::expr body);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  Tensor eval(z3::model m) const;

private:
  z3::expr to1DArrayWithOfs(
      const std::vector<z3::expr> &offbegins,
      const std::vector<z3::expr> &sizes) const;
};
