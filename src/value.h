#pragma once

#include "z3++.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <string>
#include <vector>

class Memory;

z3::expr get1DSize(const std::vector<z3::expr> &dims);
std::vector<z3::expr> getDims(
    const mlir::ShapedType &shapedTy, bool freshVarForUnknownSize = false);

class Index {
  z3::expr e;

public:
  static const unsigned BITS = 32;

  Index();
  Index(unsigned);
  Index(const std::string &name, bool freshvar = false);
  Index(const z3::expr &e);

  operator z3::expr() const { return e; }
  Index ofs(int i) const { return Index(e + i); }

  static z3::sort sort();
  static Index one();
  static Index zero();

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Index &);
  std::pair<z3::expr, std::vector<z3::expr>> refines(const Index &src) const;
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
  std::pair<z3::expr, std::vector<z3::expr>> refines(const Float &src) const;
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
  std::pair<z3::expr, std::vector<z3::expr>> refines(const Integer &src) const;
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

  z3::expr get1DSize() const { return ::get1DSize(dims); }

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

  z3::expr sum() const;

  operator z3::expr() const { return arr; }

  // If tensorTy is unsupported, return nullopt
  static std::optional<std::pair<std::vector<z3::expr>, z3::sort>>
      getDimsAndElemTy(mlir::TensorType tensorTy);

  static Tensor mkLambda(
      std::vector<z3::expr> &&newdims,
      std::vector<z3::expr> &&indexvars, z3::expr body);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  // Returns (arr[idx] == src.arr[idx], idx var)
  std::pair<z3::expr, std::vector<z3::expr>> refines(const Tensor &src) const;
  Tensor eval(z3::model m) const;

private:
  z3::expr to1DArrayWithOfs(
      const std::vector<z3::expr> &offbegins,
      const std::vector<z3::expr> &sizes) const;
};

class MemRef {
  z3::expr bid; // blockID
  Index offset; // offset
  std::vector<z3::expr> dims;

public:
  MemRef();
  MemRef(const std::string &name, const std::vector<z3::expr> &dims,
         const z3::sort &elemty);

  operator z3::expr() const { return bid && offset; }

  // If memRefTy is unsupported, return nullopt
  static std::optional<std::pair<std::vector<z3::expr>, z3::sort>>
      getDimsAndElemTy(mlir::MemRefType memRefTy);

  std::pair<z3::expr, z3::expr> get(const Memory &m, const std::vector<z3::expr> &indices) const;
  z3::expr set(const Memory &m,
    const std::vector<z3::expr> &indices,
    const z3::expr &value) const;
  z3::expr getBID() const { return bid; }
  Index getOffset() const { return offset; }
  Index getDim(uint64_t idx) const;
  std::vector<z3::expr> getDims() const { return dims; }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MemRef &);
  std::pair<z3::expr, std::vector<z3::expr>> refines(const MemRef &src) const;
  MemRef eval(z3::model m) const;

  private:
    z3::expr to1DArrayWithOfs(
      const std::vector<z3::expr> &offbegins,
      const std::vector<z3::expr> &sizes) const;
};
