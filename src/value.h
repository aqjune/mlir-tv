#pragma once

#include "smt.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <string>
#include <optional>
#include <vector>

class Memory;

class Index {
  smt::expr e;

public:
  static const unsigned BITS = 32;

  Index();
  Index(unsigned);
  Index(const std::string &name, bool freshvar = false);
  Index(const smt::expr &e);

  operator smt::expr() const { return e; }
  Index ofs(int i) const {
    uint64_t v;
    if (e.is_numeral_u64(v))
      return Index(v + i);
    return Index(e + i);
  }

  static z3::sort sort();
  static Index one();
  static Index zero();

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Index &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Index &other) const;
  Index eval(z3::model m) const;
};

class Float {
  smt::expr e;

public:
  static const unsigned BITS = 4;

  Float(const std::string &name);
  Float(const smt::expr &e): e(e) {}
  Float(const llvm::APFloat &apf);
  Float(double f);

  operator smt::expr() const { return e; }

  static z3::sort sort();

  Float add(const Float &b) const;
  Float mul(const Float &b) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Float &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Float &other) const;
  Float eval(z3::model m) const;
};

class Integer {
  smt::expr e;

public:
  Integer(const std::string &name, unsigned bw);
  Integer(const smt::expr &e): e(e) {}
  Integer(int64_t i, unsigned bw);

  operator smt::expr() const { return e; }

  static z3::sort sort(unsigned bw);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Integer &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(const Integer &other)
      const;
  Integer eval(z3::model m) const;
};

class Tensor {
  std::vector<smt::expr> dims;
  smt::expr arr;

public:
  // This may be parameterized later..
  static const unsigned MAX_TENSOR_SIZE = 10000;
  static const unsigned MAX_DIM_SIZE = 25;

  Tensor();
  // A splat tensor.
  Tensor(const smt::expr &splat_elem, const std::vector<smt::expr> &dims);
  Tensor(const std::vector<smt::expr> &elems1d);
  Tensor(const std::string &name, const std::vector<smt::expr> &dims,
         const z3::sort &elemty);

  smt::expr asArray() const { return arr; }

  smt::expr getWellDefined() const;

  // Return the element at indices.
  //   expr v = tensor.get(indices)
  //   useAsInt(Integer(v)) // valid only if tensor had integer elems
  //   useAsFloat(Float(v)) // valid only if tensor had float elems
  smt::expr get(const std::vector<smt::expr> &indices) const;

  smt::expr get1DSize() const { return smt::get1DSize(dims); }

  Index getDim(uint64_t idx) const;
  std::vector<smt::expr> getDims() const { return dims; }

  // Return a new tensor T2 s.t.
  //   T2[newidxvars] = this[srcidxs]
  // For example, if newidxvars = [x, y, z] and srcidxs = [x, y + z],
  //   T2[x][y][z] = this[x][y + z]
  Tensor affine(
      const std::vector<smt::expr> &newidxvars,
      std::vector<smt::expr> srcidxs,
      const std::vector<smt::expr> &newsizes) const;

  // Return a new tensor T2 s.t.
  //   T2[i1][i2]..[iN] = this[i2]..[iN][i1]
  Tensor rotateDimensions() const;

  // Return a new tensor which is convolution of this tensor and filter.
  Tensor conv(const Tensor &filter) const;

  Tensor reshape(const std::vector<smt::expr> &ns2) const;

  Tensor transpose() const;

  Tensor matmul(const Tensor &b) const;

  smt::expr dot(const Tensor &b) const;
  smt::expr sum() const;

  operator smt::expr() const { return arr; }

  // If tensorTy is unsupported, return nullopt
  static std::optional<std::pair<std::vector<smt::expr>, z3::sort>>
      getDimsAndElemTy(mlir::TensorType tensorTy,
                       bool freshVarForUnknownSize = true);

  static std::optional<z3::sort> getElemTy(mlir::TensorType tensorTy);

  static Tensor mkLambda(
      std::vector<smt::expr> &&newdims,
      std::vector<smt::expr> &&indexvars, smt::expr body);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  // Returns (arr[idx] == src.arr[idx], idx var)
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Tensor &other) const;
  Tensor eval(z3::model m) const;

private:
  smt::expr to1DArrayWithOfs(
      const std::vector<smt::expr> &offbegins,
      const std::vector<smt::expr> &sizes) const;
};

class MemRef {
public:
  // This may be parameterized later..
  static const unsigned MAX_MEMREF_SIZE = 1000000;
  static const unsigned MAX_DIM_SIZE = 1000;

  class Layout {
  public:
    std::vector<smt::expr> indVars;
    smt::expr expr;

    Layout(const std::vector<smt::expr> &indVars, const smt::expr &expr):
      indVars(indVars), expr(expr) {}
  };

  MemRef(Memory *m);
  MemRef(Memory *m,
    const smt::expr &bid,
    const smt::expr &offset,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const z3::sort &elemty);
  MemRef(Memory *m,
    const std::string &name,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const z3::sort &elemty);
  MemRef(Memory *m,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const z3::sort &elemty);

  operator smt::expr() const { return bid && offset; }

  smt::expr getWellDefined() const;

  // If memRefTy is unsupported, return nullopt
  static std::optional<std::tuple<std::vector<smt::expr>, Layout, z3::sort>>
    getDimsAndLayoutAndElemTy(mlir::MemRefType memRefTy,
      std::optional<std::vector<z3::expr>> predefinedDims = {},
      bool freshVarForUnknownSize = true);

  std::pair<smt::expr, smt::expr> load(const std::vector<smt::expr> &indices);
  smt::expr store(const smt::expr &value, const std::vector<smt::expr> &indices);
  smt::expr storeArray(const smt::expr &array, const smt::expr &startOffset, const smt::expr &size);
  smt::expr isInBounds() const;
  smt::expr isGlobalBlock() const;
  smt::expr isLocalBlock() const;
  smt::expr getBID() const { return bid; }
  Index getOffset() const { return offset; }
  smt::expr get1DSize() const { return smt::get1DSize(dims); }
  Index getDim(uint64_t idx) const;
  std::vector<smt::expr> getDims() const { return dims; }
  void setWritable(bool writable);
  void setMemory(Memory *m) { this->m = m; }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MemRef &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const MemRef &other) const;
  MemRef eval(z3::model m) const;

  private:
  Memory *m;
  smt::expr bid; // blockID
  Index offset; // offset
  std::vector<smt::expr> dims;
  Layout layout; // memory layout defined by affine_map (ex. s0 * idx0 + s1 * idx1 + ... + offset)

  smt::expr to1DArrayWithOfs(
      const std::vector<smt::expr> &offbegins,
      const std::vector<smt::expr> &sizes) const;
  smt::expr to1DIdxWithLayout(const std::vector<smt::expr> &idxs);
};
