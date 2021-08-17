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

  Index(unsigned);
  Index(std::string &&name, bool freshvar = false);
  Index(const smt::expr &e);

  operator smt::expr() const { return e; }
  Index ofs(int i) const {
    uint64_t v;
    if (e.is_numeral_u64(v))
      return Index(v + i);
    return Index(e + i);
  }

  static smt::sort sort();
  static Index one();
  static Index zero();

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Index &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Index &other) const;
  Index eval(smt::model m) const;
};

class Float {
  smt::expr e;

public:
  Float(std::string &&name);
  Float(const smt::expr &e): e(e) {}
  Float(const llvm::APFloat &apf);
  Float(double f);

  operator smt::expr() const { return e; }

  static smt::sort sort();

  Float add(const Float &b) const;
  Float mul(const Float &b) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Float &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Float &other) const;
  Float eval(smt::model m) const;
};

class Integer {
  smt::expr e;

public:
  Integer(std::string &&name, unsigned bw);
  Integer(const smt::expr &e): e(e) {}
  Integer(int64_t i, unsigned bw);
  Integer(const llvm::APInt &api);

  operator smt::expr() const { return e; }

  static smt::sort sort(unsigned bw);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Integer &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(const Integer &other)
      const;
  Integer eval(smt::model m) const;
};

class Tensor {
  std::vector<smt::expr> dims;
  smt::expr arr;

  Tensor(std::vector<smt::expr> &&dims, smt::expr &&arr):
      dims(std::move(dims)), arr(std::move(arr)){}

public:
  // This may be parameterized later..
  static const unsigned MAX_TENSOR_SIZE = 10000;
  static const unsigned MAX_DIM_SIZE = 25;

  // A splat tensor.
  Tensor(const smt::expr &splat_elem, const std::vector<smt::expr> &dims);
  // A sparse tensor.
  Tensor(const std::vector<std::vector<uint64_t>> &indices,
         const std::vector<smt::expr> &elems,
         const std::vector<uint64_t> &dims, const smt::expr &zero);

  Tensor(const std::vector<smt::expr> &elems1d);
  Tensor(std::string &&name, const std::vector<smt::expr> &dims,
         const smt::sort &elemty);

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
      std::vector<smt::expr> &&newsizes) const;

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
  static std::optional<std::pair<std::vector<smt::expr>, smt::sort>>
      getDimsAndElemTy(mlir::TensorType tensorTy,
                       bool freshVarForUnknownSize = true);

  static std::optional<smt::sort> getElemTy(mlir::TensorType tensorTy);

  static Tensor mkLambda(
      std::vector<smt::expr> &&newdims,
      std::vector<smt::expr> &&indexvars, smt::expr body);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  // Returns (arr[idx] == src.arr[idx], idx var)
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Tensor &other) const;
  Tensor eval(smt::model m) const;

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
    // Induction variables
    // ex) {d0, d1..}
    std::vector<smt::expr> indVars;
    // Inbounds condition for induction variables
    // ex) (d0, d1) -> 0 <= d0 < 3 && 0 <= d1 < 4 && ...
    smt::expr inbounds;
    // Layout mapping of indVars (indVars -> 1D Index)
    // ex) mapping := (d0, d1) -> (4 * d0 + d1)
    smt::expr mapping;
    // Inverse layout mapping of indVars (1D Index -> indVars)
    // If we can not give exact definition of inverseMappings, then encode it with uninterpreted function.
    // ex)
    // - If we can give exact definition
    //    inverseMappings := (idx) -> {(idx / 4), (idx % 4)}
    // - If we cannot give exact definition
    //    inverseMappings := (idx) -> {inverse0(idx), inverse1(idx)}
    std::vector<smt::expr> inverseMappings;
    // Precondition for inverse mapping function.
    // If we cannot give exact definition of inverseMappings, then give its meaning with forall quantifier.
    // This will be added to state's precondition only when inverseMappings are used explicitly.
    // ex) forall indVars, if (indVars are inbounds) then inverse0(mapping(d0, d1)) = d0 && inverse1(mapping(d0, d1)) = d1
    smt::expr precondition;

    Layout(const std::vector<smt::expr> &indVars,
        const smt::expr &inbounds,
        const smt::expr &mapping,
        const std::vector<smt::expr> &inverseMappings,
        const smt::expr &precondition):
      indVars(indVars), inbounds(inbounds),
      mapping(mapping), inverseMappings(inverseMappings), precondition(precondition) {}

    // MARK(makesource)
    // Without this copy constructor, I encounter libc+abi.dylib related error in MacOS
    Layout(const Layout& copy):
      indVars(copy.indVars), inbounds(copy.inbounds),
      mapping(copy.mapping), inverseMappings(copy.inverseMappings),
      precondition(copy.precondition) {}
  };

  MemRef(Memory *m,
    const smt::expr &bid,
    const smt::expr &offset,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const smt::sort &elemty);
  MemRef(Memory *m,
    const std::string &name,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const smt::sort &elemty);
  MemRef(Memory *m,
    const std::vector<smt::expr> &dims,
    const Layout &layout,
    const smt::sort &elemty);

  operator smt::expr() const { return bid && offset; }

  smt::expr getPrecondition() const;
  smt::expr getWellDefined() const;

  // If memRefTy is unsupported, return nullopt
  static std::optional<std::tuple<std::vector<smt::expr>, Layout, smt::sort>>
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

  // Return a new memerf which is subview of source memref.
  MemRef subview(const std::vector<smt::expr> &offsets,
      const std::vector<smt::expr> &sizes,
      const std::vector<smt::expr> &strides,
      int rankDiff = 0);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MemRef &);
  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const MemRef &other) const;
  MemRef eval(smt::model m) const;

private:
  Memory *m;
  smt::expr bid; // blockID
  Index offset; // offset
  std::vector<smt::expr> dims;
  Layout layout; // memory layout defined by affine_map (ex. s0 * idx0 + s1 * idx1 + ... + offset)

  smt::expr to1DArrayWithOfs(
      const std::vector<smt::expr> &offbegins,
      const std::vector<smt::expr> &sizes) const;
  std::pair<smt::expr, smt::expr> to1DIdxWithLayout(const std::vector<smt::expr> &idxs);

  MemRef::Layout createSubViewLayout(const std::vector<smt::expr> &indVars,
      const std::vector<smt::expr> &offsets,
      const std::vector<smt::expr> &strides);
};
