#pragma once

#include "simplevalue.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

class Memory;

std::optional<smt::Sort> convertTypeToSort(mlir::Type ty);
std::optional<smt::Expr> getZero(mlir::Type eltType);


class Float {
  smt::Expr e;
  mlir::Type type;

public:
  Float(const smt::Expr &e, mlir::Type type): e(e), type(type) {}

  operator smt::Expr() const { return e; }

  static std::optional<smt::Sort> sort(mlir::Type ty);
  static smt::Sort sortFloat32();
  static Float var(std::string &&name, mlir::Type ty, VarType vty);
  static Float constant(const llvm::APFloat &apf, mlir::Type ty);

  Float add(const Float &b) const;
  Float mul(const Float &b) const;
  Float abs() const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Float &);
  // (refinement, {})
  std::pair<smt::Expr, std::vector<smt::Expr>> refines(
      const Float &other) const;
  Float eval(smt::Model m) const;
};


class ShapedValue {
protected:
  mlir::Type elemType;

public:
  ShapedValue(mlir::Type elemType): elemType(elemType) {}

  // If (freshVarForUnknownSizes, valsForUnknownSz) is
  // (1) (false, nullopt): shapedTy must not have an unknown sized dimension
  // (2) (true, _): unknown sized dimensions are assigned fresh variables
  // (3) (false, some(exprs)): unknown sized dimensions are assigned exprs[0..]
  static std::vector<smt::Expr> getDims(
      const mlir::ShapedType &shapedTy, bool freshVarForUnknownSizes = true,
      std::optional<std::vector<smt::Expr>> &&valsForUnknownSz = std::nullopt);

  mlir::Type getElemType() const { return elemType; }

  virtual std::vector<smt::Expr> getDims() const = 0;
  virtual std::pair<smt::Expr, smt::Expr> get(const std::vector<smt::Expr> &indices) const = 0;

  // Basic dimension operation
  Index getDim(uint64_t idx) const { return Index(getDims()[idx]); }
  smt::Expr get1DSize() const { return smt::get1DSize(getDims()); }

  // Linalg convoluion operation
  // (indices, expr)
  std::pair<std::vector<smt::Expr>, smt::Expr> conv(const ShapedValue &filter,
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &dilations) const;
};

class Tensor: public ShapedValue {
  std::vector<smt::Expr> dims;
  smt::Expr arr;

  Tensor(mlir::Type elemType, std::vector<smt::Expr> &&dims, smt::Expr &&arr):
      ShapedValue(elemType), dims(std::move(dims)), arr(std::move(arr)) {}

public:
  // This may be parameterized later..
  static const unsigned MAX_TENSOR_SIZE = 10000;
  static const unsigned MAX_DIM_SIZE = 25;

  // A splat tensor.
  Tensor(mlir::Type elemType, smt::Expr &&splat_elem,
         std::vector<smt::Expr> &&dims);
  // A sparse tensor.
  Tensor(mlir::Type elemType,
         const std::vector<std::vector<uint64_t>> &indices,
         const std::vector<smt::Expr> &elems,
         const std::vector<uint64_t> &dims, const smt::Expr &zero);
  Tensor(mlir::Type elemType, std::vector<smt::Expr> &&elems1d);
  Tensor(mlir::Type elemType, std::string &&name,
         const std::vector<smt::Expr> &dims);


  smt::Expr asArray() const { return arr; }
  smt::Expr getWellDefined() const;

  // Return the element at indices.
  //   Expr v = tensor.get(indices)
  //   useAsInt(Integer(v)) // valid only if tensor had integer elems
  //   useAsFloat(Float(v)) // valid only if tensor had float elems
  std::pair<smt::Expr, smt::Expr> get(const std::vector<smt::Expr> &indices)
      const override;

  std::vector<smt::Expr> getDims() const override { return dims; }

  size_t getRank() const { return dims.size(); }

  // Return <a new tensor T2, inbounds>
  // T2[idx] = idx == indices ? value : this[idx]
  std::pair<Tensor, smt::Expr> insert(
      const smt::Expr &value, const std::vector<smt::Expr> &indices) const;

  // Return a new tensor T2 s.t.
  //   T2[newidxvars] = this[srcidxs]
  // For example, if newidxvars = [x, y, z] and srcidxs = [x, y + z],
  //   T2[x][y][z] = this[x][y + z]
  Tensor affine(
      const std::vector<smt::Expr> &newidxvars,
      std::vector<smt::Expr> srcidxs,
      std::vector<smt::Expr> &&newsizes) const;

  // Return a new tensor which is convolution of this tensor and filter.
  Tensor conv(const Tensor &filter,
      const std::vector<smt::Expr> strides,
      const std::vector<smt::Expr> dilations) const;

  Tensor reshape(const std::vector<smt::Expr> &ns2) const;

  Tensor transpose() const;

  Tensor matmul(const Tensor &b) const;

  // Return the result of an elementwise operation
  // (the resulting tensor, shape check)
  std::pair<Tensor, smt::Expr> elementwiseBinOp(
      const Tensor &b,
      const std::function<smt::Expr(smt::Expr &&, smt::Expr &&)> &op) const;

  Tensor elementwiseUnaryOp(
      mlir::Type resultElemType,
      const std::function<smt::Expr(smt::Expr &&)> &op) const;

  smt::Expr dot(const Tensor &b) const;
  smt::Expr sum() const;

  operator smt::Expr() const { return arr; }

  static bool isTypeSupported(mlir::TensorType tensorTy);

  static Tensor mkLambda(
      mlir::Type elemType,
      std::vector<smt::Expr> &&newdims,
      std::vector<smt::Expr> &&indexvars, smt::Expr body);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  // Returns (arr[idx] == src.arr[idx], unbound idx vars)
  std::pair<smt::Expr, std::vector<smt::Expr>> refines(
      const Tensor &other) const;
  Tensor eval(smt::Model m) const;

private:
  smt::Expr to1DArrayWithOfs(
      const std::vector<smt::Expr> &offbegins,
      const std::vector<smt::Expr> &sizes) const;
};

class MemRef: public ShapedValue {
public:
  // This may be parameterized later..
  static const unsigned MAX_MEMREF_SIZE = 1000000;
  static const unsigned MAX_DIM_SIZE = 1000;

  class Layout {
  public:
    // Induction variables; they are bound (Expr::mkVar's flag is true)
    // ex) {d0, d1..}
    std::vector<smt::Expr> indVars;
    // Inbounds condition for induction variables
    // ex) (d0, d1) -> 0 <= d0 < 3 && 0 <= d1 < 4 && ...
    smt::Expr inbounds;
    // Layout mapping of indVars (indVars -> 1D Index)
    // ex) mapping := (d0, d1) -> (4 * d0 + d1)
    smt::Expr mapping;
    // Inverse layout mapping of indVars (1D Index -> indVars)
    // If we can not give exact definition of inverseMappings, then encode it with uninterpreted function.
    // ex)
    // - If we can give exact definition
    //    inverseMappings := (idx) -> {(idx / 4), (idx % 4)}
    // - If we cannot give exact definition
    //    inverseMappings := (idx) -> {inverse0(idx), inverse1(idx)}
    std::vector<smt::Expr> inverseMappings;
    // Precondition for inverse mapping function.
    // If we cannot give exact definition of inverseMappings, then give its meaning with forall quantifier.
    // This will be added to state's precondition only when inverseMappings are used explicitly.
    // If the layout has simple identity mapping, this will be constantly true.
    // ex) forall indVars, if (indVars are inbounds) then inverse0(mapping(d0, d1)) = d0 && inverse1(mapping(d0, d1)) = d1
    smt::Expr precondition;

    Layout(const std::vector<smt::Expr> &dims);

    Layout(const std::vector<smt::Expr> &indVars,
        const smt::Expr &layout,
        const smt::Expr &inbounds,
        bool useUF = false); // encode "mapping" using uninterpreted function

    // MARK(makesource)
    // Without this copy constructor, I encounter libc+abi.dylib related error in MacOS
    Layout(const Layout& copy):
      indVars(copy.indVars), inbounds(copy.inbounds),
      mapping(copy.mapping), inverseMappings(copy.inverseMappings),
      precondition(copy.precondition) {}

    std::vector<smt::Expr> getInverseIndices(const smt::Expr &idx) const;
  };

  MemRef(Memory *m,
    const mlir::Type &elemty,
    const smt::Expr &bid,
    const smt::Expr &offset,
    const std::vector<smt::Expr> &dims,
    const Layout &layout);
  // Makes an unbound variable.
  MemRef(Memory *m,
    const mlir::Type &elemty,
    const std::string &name,
    const std::vector<smt::Expr> &dims,
    const Layout &layout);
  MemRef(Memory *m,
    const mlir::Type &elemty,
    const std::vector<smt::Expr> &dims,
    const Layout &layout);

  // FIXME: Remove this function?
  operator smt::Expr() const { return bid; }

  smt::Expr getPrecondition() const;
  smt::Expr getWellDefined() const;

  static bool isTypeSupported(mlir::MemRefType memRefTy);
  // memRefTy must pass isTypeSupported(memRefTy)
  static Layout getLayout(
      mlir::MemRefType memRefTy, const std::vector<smt::Expr> &dims);

  // Property getters
  smt::Expr getBID() const { return bid; }
  Index getOffset() const { return offset; }
  std::vector<smt::Expr> getDims() const override { return dims; }

  std::pair<smt::Expr, smt::Expr> get(const std::vector<smt::Expr> &indices) const override;
  smt::Expr store(const smt::Expr &value, const std::vector<smt::Expr> &indices)
      const;
  smt::Expr storeArray(const smt::Expr &array, const smt::Expr &startOffset,
      const smt::Expr &size, bool ubIfReadonly = true) const;
  smt::Expr isInBounds() const;
  smt::Expr isGlobalBlock() const;
  smt::Expr isLocalBlock() const;
  smt::Expr noalias(const MemRef &other) const;
  void setWritable(bool writable);
  void setMemory(Memory *m) { this->m = m; }
  bool isIdentityMap() const;

  // Return a new memref which is a subview of the source memref.
  MemRef subview(const std::vector<smt::Expr> &offsets,
      const std::vector<smt::Expr> &sizes,
      const std::vector<smt::Expr> &strides,
      const llvm::SmallDenseSet<unsigned> &unusedDims,
      int rankDiff = 0);

  // Store results which is convolution of input, filter and return wellDefined.
  smt::Expr conv(const MemRef &input,
      const MemRef &filter,
      const std::vector<smt::Expr> strides,
      const std::vector<smt::Expr> dilations);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MemRef &);
  // (refinement, unbound variables used in the refinement formula)
  std::pair<smt::Expr, std::vector<smt::Expr>> refines(
      const MemRef &other) const;
  MemRef eval(smt::Model m) const;

private:
  Memory *m;
  smt::Expr bid; // blockID
  Index offset; // offset
  std::vector<smt::Expr> dims;
  Layout layout; // memory layout defined by affine_map
                 // (ex. s0 * idx0 + s1 * idx1 + ... + offset)

  smt::Expr to1DArrayWithOfs(
      const std::vector<smt::Expr> &offbegins,
      const std::vector<smt::Expr> &sizes) const;
  std::pair<smt::Expr, smt::Expr> to1DIdxWithLayout(
      const std::vector<smt::Expr> &idxs) const;

  MemRef::Layout createSubViewLayout(const std::vector<smt::Expr> &indVars,
      const std::vector<smt::Expr> &offsets,
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &sizes);
};
