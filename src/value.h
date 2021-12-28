#pragma once

#include "abstractops.h"
#include "simplevalue.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <variant>

class Memory;
class AccessInfo;

std::optional<smt::Sort> convertPrimitiveTypeToSort(mlir::Type ty);
std::optional<smt::Expr> getZero(mlir::Type eltType);
std::optional<smt::Expr> getIdentity(mlir::Type eltType);
void resetAbstractlyEncodedAttrs();

class Float {
  smt::Expr e;
  mlir::Type type;

public:
  Float(const smt::Expr &e, mlir::Type type): e(e), type(type) {
    assert(type.isa<mlir::FloatType>());
    this->e.unlockOps();
  }

  operator smt::Expr() const { return e; }

  static std::optional<smt::Sort> sort(mlir::Type ty);
  static smt::Sort sortFloat32();
  static Float var(std::string &&name, mlir::Type ty, VarType vty);
  static Float constant(const llvm::APFloat &apf, mlir::Type ty);
  static Float one(mlir::Type ty);

  // Returns e^x
  static Float exp(const Float &x);

  Float add(const Float &b) const;
  Float mul(const Float &b) const;
  Float div(const Float &b) const;
  Integer cmp(const mlir::arith::CmpFPredicate pred, const Float &b) const;
  Float abs() const;
  Float neg() const;
  Float extend(const mlir::Type &tgt_type) const;
  Float truncate(const mlir::Type &tgt_type) const;

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
  uint64_t getRank() const { return getDims().size(); }

  std::vector<Index> getDimsAsIndices() const {
    std::vector<Index> vec;
    auto dims = getDims();
    for (auto &d: dims)
      vec.emplace_back(d);
    return vec;
  }
  // Returns the element value.
  //   auto [v, inbounds] = shaped_value.get(indices)
  //   useAsInt(Integer(v)) // valid only if shaped_value has integer elems
  //   useAsFloat(Float(v)) // valid only if shaped_value has float elems
  // NOTE: Don't directly use the returned element (v)!
  // Please use it with a proper wrapper (Float, Index, Integer).
  // Using it without wrapper will raise an assertion failure 
  virtual smt::Expr get(const std::vector<smt::Expr> &indices) const = 0;

  // Basic dimension operation
  Index getDim(uint64_t idx) const { return Index(getDims()[idx]); }
  smt::Expr get1DSize() const { return smt::get1DSize(getDims()); }

  enum class ConvLayout {
    NHWC_HWCF, // image: nhwc, filter: hwcf, output: nhwf
    NCHW_FCHW, // image: nchw, filter: fchw, output: nfhw
    NHWC_FHWC  // image: nhwc, filter: fhwc, output: nhwf
  };

protected:
  // Linalg convolution operation.
  // returns: (indices, expr)
  // Caller must check the validity of inputs (e.g. inbounds, initializedness)
  std::pair<std::vector<smt::Expr>, smt::Expr> conv(const ShapedValue &filter,
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &dilations,
      ConvLayout layout) const;
};

class Tensor: public ShapedValue {
  std::vector<smt::Expr> dims;
  smt::Expr arr;
  // Index -> bool; Getting an uninitialized element is UB.
  smt::Expr initialized;

  Tensor(mlir::Type elemType, std::vector<smt::Expr> &&dims, smt::Expr &&arr,
         smt::Expr &&initialized):
      ShapedValue(elemType), dims(std::move(dims)), arr(std::move(arr)),
      initialized(std::move(initialized)) {}

public:
  static inline unsigned MAX_TENSOR_SIZE;
  static inline unsigned MAX_DIM_SIZE;
  static inline unsigned MAX_CONST_SIZE; // -1 if unbounded

  // A splat tensor.
  Tensor(mlir::Type elemType, smt::Expr &&splat_elem,
         std::vector<smt::Expr> &&dims);
  // A sparse tensor.
  Tensor(mlir::Type elemType,
         const std::vector<std::vector<uint64_t>> &indices,
         const std::vector<smt::Expr> &elems,
         const std::vector<uint64_t> &dims, const smt::Expr &zero);
  // A dense tensor (1 dimensional).
  Tensor(mlir::Type elemType, std::vector<smt::Expr> &&elems);


  smt::Expr asArray() const { return arr; }
  smt::Expr getWellDefined() const;

  smt::Expr isInBounds(const std::vector<smt::Expr> &indices) const;
  smt::Expr get(const std::vector<smt::Expr> &indices) const override;
  // Return arr[indexRaw]. The returned expr is locked.
  smt::Expr getRaw(const smt::Expr &indexRaw) const;
  smt::Expr isInitialized(const std::vector<smt::Expr> &indices) const;
  smt::Expr isFullyInitialized() const;

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

  // Concatenates this and t2 along a given axis.
  // ex) If this: <2x3xf32>, t2:<2x5xf32> and axis = 1, the result is a tensor
  //     of size <2x8xf32>.
  Tensor concat(const Tensor &t2, size_t axis);

  // Return a new tensor which is convolution of this tensor and filter.
  // Callers of conv must check whether filters/inputs/.. are initialized
  // (otherwise UB).
  Tensor conv(const Tensor &filter,
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &dilations,
      ConvLayout layout) const;

  // Return a new tensor which is depthwise convolution of this 2D tensor and filter.
  // Callers of conv must check whether filters/inputs/.. are initialized
  // (otherwise UB).
  Tensor depthwiseConv2D(const Tensor &filter,
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &dilations,
      const std::optional<Tensor> bias = std::nullopt) const;

  Tensor reshape(const std::vector<smt::Expr> &ns2) const;

  // Return a new tensor t s.t.
  // t[i_0]..[i_axis]..[i_(N-1)] = this[i_0]..[dim_axis - i_axis - 1]..[i_(N-1)]
  Tensor reverse(unsigned axis) const;

  // Return a new tensor that repeats for the given amount in each axis.
  Tensor tile(const std::vector<unsigned> &repeat) const;

  Tensor transpose() const;

  // Given two 2-dim tensors this and b, return their matrix multiplication
  // bTransposed is true, don't transpose b internally
  Tensor matmul(const Tensor &b, bool bTransposed = false) const;

  // Return the result of an elementwise operation.
  // Assume that the shapes are equivalent.
  Tensor elementwiseBinOp(const Tensor &b,
      mlir::Type resultElemType,
      const std::function<smt::Expr(smt::Expr &&, smt::Expr &&)> &op) const;

  Tensor elementwiseUnaryOp(
      mlir::Type resultElemType,
      const std::function<smt::Expr(smt::Expr &&)> &op) const;

  smt::Expr dot(const Tensor &b) const;
  smt::Expr sum() const;
  // Equivalent to tosa.reduce_sum
  // If this is a <N1 x N2 x ...> tensor, return a new tensor whose size at
  // the axis dimension is 1 and the corresponding elements contain summations
  // of elements.
  // Note that sum does not decrement the rank.
  Tensor sum(unsigned axis) const;

  operator smt::Expr() const { return arr; }

  static bool isTypeSupported(mlir::TensorType tensorTy);

  // Elements:    lambda indexvars, body
  // Initialized: lambda indexvars, isInitialized
  static Tensor mkLambda(
      mlir::Type elemType,
      std::vector<smt::Expr> &&newdims,
      std::vector<smt::Expr> &&indexvars,
      smt::Expr body, smt::Expr isInitialized);
  // Elements:    lambda indexvars, body
  static Tensor mkInitializedLambda(
      mlir::Type elemType,
      std::vector<smt::Expr> &&newdims,
      std::vector<smt::Expr> &&indexvars,
      smt::Expr body);
  // Elements:    lambda indexvar, body
  // Initialized: lambda indexvar, isInitialized
  static Tensor mkLambdaFrom1D(
      mlir::Type elemType,
      std::vector<smt::Expr> &&newdims,
      smt::Expr &&indexvar,
      smt::Expr body, smt::Expr initialized);

  // Returns (cond ? trueValue : falseValue).
  // The shapes of trueValue and falseValue must be equivalent.
  static Tensor mkIte(
      // Index -> boolean function
      std::function<smt::Expr(const std::vector<smt::Expr> &)> condFn,
      const Tensor &trueValue, const Tensor &falseValue);

  // A constant tensor from mlir::ElementsAttr and a static shape.
  static Tensor fromElemsAttr(mlir::RankedTensorType tensorTy,
      mlir::ElementsAttr attr);

  // A fresh tensor.
  static Tensor var(mlir::Type elemType, std::string &&name,
         const std::vector<uint64_t> &dims, bool initialized = true);
  static Tensor var(mlir::Type elemType, std::string &&name,
         const std::vector<smt::Expr> &dims, bool initialized = true);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Tensor &);
  // Returns (arr[idx] == src.arr[idx], unbound idx vars)
  // this: tgt, other: src
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
  // This should be parameterized later..
  static const unsigned MAX_MEMREF_SIZE = 1000000;
  static inline unsigned MAX_DIM_SIZE;

  class Layout {
  public:
    // Induction variables; they are bound (Expr::mkVar's flag is true)
    // ex) {d0, d1..}
    std::vector<smt::Expr> indVars;
    // Inbounds condition generator
    // ex) (d0, d1) -> 0 <= d0 < 3 && 0 <= d1 < 4 && ...
    using Fn = std::function<smt::Expr(const std::vector<smt::Expr> &args)>;
    Fn inbounds;
    // Layout mapping generator (ind vars -> 1D Index)
    // ex) mapping := (d0, d1) -> (4 * d0 + d1)
    Fn mapping;
    // Inverse layout mapping of indVars (1D Index -> indVars)
    // If we can not give exact definition of inverseMappings, then encode it
    // with uninterpreted function.
    // ex)
    // - If we can give exact definition
    //    inverseMappings := (idx) -> {(idx / 4), (idx % 4)}
    // - If we cannot give exact definition
    //    inverseMappings := (idx) -> {inverse0(idx), inverse1(idx)}
    using Fn2 = std::function<std::vector<smt::Expr>(const smt::Expr &)>;
    Fn2 inverseMappings;
    // Precondition for inverse mapping function.
    // If we cannot give exact definition of inverseMappings, then give its
    // meaning with forall quantifier.
    // This will be added to state's precondition only when inverseMappings are
    // used explicitly.
    // If the layout has a simple identity mapping, this will be constantly true
    // ex) forall indVars, if (indVars are inbounds) then
    //     inverse0(mapping(d0, d1)) = d0 && inverse1(mapping(d0, d1)) = d1
    smt::Expr precondition;

    Layout(const std::vector<smt::Expr> &dims);

    Layout(const std::vector<smt::Expr> &indVars,
        const Fn &layout,
        const Fn &inbounds,
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
    const Layout &layout,
    const smt::Expr &isViewRef);
  // Makes unbound SMT variables.
  MemRef(Memory *m,
    const mlir::Type &elemty,
    const std::string &name,
    const std::vector<smt::Expr> &dims,
    const Layout &layout);
  // Makes unbound SMT variables with fresh names.
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
  smt::Expr isViewReference() const { return isViewRef; }

  // Returns the value
  smt::Expr get(const std::vector<smt::Expr> &indices) const override;

  std::pair<smt::Expr, AccessInfo> getWithAccessInfo(
      const std::vector<smt::Expr> &indices) const;

  AccessInfo store(const smt::Expr &value,
      const std::vector<smt::Expr> &indices) const;
  AccessInfo storeArray(const smt::Expr &array,
      const smt::Expr &startOffset, const smt::Expr &size) const;

  Tensor loadTensorWithoutCheck() const;

  smt::Expr isInBounds() const;
  smt::Expr isGlobalBlock() const;
  smt::Expr isLocalBlock() const;
  smt::Expr getLiveness() const;
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
      const std::vector<smt::Expr> &strides,
      const std::vector<smt::Expr> &dilations,
      ConvLayout convLayout);

  // Returns (cond ? trueValue : falseValue).
  // It is assumed that trueValue.layout is equivalent to falseValue.layout.
  // Also trueValue.dims == falseValue.dims is assumed, to be consistent with
  // layout info.
  static MemRef mkIte(smt::Expr cond,
      const MemRef &trueValue, const MemRef &falseValue);

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
  smt::Expr isViewRef; // Is this MemRef created from view operations?

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


using ValueTy = std::variant<Tensor, MemRef, Index, Float, Integer>;

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const ValueTy &);
smt::Expr getExpr(const ValueTy &vty);
ValueTy eval(const ValueTy &vty, smt::Model m);
ValueTy attrToValueTy(mlir::Attribute a);
std::optional<ValueTy> fromExpr(smt::Expr &&e, mlir::Type ty);
std::pair<smt::Expr, std::vector<smt::Expr>> refines(
    const ValueTy &v_tgt, const ValueTy &v_src);
