#include "encode.h"
#include "abstractops.h"
#include "debug.h"
#include "function.h"
#include "opts.h"
#include "smt.h"
#include "utils.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

using namespace smt;
using namespace std;

llvm::cl::opt<bool> arg_assign_random_to_unsupported_ops(
      "assign-random-to-unsupported-ops",
  llvm::cl::desc("Assign a random value to the result of unsupported ops. "
      "Note that this option is purely for debugging purpose. This flag will "
      "make the validation result meaningless."),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> arg_use_neg_zero(
      "use-neg-zero",
  llvm::cl::desc("For linalg.fill operations filling positive zero or "
      "linalg.yield with positive zero operand, use negative zero instead. "
      "This is a workaround to check signed zero issue when lowering tosa"
      " to linalg."),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::list<string> arg_use_arg_dims("use-fn-argument-dims",
  llvm::cl::desc(
    "Specify the function argument to use as a reference for the "
          "output dynamic dims"),
  llvm::cl::value_desc("<function_name>@idx"),
  llvm::cl::CommaSeparated,
  llvm::cl::cat(MlirTvCategory)
);

namespace {
  optional<map<string, int64_t, std::less<>>> dimsReferenceIdxMap;

  optional<int64_t> getDimsReferenceIdx(const string_view functionName) {
    auto itr = dimsReferenceIdxMap->find(functionName);
    if (itr != dimsReferenceIdxMap->end()) {
      return itr->second;
    } else {
      return nullopt;
    }
  }

  void encodeShiftAmountBound(State &st, mlir::Operation *op) {
    const auto arg = op->getOperand(0);
    const auto argTy = arg.getType();
    const auto amnt = op->getOperand(1);
    const auto amntTy = amnt.getType();
    smart_assert(argTy == amntTy,
                 "Shift argument and amount types must be the same!");

    if (amntTy.isa<mlir::TensorType>()) {
      const auto amntTensor = st.regs.get<Tensor>(amnt);
      const auto bw = amntTensor.getElemType().getIntOrFloatBitWidth();
      const auto vars = Index::boundIndexVars(amntTensor.getRank());
      const auto amntBound = Integer(bw, bw);
      st.wellDefined(
          op,
          Expr::mkForall(
              vars, static_cast<Expr>(amntTensor).select(vars).ult(amntBound)));
    } else if (amntTy.isa<mlir::IntegerType>()) {
      const auto amntInteger = st.regs.get<Integer>(amnt);
      const auto bw = amntInteger.bitwidth();
      const auto amntBound = Integer(bw, bw);
      st.wellDefined(op, static_cast<Expr>(amntInteger).ult(amntBound));
    } else if (argTy.isa<mlir::IndexType>() && amntTy.isa<mlir::IndexType>()) {
      const auto amntIndex = st.regs.get<Index>(amnt);
      const auto bw = Index::BITS;
      const auto amntBound = Index(bw);
      st.wellDefined(op, static_cast<Expr>(amntIndex).ult(amntBound));
    } else {
      throw UnsupportedException(op, "Unsupported shift operands");
    }
  }

enum class FPPrecision {
  // F16,
  F32,
  F64
};

FPPrecision getPrecision(const mlir::Type &type) {
  if (type.isF16()) {
    // tgt_prec = FPPrecision::F16;
    throw UnsupportedException(type, "F16 is not supported yet");
  } else if (type.isF32()) {
    return FPPrecision::F32;
  } else if (type.isF64()) {
    return FPPrecision::F64;
  } else {
    throw UnsupportedException(type, "unsupported FP type");
  }
}
} // namespace

// map := (i, j, k) -> (j, k, i)
// input := [a, b, c]
// output := [b, c, a]
static vector<Expr> doMap(
    const vector<Expr> &input, const mlir::AffineMap &map) {
  if (map.isIdentity())
    return input;

  vector<Expr> output;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    auto ade = map.getResult(i).dyn_cast<mlir::AffineDimExpr>();
    output.push_back(input[ade.getPosition()]);
  }
  return output;
}

template<class T>
static vector<T> vecAddElem(const vector<T> &a, const T &b) {
  vector<T> c;
  for (unsigned i = 0; i < a.size(); ++i)
    c.push_back(a[i] + b);
  return c;
}

static vector<Expr> addOne(vector<Expr> &&vec) {
  if (vec.empty())
    return {};
  return vecAddElem(vec, Expr::mkBV(1, vec[0].bitwidth()));
}

template<class T>
static vector<T> vecAdd(const vector<T> &a, const vector<T> &b) {
  assert(a.size() == b.size());
  vector<T> c;
  for (unsigned i = 0; i < a.size(); ++i)
    c.push_back(a[i] + b[i]);
  return c;
}

static Expr evalIndexCastOp(mlir::Type src, mlir::Type tgt, Index &&idx) {
  Expr val = idx;
  assert(val.sort().isBV());

  unsigned srcWidth = val.sort().bitwidth();

  unsigned destWidth = 0;
  if (auto dstty = tgt.dyn_cast<mlir::IntegerType>())
    destWidth = dstty.getWidth();
  else {
    assert(tgt.isIndex());
    destWidth = Index::BITS;
  }

  Expr casted = val;
  if (srcWidth > destWidth)
    casted = val.extract(destWidth - 1, 0);
  else if (srcWidth < destWidth)
    casted = val.sext(destWidth - srcWidth);
  return casted;
}

static void addToBoolMap(map<string, Expr> &m, std::string &&key, Expr &&b) {
  auto itr = m.find(key);
  if (itr == m.end()) {
    m.emplace(std::move(key), std::move(b));
  } else {
    itr->second = itr->second & std::move(b);
  }
}

static void storeTensorTo(
    State &st, mlir::Operation *op, Tensor &&tensor, const MemRef &memref,
    mlir::MemRefType memrefTy, bool ubIfReadOnly) {
  // Accessing uninitialized elem is UB.
  st.wellDefined(op, tensor.isFullyInitialized(), "tensor initialized");
  auto elemTy = memrefTy.getElementType();

  if (memrefTy.getLayout().isIdentity()) {
    // memref with identity map
    auto success = st.m->storeArray(elemTy,
        tensor.asArray(), memref.getBID(), memref.getOffset(),
        tensor.get1DSize());
    st.wellDefined(op, success.checkWrite(!ubIfReadOnly),
        "storing to dest");
    st.hasQuantifier |= tensor.isFullyInitialized().hasQuantifier();

  } else {
    // TODO: can we further optimize this if we know that memref is a
    // freshly created block?
    // We may not need to preserve the 'previous' bytes.

    vector<Expr> idxs = Index::boundIndexVars(memrefTy.getRank());
    auto ofs1d = Index::var("ofs", VarType::BOUND);
    auto tVal = tensor.get(idxs);
    auto tInBounds = tensor.isInBounds(idxs);

    auto [mValBefore1d, mInfoBefore1d] = st.m->load(elemTy, memref.getBID(),
        ofs1d);
    auto mInitializedBefore1d = mInfoBefore1d.initialized;

    // Create a fresh SMT array for the block pointed by memref!
    st.m->freshArray(elemTy, memref.getBID());

    auto [mValAfter, mInfoAfter] = memref.getWithAccessInfo(idxs);

    auto [mValAfter1d, mInfoAfter1d] = st.m->load(elemTy, memref.getBID(),
        ofs1d);
    auto mInitializedAfter1d = mInfoAfter1d.initialized;

    // Wrote successfully
    st.wellDefined(op, Expr::mkForall(idxs,
        tInBounds.implies(mInfoAfter.checkWrite(!ubIfReadOnly))),
        "write successful");

    // Write preconditions that relates the arrays before/after writes.
    // A precondition for the updated elements
    auto precUpdated = Expr::mkForall(idxs,
        tInBounds.implies(mValAfter == tVal));
    st.addPrecondition(std::move(precUpdated));
    auto precInit = Expr::mkForall(idxs,
        tInBounds.implies(mInfoAfter.initialized));
    st.addPrecondition(std::move(precInit));

    // A precondition for the untouched elements
    auto precPreserved = Expr::mkForall({ofs1d},
        // If ofs1d is a valid block offset that cannot be reached using this
        // memref...
        (mInfoAfter1d.inbounds & !memref.isValid1DOffset(ofs1d))
          // The values and initialized bits are preserved.
          .implies((mValBefore1d == mValAfter1d) &
                   (mInitializedBefore1d == mInitializedAfter1d)));
    st.addPrecondition(std::move(precPreserved));
    st.hasQuantifier = true;
  }
}

static Tensor loadTensor(
    State &st, mlir::Operation *op, const MemRef &memref,
    mlir::MemRefType memrefTy) {
  mlir::Type elemTy = memrefTy.getElementType();

  if (memrefTy.getLayout().isIdentity()) {
    // memref with identity map
    auto [arr, info] = st.m->loadArray(elemTy,
        memref.getBID(), memref.getOffset(), memref.get1DSize());
    st.wellDefined(op, info.checkRead());
    st.hasQuantifier |= info.checkRead().hasQuantifier();

    auto idx = Index::var("loadidx", VarType::BOUND);
    return Tensor::mkLambdaFrom1D(elemTy, memref.getDims(),
        std::move(idx), arr.select(idx), Expr::mkBool(true));

  } else {
    vector<Expr> idxs = Index::boundIndexVars(memrefTy.getRank());
    auto [val, info] = memref.getWithAccessInfo(idxs);

    st.wellDefined(op, Expr::mkForall(idxs,
        fitsInDims(idxs, memref.getDims()).implies(info.checkRead())));
    st.hasQuantifier = true;

    return Tensor::mkInitializedLambda(elemTy, memref.getDims(),
        std::move(idxs), std::move(val));
  }
}

template<class ValTy>
static vector<ValTy> getFromMixedOps(
    const State &st, const llvm::SmallVector<mlir::OpFoldResult> &mixedOps) {
  vector<ValTy> vec;
  for (auto s: mixedOps) {
    vec.push_back(s.is<mlir::Value>() ?
      st.regs.get<ValTy>(s.get<mlir::Value>()) :
      Index(s.get<mlir::Attribute>().dyn_cast<mlir::IntegerAttr>().getInt()));
  }
  return vec;
}

template <class ValTy>
static vector<Expr> getFromArrayI64(const mlir::ArrayRef<int64_t> &attr) {
  vector<Expr> vec;
  for (auto s : attr) {
    vec.push_back(ValTy(s));
  }
  return vec;
}

template<class ValTy>
static vector<Expr> getFromArrayAttr(const mlir::ArrayAttr &attr) {
  vector<Expr> vec;
  for (auto s: attr) {
    vec.push_back(ValTy(s.dyn_cast<mlir::IntegerAttr>().getInt()));
  }
  return vec;
}


template<class T>
static optional<Expr> encodeAffineExpr(
    mlir::AffineExpr ae, const vector<T> &dimvars, const vector<T> &symbolvars
) {
  switch (ae.getKind()) {
  case mlir::AffineExprKind::Add:
  case mlir::AffineExprKind::Mul: {
    auto aboe = ae.dyn_cast<mlir::AffineBinaryOpExpr>();
    auto lhs = encodeAffineExpr(aboe.getLHS(), dimvars, symbolvars);
    auto rhs = encodeAffineExpr(aboe.getRHS(), dimvars, symbolvars);
    if (!lhs || !rhs)
      return {};
    return (ae.getKind() == mlir::AffineExprKind::Add) ?
        *lhs + *rhs : *lhs * *rhs;
  }
  case mlir::AffineExprKind::DimId: {
    auto ade = mlir::dyn_cast<mlir::AffineDimExpr>(ae);
    auto id = ade.getPosition();
    assert(id < dimvars.size());
    return dimvars[id];
  }
  case mlir::AffineExprKind::SymbolId: {
    auto ade = mlir::dyn_cast<mlir::AffineSymbolExpr>(ae);
    auto id = ade.getPosition();
    assert(id < symbolvars.size());
    return symbolvars[id];
  }
  case mlir::AffineExprKind::Constant: {
    auto ac = mlir::dyn_cast<mlir::AffineConstantExpr>(ae);
    if (ac.getValue() < 0)
      return {};
    return Index(ac.getValue());
  }
  default:
    // Unsupported
    return {};
  }
}

static mlir::Type getElemTy(mlir::Value v) {
  return mlir::dyn_cast<mlir::ShapedType>(v.getType()).getElementType();
}


static optional<pair<Tensor, Tensor>>
broadcastTensors(State &st, mlir::Value arg0, mlir::Value arg1) {
  // reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
  auto ty0 = arg0.getType().cast<mlir::RankedTensorType>();
  auto ty1 = arg1.getType().cast<mlir::RankedTensorType>();
  auto t0 = st.regs.get<Tensor>(arg0);
  auto t1 = st.regs.get<Tensor>(arg1);
  auto ty0rank = max(ty0.getRank(), (int64_t)1);
  auto ty1rank = max(ty1.getRank(), (int64_t)1);
  auto getDimSize = [](mlir::RankedTensorType ty, int idx) -> int64_t {
    if (ty.getRank() == 0) {
      assert(idx == 0);
      return 1;
    }
    return ty.getDimSize(idx);
  };

  auto resRank = max(ty0rank, ty1rank);
  auto inVars0 = Index::boundIndexVars(resRank);
  auto inVars1 = Index::boundIndexVars(resRank);
  Expr izero = Index(0);

  vector<Expr> outVars0, outVars1;
  // The dimensions of broadcasted t0 and t1 are separately maintained (not
  // mixed). This is for a correct encoding of shape check (shape mismatch is
  // UB)
  vector<Expr> resDims0, resDims1;
  for (int64_t i = 0; i < min(ty0rank, ty1rank); i++) {
    int64_t idx0 = ty0rank - 1 - i;
    int64_t idx1 = ty1rank - 1 - i;
    int64_t inIdx = max(ty0rank, ty1rank) - 1 - i;

    auto d1 = getDimSize(ty0, idx0);
    auto d2 = getDimSize(ty1, idx1);

    bool dyn0 = d1 == mlir::ShapedType::kDynamic;
    bool dyn1 = d2 == mlir::ShapedType::kDynamic;
    if (dyn0 ^ dyn1)
      return nullopt;

    assert(d1 == 1 || d2 == 1 || d1 == d2);

    if (dyn0 && dyn1) {
      resDims0.insert(resDims0.begin(), t0.getDim(idx0));
      resDims1.insert(resDims1.begin(), t1.getDim(idx1));
    } else {
      resDims0.insert(resDims0.begin(), Index(max(d1,d2)));
      resDims1.insert(resDims1.begin(), Index(max(d1,d2)));
    }

    outVars0.insert(outVars0.begin(), d1 == 1 ? izero : inVars0[inIdx]);
    outVars1.insert(outVars1.begin(), d2 == 1 ? izero : inVars1[inIdx]);
  }

  if (ty0rank < ty1rank) {
    for (int64_t i = ty1rank - ty0rank - 1; i >= 0; --i) {
      auto d = t1.getDim(i);
      resDims0.insert(resDims0.begin(), d);
      resDims1.insert(resDims1.begin(), d);
      outVars1.insert(outVars1.begin(), inVars1[i]);
    }
  } else if (ty1rank < ty0rank) {
    for (int64_t i = ty0rank - ty1rank - 1; i >= 0; --i) {
      auto d = t0.getDim(i);
      resDims0.insert(resDims0.begin(), d);
      resDims1.insert(resDims1.begin(), d);
      outVars0.insert(outVars0.begin(), inVars0[i]);
    }
  }
  auto m0 = Tensor::mkInitializedLambda(t0.getElemType(),
      std::move(resDims0), std::move(inVars0), t0.get(outVars0));

  auto m1 = Tensor::mkInitializedLambda(t1.getElemType(),
      std::move(resDims1), std::move(inVars1), t1.get(outVars1));

  return {{m0, m1}};
}

template<class OpTy>
static void
encodeBinaryOp(State &st, OpTy op, mlir::Value arg0, mlir::Value arg1,
    function<Float(Float &&e1, Float &&e2)> f_float,
    function<Integer(Integer &&e1, Integer &&e2)> f_int) {

  mlir::Operation *opr = op.getOperation();

  if (mlir::isa<mlir::FloatType>(arg0.getType())) {
    auto a = st.regs.get<Float>(arg0);
    auto b = st.regs.get<Float>(arg1);
    st.regs.add(op, f_float(std::move(a), std::move(b)));

  } else if (mlir::isa<mlir::IntegerType>(arg0.getType())) {
    auto a = st.regs.get<Integer>(arg0);
    auto b = st.regs.get<Integer>(arg1);
    st.regs.add(op, f_int(std::move(a), std::move(b)));

  } else if (mlir::isa<mlir::IndexType>(arg0.getType())) {
    auto a = st.regs.get<Index>(arg0);
    auto b = st.regs.get<Index>(arg1);
    st.regs.add(op, Index::fromInteger(f_int(a.asInteger(), b.asInteger())));

  } else if (auto tty = mlir::dyn_cast<mlir::RankedTensorType>(arg0.getType())) {
    auto elemty = tty.getElementType();
    if (!elemty.isIntOrFloat())
      throw UnsupportedException(opr, "Unsupported element type");

    auto bts = broadcastTensors(st, arg0, arg1);
    if (!bts)
      throw UnsupportedException(opr, "Unsupported broadcast form");
    auto [a, b] = *bts;

    auto f = [&](Expr &&a, Expr &&b) -> Expr {
      if (elemty.isa<mlir::FloatType>()) {
        return f_float(Float(a, elemty), Float(b, elemty));
      } else if (elemty.isa<mlir::IntegerType>()) {
        return f_int(Integer(a), Integer(b));
      } else if (elemty.isa<mlir::IndexType>()) {
        return Index::fromInteger(f_int(Integer(a), Integer(b)));
      }
      throw UnsupportedException(opr, "Unknown value type");
    };
    st.regs.add(op, a.elementwiseBinOp(b, elemty, f));
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()),
        "shape match check");
    st.wellDefined(op, a.isFullyInitialized(), "op 0 initialized");
    st.wellDefined(op, b.isFullyInitialized(), "op 1 initialized");

  } else {
    throw UnsupportedException(opr, "Unsupported type");
  }
}

template<class OpTy>
static void
encodeUnaryOp(State &st, OpTy op, mlir::Value arg,
    function<Float(Float &&e)> f_float,
    function<Integer(Integer &&e)> f_int) {

  mlir::Operation *opr = op.getOperation();

  if (arg.getType().isa<mlir::FloatType>()) {
    st.regs.add(op, f_float(st.regs.get<Float>(arg)));

  } else if (arg.getType().isa<mlir::IntegerType>()) {
    st.regs.add(op, f_int(st.regs.get<Integer>(arg)));

  } else if (auto tty = arg.getType().dyn_cast<mlir::RankedTensorType>()) {
    auto elemty = tty.getElementType();
    if (!elemty.isIntOrFloat())
      throw UnsupportedException(opr, "Unsupported element type");

    auto a = st.regs.get<Tensor>(arg);

    auto f = [&](Expr &&a) -> Expr {
      if (elemty.isa<mlir::FloatType>()) {
        return f_float(Float(a, elemty));
      } else if (elemty.isa<mlir::IntegerType>()) {
        return f_int(Integer(a));
      }
      throw UnsupportedException(opr, "Unknown value type");
    };
    const auto resultElemTy =
        mlir::getElementTypeOrSelf(opr->getResult(0).getType());
    st.regs.add(op, a.elementwiseUnaryOp(resultElemTy, f));
    st.wellDefined(op, a.isFullyInitialized(), "the input is initialized");

  } else {
    throw UnsupportedException(opr, "Unsupported type");
  }
}


template<class T>
static void encodeOp(State &st, T op, bool encodeMemWriteOp);

// Encode the final state after executing this block.
static void encodeBlock(
    State &st, mlir::Block &block, bool printOps, bool encodeMemWriteOps,
    // checkBeforeEnc: return true if the op is to be ignored
    function<bool(mlir::Operation *, int)> checkBeforeEnc,
    function<void(mlir::Operation *)> callbackAfterEnc);


template<>
void encodeOp(State &st, mlir::arith::AddFOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::MulFOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.mul(b); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::DivFOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.div(b); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::NegFOp op, bool) {
  mlir::Value arg = op.getOperand();

  encodeUnaryOp(st, op, arg,
      [](auto &&a) { return a.neg(); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::SubFOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b.neg()); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::AddIOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1, {},
      [](auto &&a, auto &&b) { return (Expr)a + (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::arith::SubIOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1, {},
      [](auto &&a, auto &&b) { return (Expr)a - (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::arith::MulIOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1, {},
      [](auto &&a, auto &&b) { return (Expr)a * (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::arith::XOrIOp op, bool) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1, {},
      [](auto &&a, auto &&b) { return (Expr)a ^ (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::arith::SIToFPOp op, bool) {
  auto arg = st.regs.get<Integer>(op.getOperand());
  auto rty = op.getOut().getType();
  st.regs.add(op, Float::castFromSignedInt(arg, rty));
}

template<>
void encodeOp(State &st, mlir::arith::CmpFOp op, bool) {
  auto pred = op.getPredicate();
  auto op1Type = op.getOperand(0).getType();
  auto op2Type = op.getOperand(1).getType();

  if (op1Type.isa<mlir::TensorType>() && op2Type.isa<mlir::TensorType>()) {
    auto a = st.regs.get<Tensor>(op.getOperand(0));
    auto b = st.regs.get<Tensor>(op.getOperand(1));
    assert(a.getElemType() == b.getElemType());

    auto elemty = a.getElemType();
    auto resultElemTy = getElemTy(op.getResult());
    auto f = [&](Expr &&a, Expr &&b) -> Expr {
      if (elemty.isa<mlir::FloatType>()) {
        return Float(a, elemty).cmp(pred, Float(b, elemty));
      }
      throw UnsupportedException(op.getOperation(),
          "cmpf only accepts floating points");
    };
    st.regs.add(op, a.elementwiseBinOp(b, resultElemTy, f));
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()),
        "shape match check");
    st.wellDefined(op, a.isFullyInitialized(), "op 0 initialized");
    st.wellDefined(op, b.isFullyInitialized(), "op 1 initialized");

  } else if (op1Type.isa<mlir::FloatType>() &&
              op2Type.isa<mlir::FloatType>()) {
    auto a = st.regs.get<Float>(op.getOperand(0));
    auto b = st.regs.get<Float>(op.getOperand(1));
    st.regs.add(op, Integer(a.cmp(pred, b)));

  } else {
    throw UnsupportedException(op.getOperation(), "Unsupported cmpf operand");
  }
}

template<>
void encodeOp(State &st, mlir::arith::CmpIOp op, bool) {
  auto op1Type = op.getOperand(0).getType();
  auto op2Type = op.getOperand(1).getType();
  auto fn = [&](Expr &&a0, Expr &&b0) -> Expr {
    Expr a = Integer(a0), b = Integer(b0); // unlock arith ops
    switch (op.getPredicate()){
    case mlir::arith::CmpIPredicate::eq: return (a == b).toOneBitBV();
    case mlir::arith::CmpIPredicate::ne: return (a != b).toOneBitBV();
    case mlir::arith::CmpIPredicate::ule: return (a.ule(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::ult: return (a.ult(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::uge: return (a.uge(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::ugt: return (a.ugt(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::sle: return (a.sle(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::slt: return (a.slt(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::sge: return (a.sge(b)).toOneBitBV();
    case mlir::arith::CmpIPredicate::sgt: return (a.sgt(b)).toOneBitBV();
    }
    llvm_unreachable("Unknown cmpi predicate");
  };

  if (op1Type.isa<mlir::TensorType>() && op2Type.isa<mlir::TensorType>()) {
    auto a = st.regs.get<Tensor>(op.getOperand(0));
    auto b = st.regs.get<Tensor>(op.getOperand(1));
    assert(a.getElemType() == b.getElemType());

    auto resultElemTy = getElemTy(op.getResult());
    st.regs.add(op, a.elementwiseBinOp(b, resultElemTy, fn));
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()),
        "shape match check");
    st.wellDefined(op, a.isFullyInitialized(), "op 0 initialized");
    st.wellDefined(op, b.isFullyInitialized(), "op 1 initialized");

  } else if (op1Type.isa<mlir::IntegerType>() &&
              op2Type.isa<mlir::IntegerType>()) {
    auto a = st.regs.get<Integer>(op.getOperand(0));
    auto b = st.regs.get<Integer>(op.getOperand(1));
    st.regs.add(op, Integer(fn(a, b)));

  } else if (op1Type.isa<mlir::IndexType>() &&
              op2Type.isa<mlir::IndexType>()) {
    auto a = st.regs.get<Index>(op.getOperand(0));
    auto b = st.regs.get<Index>(op.getOperand(1));
    st.regs.add(op, Integer(fn(a, b)));

  } else {
    throw UnsupportedException(op.getOperation(), "Unsupported cmpi operand");
  }
}

template<>
void encodeOp(State &st, mlir::arith::ConstantIndexOp op, bool) {
  st.regs.add(op, Index(op.value()));
}

template<>
void encodeOp(State &st, mlir::arith::ConstantIntOp op, bool) {
  st.regs.add(op, Integer(op.value(), op.getType().getIntOrFloatBitWidth()));
}

template<>
void encodeOp(State &st, mlir::arith::ConstantFloatOp op, bool) {
  if (Float::sort(op.getType()) == nullopt)
    throw UnsupportedException(op.getOperation(), "unsupported constant type");

  auto fp = op.value();
  st.regs.add(op, Float::constant(fp, op.getType()));
}

template<>
void encodeOp(State &st, mlir::arith::ConstantOp op, bool) {
  auto attr = op.getValue();
  auto ty = op.getType();
  auto rty = ty.dyn_cast<mlir::RankedTensorType>();

  if (rty && attr.isa<mlir::ElementsAttr>()) {
    auto te = Tensor::fromElemsAttr(rty, attr.cast<mlir::ElementsAttr>());

    if (attr.isa<mlir::SparseElementsAttr>())
      // XXX: Don't know exactly when Z3 requires 'ALL' logic :/
      // This is empirical.
      st.hasConstArray = true;

    st.regs.add(op, std::move(te));

  } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    st.regs.add(op, attrToValueTy(intAttr));

  } else {
    throw UnsupportedException(op.getOperation(), "Unsupported constant");
  }
}

template <>
void encodeOp(State &st, mlir::arith::ExtFOp op, bool) {
  const auto srcElemType =
      mlir::getElementTypeOrSelf(op.getOperand().getType());
  FPPrecision src_prec = getPrecision(srcElemType);

  const auto tgtElemType = mlir::getElementTypeOrSelf(op.getType());
  FPPrecision tgt_prec = getPrecision(tgtElemType);

  if (src_prec == tgt_prec) {
    st.regs.add(op.getResult(), st.regs.get<Float>(op.getOperand()));
    return; // extending into identical type is a no-op
  } else if (src_prec > tgt_prec) {
    throw UnsupportedException(op.getOperation(),
                               "cannot ExtF into lower precision type!");
  }

  auto arg = op.getOperand();
  encodeUnaryOp(st, op, arg,
                [tgtElemType](auto &&a) { return a.extend(tgtElemType); }, {});
}

template <>
void encodeOp(State &st, mlir::arith::ExtSIOp op, bool) {
  const auto srcElemType =
      mlir::getElementTypeOrSelf(op.getOperand().getType());
  const auto src_bw = srcElemType.getIntOrFloatBitWidth();

  const auto tgtElemType = mlir::getElementTypeOrSelf(op.getType());
  const auto tgt_bw = tgtElemType.getIntOrFloatBitWidth();

  smart_assert(src_bw < tgt_bw, "Source's bitwidth must be smaller than "
                                "target's bitwidth, but got "
                                    << src_bw << " >= " << tgt_bw);

  auto arg = op.getOperand();
  auto amnt = tgt_bw - src_bw;
  encodeUnaryOp(st, op, arg, {},
                [amnt](Integer &&a) { return ((Expr)a).sext(amnt); });
}

template <>
void encodeOp(State &st, mlir::arith::ExtUIOp op, bool) {
  const auto srcElemType =
      mlir::getElementTypeOrSelf(op.getOperand().getType());
  const auto src_bw = srcElemType.getIntOrFloatBitWidth();

  const auto tgtElemType = mlir::getElementTypeOrSelf(op.getType());
  const auto tgt_bw = tgtElemType.getIntOrFloatBitWidth();

  smart_assert(src_bw < tgt_bw, "Source's bitwidth must be smaller than "
                                "target's bitwidth, but got "
                                    << src_bw << " >= " << tgt_bw);

  auto arg = op.getOperand();
  auto amnt = tgt_bw - src_bw;
  encodeUnaryOp(st, op, arg, {},
                [amnt](Integer &&a) { return ((Expr)a).zext(amnt); });
}

template <>
void encodeOp(State &st, mlir::arith::ShLIOp op, bool) {
  encodeShiftAmountBound(st, op.getOperation());

  auto arg = op.getOperand(0);
  auto amnt = op.getOperand(1);
  encodeBinaryOp(st, op, std::move(arg), std::move(amnt), {},
                 [](Integer &&a, Integer &&amnt) {
                   return static_cast<Expr>(a).shl(amnt);
                 });
}

template <>
void encodeOp(State &st, mlir::arith::ShRSIOp op, bool) {
  encodeShiftAmountBound(st, op.getOperation());

  auto arg = op.getOperand(0);
  auto amnt = op.getOperand(1);
  encodeBinaryOp(st, op, std::move(arg), std::move(amnt), {},
                 [](Integer &&a, Integer &&amnt) {
                   return static_cast<Expr>(a).ashr(amnt);
                 });
}

template <>
void encodeOp(State &st, mlir::arith::ShRUIOp op, bool) {
  encodeShiftAmountBound(st, op.getOperation());

  auto arg = op.getOperand(0);
  auto amnt = op.getOperand(1);
  encodeBinaryOp(st, op, std::move(arg), std::move(amnt), {},
                 [](Integer &&a, Integer &&amnt) {
                   return static_cast<Expr>(a).lshr(amnt);
                 });
}

template<>
void encodeOp(State &st, mlir::arith::TruncFOp op, bool) {
  const auto srcElemType =
      mlir::getElementTypeOrSelf(op.getOperand().getType());
  FPPrecision src_prec = getPrecision(srcElemType);

  const auto tgtElemType = mlir::getElementTypeOrSelf(op.getType());
  FPPrecision tgt_prec = getPrecision(tgtElemType);

  if (src_prec == tgt_prec) {
    st.regs.add(op.getResult(), st.regs.get<Float>(op.getOperand()));
    return; // truncating into identical type is a no-op
  } else if (src_prec < tgt_prec) {
    throw UnsupportedException(op.getOperation(),
                               "cannot TruncF into higher precision type!");
  }

  auto arg = op.getOperand();
  encodeUnaryOp(st, op, arg,
                [tgtElemType](auto &&a) { return a.truncate(tgtElemType); },
                {});
}

template <>
void encodeOp(State &st, mlir::arith::TruncIOp op, bool) {
  const auto srcElemType =
      mlir::getElementTypeOrSelf(op.getOperand().getType());
  const auto src_bw = srcElemType.getIntOrFloatBitWidth();

  const auto tgtElemType = mlir::getElementTypeOrSelf(op.getType());
  const auto tgt_bw = tgtElemType.getIntOrFloatBitWidth();

  smart_assert(src_bw > tgt_bw, "Source's bitwidth must be larger than "
                                "target's bitwidth, but got "
                                    << src_bw << " <= " << tgt_bw);

  auto arg = op.getOperand();
  auto amnt = src_bw - tgt_bw;
  encodeUnaryOp(st, op, arg, {},
                [amnt](Integer &&a) { return ((Expr)a).trunc(amnt); });
}

template<>
void encodeOp(State &st, mlir::linalg::IndexOp op, bool) {
  uint64_t i = op.getDim();
  assert(i < st.linalgGenericScopes.top().indVars.size());
  Expr idxvar = st.linalgGenericScopes.top().indVars[i];
  st.regs.add(op, Index(idxvar));
}

template<>
void encodeOp(State &st, mlir::math::AbsFOp op, bool) {
  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0, [](Float &&a) { return a.abs(); }, {});
}

template<>
void encodeOp(State &st, mlir::math::AbsIOp op, bool) {
  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0, {}, [](Integer &&a) { return ((Expr)a).abs(); });
}

template<>
void encodeOp(State &st, mlir::math::ExpOp op, bool) {
  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0, [](auto &&a) { return Float::exp(a); }, {});
}

template<>
void encodeOp(State &st, mlir::arith::IndexCastOp op, bool) {
  auto srcty = op.getOperand().getType();
  auto dstty = op.getType();

  if (srcty.isa<mlir::MemRefType>() || dstty.isa<mlir::MemRefType>())
    throw UnsupportedException(op.getOperation(),
        "index_cast of memref is not supported");

  if (auto src_tensorty = srcty.dyn_cast<mlir::TensorType>()) {
    auto dst_tensorty = dstty.dyn_cast<mlir::TensorType>();
    if (!dst_tensorty)
      throw UnsupportedException(op.getOperation(), "Unknown type");

    auto src = st.regs.get<Tensor>(op.getOperand());
    auto dst_elemty = dst_tensorty.getElementType();
    auto res = src.elementwiseUnaryOp(dst_elemty, [&](auto &&e) {
      return evalIndexCastOp(src_tensorty.getElementType(),
          dst_elemty, std::move(e));
    });
    st.regs.add(op, std::move(res));
    st.wellDefined(op, src.isFullyInitialized(), "the input is initialized");

  } else {
    auto src = st.regs.getExpr(op.getOperand());
    auto res = evalIndexCastOp(srcty, dstty, std::move(src));
    if (dstty.isIndex())
      st.regs.add(op, Index(res));
    else
      st.regs.add(op, Integer(res));
  }
}

template<>
void encodeOp(State &st, mlir::affine::AffineApplyOp op, bool) {
  auto m = op.getAffineMap();
  if (m.getNumResults() != 1)
    throw UnsupportedException(
        op.getOperation(), "num results is larger than one");

  auto dimOperands = op.getMapOperands().take_front(m.getNumDims());
  auto symbolOperands = op.getMapOperands().take_back(m.getNumSymbols());

  vector<Index> indices, symbols;
  for (auto arg: dimOperands)
    indices.push_back(st.regs.get<Index>(arg));
  for (auto symbol: symbolOperands)
    symbols.push_back(st.regs.get<Index>(symbol));

  auto res = encodeAffineExpr(m.getResult(0), indices, symbols);
  if (!res)
    throw UnsupportedException(op.getOperation(), "unsupported affine Expr");
  st.regs.add(op, Index(std::move(*res)));
}

template<>
void encodeOp(State &st, mlir::arith::SelectOp op, bool) {
  auto condTy = op.getCondition().getType();
  auto trueTy = op.getTrueValue().getType();
  auto falseTy = op.getFalseValue().getType();

  if (trueTy.isa<mlir::TensorType>() && falseTy.isa<mlir::TensorType>()) {
    if (trueTy.isa<mlir::UnrankedTensorType>() ||
        falseTy.isa<mlir::UnrankedTensorType>())
      throw UnsupportedException(op.getOperation(), "Unsupported operands");
    // It is guaranteed by mlir's verifier that condTy cannot be unranked
    assert(!condTy.isa<mlir::UnrankedTensorType>());

    auto trueValue = st.regs.get<Tensor>(op.getTrueValue());
    auto falseValue = st.regs.get<Tensor>(op.getFalseValue());
    // Encoding UB is necessary to support select of tensors -> linalg.generic
    Expr welldef = listsEqual(trueValue.getDims(), falseValue.getDims());
    function<Expr(const vector<Expr>&)> condFn =
        [&](const vector<Expr> &indices) -> Expr {
      return st.regs.get<Integer>(op.getCondition());
    };
    if (condTy.isa<mlir::RankedTensorType>()) {
      auto condValue = st.regs.get<Tensor>(op.getCondition());
      // Copy condValue
      condFn = [condValue](const vector<Expr> &indices) -> Expr {
        return condValue.get(indices);
      };
      welldef &= listsEqual(trueValue.getDims(), condValue.getDims());
    }

    auto result = Tensor::mkIte(condFn, trueValue, falseValue);
    st.regs.add(op, result);
    st.wellDefined(op, std::move(welldef), "shape match check");
    // Operands must be initialized.
    st.wellDefined(op, trueValue.isFullyInitialized(), "true op initialized");
    st.wellDefined(op, falseValue.isFullyInitialized(), "false op initialized");

  } else if (trueTy.isa<mlir::MemRefType>() &&
             falseTy.isa<mlir::MemRefType>()) {
    if (trueTy.isa<mlir::UnrankedMemRefType>() ||
        falseTy.isa<mlir::UnrankedMemRefType>())
      throw UnsupportedException(op.getOperation(), "Unsupported operands");
    if (!condTy.isa<mlir::IntegerType>())
      throw UnsupportedException(
          op.getOperation(),
          "For MemRef operands, i1 typed condition is supported only");

    auto trueValue = st.regs.get<MemRef>(op.getTrueValue());
    auto falseValue = st.regs.get<MemRef>(op.getFalseValue());
    auto condValue = st.regs.get<Integer>(op.getCondition());
    auto result = MemRef::mkIte(condValue, trueValue, falseValue);

    st.regs.add(op, result);
    // Constrain the dimensions to be equivalent, otherwise the layout info
    // becomes bogus.
    st.wellDefined(op, listsEqual(trueValue.getDims(), falseValue.getDims()),
        "shape match check");

  } else {
    assert(trueTy.isIntOrFloat() || trueTy.isIndex());

    auto trueValue = st.regs.getExpr(op.getTrueValue());
    auto falseValue = st.regs.getExpr(op.getFalseValue());
    auto condValue = st.regs.get<Integer>(op.getCondition());
    auto isTrue = (Expr)condValue == Integer::boolTrue();
    st.regs.add(op, Expr::mkIte(isTrue, trueValue, falseValue), op.getType());
  }
}

template<>
void encodeOp(State &st, mlir::func::CallOp op, bool) {
  if (op.getNumResults() != 1) {
    throw UnsupportedException(
      op.getOperation(),
      "Invalid number of return values");
  }

  if (!dimsReferenceIdxMap) {
    // parse user-specified dims
    dimsReferenceIdxMap = map<string, int64_t, std::less<>>();

    for (const auto &dimRefArg: arg_use_arg_dims) {
      // parse function name and argument index
        const auto atPos = dimRefArg.find("@");
        auto fnName = dimRefArg.substr(0, atPos);
        const auto argIndex = stoll(dimRefArg.substr(atPos + 1));
        const auto [_, success] = dimsReferenceIdxMap->insert(
          {std::move(fnName), argIndex});
        smart_assert(success, "Dims reference argument for '" << fnName
                      << "' is specified more than once");
    }
  }

  const auto callee = op.getCallee();
  if (!getDeclaredFunction(callee)) {
    vector<mlir::Type> domain(op.getOperandTypes().begin(),
                              op.getOperandTypes().end());
    auto range = op.getResultTypes().front();
    try {
      declareFunction(std::move(domain), std::move(range), std::move(callee),
                      getDimsReferenceIdx(callee));
    } catch (UnsupportedException e) {
      throw UnsupportedException(op.getOperation(), e.getReason());
    }
  }

  vector<ValueTy> operands;
  operands.reserve(op.getNumOperands());
  for (const auto& operand: op.getOperands()) {
    operands.push_back(st.regs.findOrCrash(operand));
  }

  auto calleeUF = *getDeclaredFunction(callee);
  st.regs.add(op.getResult(0), calleeUF.apply(operands));
}

template<>
void encodeOp(State &st, mlir::func::ReturnOp op, bool) {
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    st.retValues.push_back(st.regs.findOrCrash(op.getOperand(i)));
}

template<>
void encodeOp(State &st, mlir::shape::ShapeOfOp op, bool) {
  if (!op.getType().isa<mlir::TensorType>())
    throw UnsupportedException(op.getOperation(), "unsupported type");

  auto tensor = op.getOperand();
  if (!tensor.getType().isa<mlir::TensorType>())
    throw UnsupportedException(op.getOperation(), "unsupported type");

  auto tt = st.regs.get<Tensor>(tensor);
  auto elemTy = getElemTy(op.getResult());
  st.regs.add(op, Tensor(elemTy, tt.getDims()));
  // Note: tensor's elements do not need to be initialized.
}

template<>
void encodeOp(State &st, mlir::tosa::AbsOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto t = st.regs.get<Tensor>(op.getOperand());
  auto ety = dty.getElementType();
  st.regs.add(op.getResult(), t.elementwiseUnaryOp(ety, [&](auto &&e) {
    return Float(e, ety).abs();
  }));
  st.wellDefined(op, t.isFullyInitialized(), "the input is initialized");
}

template<>
void encodeOp(State &st, mlir::tosa::ConcatOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  uint64_t axis = op.getAxis();
  auto t = st.regs.get<Tensor>(op.getOperand(0));
  st.wellDefined(op, t.isFullyInitialized(), "op 0 initialized");

  for (auto tensor: op.getOperands().drop_front()) {
    auto t2 = st.regs.get<Tensor>(tensor);
    st.wellDefined(op, t2.isFullyInitialized(), "following ops initialized");
    for (unsigned i = 0; i < t2.getRank(); ++i) {
      if (i != axis)
        st.wellDefined(op, t.getDim(i) == t2.getDim(i),
            "shape match check");
    }

    t = t.concat(t2, axis);
  }

  st.regs.add(op.getResult(), t);
}

template<>
void encodeOp(State &st, mlir::tosa::ClampOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");
  auto elemTy = dty.getElementType();

  auto input = st.regs.get<Tensor>(op.getInput());

  auto unaryFn = [elemTy, &op](smt::Expr &&elem0) -> smt::Expr {
    // In TOSA 0.23:
    // apply_clip := apply_min(apply_max(value, minval), maxval)
    // apply_max: (a >= b) ? a : b, NaN if either a or b is NaN (TOSA 0.23, Sec.1.9)
    // apply_min: (a < b)  ? a : b, NaN if either a or b is NaN (TOSA 0.23, Sec.1.9) 

    if (elemTy.isa<mlir::IntegerType>()) {
      Integer minval(op.getMinInt(), elemTy.getIntOrFloatBitWidth());
      Integer maxval(op.getMaxInt(), elemTy.getIntOrFloatBitWidth());
      Integer elem(elem0);
      elem = Expr::mkIte(((Expr)elem).sge(minval), elem, minval);
      elem = Expr::mkIte(((Expr)elem).slt(maxval), elem, maxval);

      return elem;
    } else {
      Float minval = Float::constant(op.getMinFp(), elemTy);
      Float maxval = Float::constant(op.getMaxFp(), elemTy);
      Float elem(elem0, elemTy);
      auto olt = mlir::arith::CmpFPredicate::OLT;
      auto one = Expr::mkBV(1, 1);
      // To make clamp return NaN on NaN inputs, the first cmp predicate is
      // reversed.
      Expr e1 = Float(Expr::mkIte(
          (Expr)elem.cmp(olt, minval) == one, minval, elem), elemTy);
      Expr e2 = Float(Expr::mkIte(
          (Expr)maxval.cmp(olt, elem) == one, maxval, e1), elemTy);

      return e2;
    }
  };

  auto output = input.elementwiseUnaryOp(elemTy, unaryFn);
  
  st.wellDefined(op, input.isFullyInitialized(), "the input is initialized");
  st.regs.add(op, output);
}

template<>
void encodeOp(State &st, mlir::tosa::ConstOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");
  auto eattr = op.getValue().dyn_cast<mlir::ElementsAttr>();
  if (!eattr)
    throw UnsupportedException(op.getOperation(), "Unsupported attribute");

  st.regs.add(op, Tensor::fromElemsAttr(dty, eattr));
  if (eattr.isa<mlir::SparseElementsAttr>())
    st.hasConstArray = true;
}

template<>
void encodeOp(State &st, mlir::tosa::ReverseOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto t = st.regs.get<Tensor>(op.getInput1());
  auto axis = op.getAxis();

  st.regs.add(op, t.reverse(axis));
  st.wellDefined(op, t.isFullyInitialized(), "the input is initialized");
}

template<>
void encodeOp(State &st, mlir::tosa::TileOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto t = st.regs.get<Tensor>(op.getInput1());
  vector<unsigned> repeat;
  for (long val: op.getMultiples()) {
    if (val < 0)
      throw UnsupportedException(op.getOperation(), "Negative multiple");
    repeat.push_back(val);
  }

  st.regs.add(op, t.tile(repeat));
  st.wellDefined(op, t.isFullyInitialized(), "the input is initialized");
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseAndOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.getInput1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.getInput2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type"); 

  mlir::Value i1 = op.getInput1();
  mlir::Value i2 = op.getInput2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a & (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseNotOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.getInput1()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");

  mlir::Value i1 = op.getInput1();

  encodeUnaryOp(st, op, i1,
      nullptr,
      [](auto &&a) { return ~(Expr)a; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseOrOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.getInput1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.getInput2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type"); 

  mlir::Value i1 = op.getInput1();
  mlir::Value i2 = op.getInput2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a | (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseXorOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.getInput1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.getInput2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");

  mlir::Value i1 = op.getInput1();
  mlir::Value i2 = op.getInput2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a ^ (Expr)b; });
}

static Tensor getPaddedTensor2D(mlir::Type elemTy, 
                                Tensor input,
                                mlir::ArrayRef<int64_t> padding) {
  if (!llvm::all_of(padding, [](int64_t a) {
      return a == 0; })) {

    // pad = [top, bottom, left, right], filled with zero
    vector<Expr> pad = getFromArrayI64<Index>(padding);
    assert(pad.size() == 4);

    // input rank should be 4
    vector<Expr> padInd = Index::boundIndexVars(input.getRank());
    vector<Expr> srcDims = input.getDims();

    vector<Expr> srcInd = {padInd[0], padInd[1] - pad[0],
                              padInd[2] - pad[2], padInd[3]};

    vector<Expr> padDims = {srcDims[0], srcDims[1] + pad[0] + pad[1],
                              srcDims[2] + pad[2] + pad[3], srcDims[3]};

    auto cond = padInd[1].uge(pad[0]) & padInd[1].ult(pad[0] + srcDims[1]) &
                  padInd[2].uge(pad[2]) & padInd[2].ult(pad[2] + srcDims[2]);

    // TOSA pad operands fill padded area as +0.0.
    // If --use-neg-zero is given, use -0.0 instead.
    auto zero = arg_use_neg_zero.getValue() ?
        *getIdentity(elemTy) : *getZero(elemTy);
    Expr padVal = Expr::mkIte(cond, input.get(srcInd), zero);

    return Tensor::mkInitializedLambda(
                    elemTy, std::move(padDims), std::move(padInd), padVal);

  } else {
    return input;
  }
}

static Tensor addBias2D(mlir::Type elemTy,
                        vector<Expr> dims,
                        Tensor acc, Tensor bias) {
  vector<Expr> ind = Index::boundIndexVars(4);
  auto tf = Float(acc.get(ind), elemTy);
  auto biasf = Float(bias.get({ind[3]}), elemTy);
  return Tensor::mkInitializedLambda(
            elemTy, std::move(dims), std::move(ind),
            tf.add(biasf)
          );
}

template<>
void encodeOp(State &st, mlir::tosa::DepthwiseConv2DOp op, bool) {
  // input's dim sizes = [N, H, W, C]
  auto input = st.regs.get<Tensor>(op.getInput());
  // weight's dim sizes = [H, W, C, M]
  auto weight = st.regs.get<Tensor>(op.getWeight());
  // bias: a 1-dim array whose size is C * M
  auto bias = st.regs.get<Tensor>(op.getBias());
  // strides = [strides_y, strides_x]
  vector<Expr> strides = getFromArrayI64<Index>(op.getStride());
  // dilations = [dilations_y, dilations_x]
  vector<Expr> dilations = getFromArrayI64<Index>(op.getDilation());

  auto elemTy = getElemTy(op.getResult());
  if (!elemTy.isa<mlir::FloatType>())
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto C = weight.getDim(2);
  auto M = weight.getDim(3);

  // Check whether C is identical
  st.wellDefined(op, input.getDim(3) == C, "input and weight's shapes check");
  // Check whether C * M is identical
  st.wellDefined(op, bias.getDim(0) == (C * M),
      "bias and weight's shapes check");

  auto paddedTensor = getPaddedTensor2D(elemTy, input, op.getPad());

  auto output = paddedTensor.depthwiseConv2D(weight, strides, dilations, bias);

  st.wellDefined(op, input.isFullyInitialized(), "input is initialized");
  st.wellDefined(op, weight.isFullyInitialized(), "weight is initialized");
  st.wellDefined(op, bias.isFullyInitialized(), "bias is initialized");

  st.regs.add(op, output);

}

template<>
void encodeOp(State &st, mlir::tosa::Conv2DOp op, bool) {
  // input's dim sizes = [N, H, W, C]
  auto input = st.regs.get<Tensor>(op.getInput());
  // weight's dim sizes = [F, H, W, C]
  auto weight = st.regs.get<Tensor>(op.getWeight());
  // bias: a 1-dim array whose size is F
  auto bias = st.regs.get<Tensor>(op.getBias());
  // strides = [strides_y, strides_x]
  vector<Expr> strides = getFromArrayI64<Index>(op.getStride());
  // dilations = [dilations_y, dilations_x]
  vector<Expr> dilations = getFromArrayI64<Index>(op.getDilation());

  // Check whether C is identical
  st.wellDefined(op, input.getDim(3) == weight.getDim(3),
      "input and weight's shapes check");
  // Check whether F is identical
  st.wellDefined(op, weight.getDim(0) == bias.getDim(0),
      "bias and weight's shapes check");

  assert(strides.size() == 2 && dilations.size() == 2);

  auto elemTy = getElemTy(op.getResult());
  if (!elemTy.isa<mlir::FloatType>())
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto paddedTensor = getPaddedTensor2D(elemTy, input, op.getPad());


  auto acc = paddedTensor.conv(weight,
                      strides, dilations, ShapedValue::ConvLayout::NHWC_FHWC);

  vector<Expr> outDims = acc.getDims();
  auto output = addBias2D(elemTy, outDims, acc, bias);

  st.wellDefined(op, input.isFullyInitialized(), "input is initialized");
  st.wellDefined(op, weight.isFullyInitialized(), "weight is initialized");
  st.wellDefined(op, bias.isFullyInitialized(), "bias is initialized");

  st.regs.add(op, output);

}

template<>
void encodeOp(State &st, mlir::tosa::TransposeOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  mlir::Value i = op.getInput1();
  mlir::Value p = op.getPerms();

  auto ity = i.getType().dyn_cast<mlir::RankedTensorType>();
  auto pty = p.getType().dyn_cast<mlir::RankedTensorType>();
  if(!getElemTy(p).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");

  smart_assert(pty.getRank() == 1, "Perms' rank must be 1, but got " << pty);
  smart_assert(pty.getDimSize(0) == ity.getRank(),
      "Perm's dim size must be equal to Input1's rank, but got "
      << pty.getDimSize(0) << " != " << ity.getRank());

  auto input = st.regs.get<Tensor>(i);
  auto perms = st.regs.get<Tensor>(p);

  vector<Expr> indVars = Index::boundIndexVars(input.getRank());
  vector<Expr> dims, outVars;
  vector<uint64_t> idxs;
  bool isIdentity = true;

  for (unsigned i = 0; i < input.getRank(); i++) {
    uint64_t v;
    // We expect simplify() to succeed since perms is a small Tensor
    if(!perms.get({Index(i)}).simplify().isUInt(v))
      throw UnsupportedException(op.getOperation(),
          "Unsupported perms element type");
    idxs.push_back(v);
    dims.push_back(input.getDim(v));
    isIdentity &= v == i;
  }

  st.wellDefined(op, input.isFullyInitialized(), "input is initialized");
  if (isIdentity) {
    st.regs.add(op, input);
    return;
  }

  // check the validity of perms
  for (unsigned i = 0; i < input.getRank(); i++) {
    int count = 0;
    for (unsigned j = 0; j < input.getRank(); j++) {
      assert(idxs[j] >= 0 && idxs[j] < input.getRank());
      if (idxs[j] == i)
        count++;
    }
    assert(count == 1);
  }

  for (unsigned i = 0; i < input.getRank(); i++) {
    bool pushed = false;
    for(unsigned idx = 0; idx < input.getRank(); idx++) {
      if(idxs[idx] == i) {
        outVars.push_back(indVars[idx]);
        pushed = true;
        break;
      }
    }
    assert(pushed && "transpose's perms is not permutation!");
  }

  auto output = input.get(outVars);

  st.regs.add(op, Tensor::mkLambda(input.getElemType(),
                    std::move(dims), std::move(indVars), output, Expr::mkBool(true)));

}

template<>
void encodeOp(State &st, mlir::tosa::GatherOp op, bool) {
  // values, output - 3D dimension, indices - 2D dimension.
  // These were checked by default MLIR verifier

  // input's dim sizes = [N, K, C]
  auto values = st.regs.get<Tensor>(op.getValues());
  // indices's dim sizes = [N, W]
  auto indices = st.regs.get<Tensor>(op.getIndices());
  // output tensor dim = [N, W, C]
  // output[n][w][c] = values[n][indices[n][w]][c]
  vector<Expr> outputDims =
      {values.getDim(0), indices.getDim(1), values.getDim(2)};
  vector<Expr> indVars = Index::boundIndexVars(outputDims.size());
  auto inBounds = fitsInDims(indVars, outputDims);

  auto idx0 = indices.get({indVars[0], indVars[1]});
  auto idxInBounds = indices.isInBounds({indVars[0], indVars[1]});
  Index idx(idx0); // unlock ops

  auto outputValue = values.get({indVars[0], idx, indVars[2]});
  auto inputInBounds = values.isInBounds({indVars[0], idx, indVars[2]});
  auto isInitialized = values.isInitialized({indVars[0], idx, indVars[2]});

  // Touched elements must be in bounds & have been initialized.
  st.wellDefined(op, Expr::mkForall(indVars,
      inBounds.implies(std::move(idxInBounds) & std::move(inputInBounds))),
      "indices and input's indices are inbounds");
  st.wellDefined(op, Expr::mkForall(indVars,
      inBounds.implies(std::move(isInitialized))),
      "chosen inputs are initialized");
  st.wellDefined(op, indices.isFullyInitialized(),
      "indices tensor is initialized");

  st.regs.add(op, Tensor::mkInitializedLambda(
      values.getElemType(), std::move(outputDims), std::move(indVars),
      std::move(outputValue)));
}

template<>
void encodeOp(State &st, mlir::tosa::AvgPool2dOp op, bool) {
  auto input = st.regs.get<Tensor>(op.getInput());
  auto kernelDims = getFromArrayI64<Index>(op.getKernel());
  auto paddings = getFromArrayI64<Index>(op.getPad());
  auto strides = getFromArrayI64<Index>(op.getStride());

  if (!input.getElemType().isa<mlir::FloatType>()) {
    throw UnsupportedException(op.getOperation(),
          "Unsupported element type");
  }

  for (unsigned i = 0; i < input.getRank(); i ++) {
    uint64_t v;
    if(!paddings[i].isUInt(v))
      throw UnsupportedException(op.getOperation(),
          "Unsupported pad element type");
    if (v > 0)
      throw UnsupportedException(op.getOperation(),
          "Zero-padded pooling is supported only.");
  }

  // TODO: The current modeling ignores the acc_type attribute.

  auto result = input.avgPool(kernelDims, strides);
  st.regs.add(op.getResult(), std::move(result));
  st.wellDefined(op, input.isFullyInitialized(), "source tensor initialized");
}

template<>
void encodeOp(State &st, mlir::tosa::MaxPool2dOp op, bool) {
  auto input = st.regs.get<Tensor>(op.getInput());
  auto kernelDims = getFromArrayI64<Index>(op.getKernel());
  auto paddings = getFromArrayI64<Index>(op.getPad());
  auto strides = getFromArrayI64<Index>(op.getStride());

  if (!input.getElemType().isa<mlir::FloatType>()) {
    throw UnsupportedException(op.getOperation(),
          "Unsupported element type");
  }

  for (unsigned i = 0; i < input.getRank(); i ++) {
    uint64_t v;
    if(!paddings[i].isUInt(v))
      throw UnsupportedException(op.getOperation(),
          "Unsupported pad element type");
    if (v > 0)
      throw UnsupportedException(op.getOperation(),
          "Zero-padded pooling is supported only.");
  }

  auto result = input.maxPool(kernelDims, strides);
  st.regs.add(op.getResult(), std::move(result));
  st.wellDefined(op, input.isFullyInitialized(), "source tensor initialized");
}

template<>
void encodeOp(State &st, mlir::tensor::ExtractOp op, bool) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.

  auto t = st.regs.get<Tensor>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.getIndices())
    indices.emplace_back(st.regs.get<Index>(idx0));
  if (indices.empty())
    // Deal with the zero-rank tensor case
    indices.push_back(Index(0));

  auto elem = t.get(indices);
  if (auto v = fromExpr(std::move(elem), op.getType()))
    st.regs.add(op, std::move(*v));
  else
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  st.wellDefined(op, t.isInBounds(indices), "inbounds");
  st.wellDefined(op, t.isInitialized(indices), "initialized");
}

static Expr getValueOrNegZero(State &st, mlir::Value v) {
  if (arg_use_neg_zero) {
    auto fty = v.getType().dyn_cast<mlir::FloatType>();
    auto iop = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(
        *v.getDefiningOp());
    if (fty && iop && iop.value().isPosZero()) {
      verbose("getValueOrNegZero") << "Using negative zero instead\n";
      return aop::getFpEncoding(fty).zero(true);
    }
  }
  return st.regs.getExpr(v);
}

static void encodeParallelLoopBodyAndOutputs(
    State &newst, mlir::Block &block, const mlir::AffineMap &outputMap,
    optional<vector<Tensor>> &tvec_res, Expr &welldef,
    // (yielded value, ind var) -> newly mapped value
    optional<function<Expr(const Expr&, const vector<Expr>&)>>
        outputValMap = nullopt) {
  // Encode the loop body
  // TODO: deal with merging memories
  vector<mlir::Value> yieldedValues;

  encodeBlock(newst, block, /*print ops*/false, /*encode mem writes*/false,
      [&yieldedValues](mlir::Operation *op, int opindex) {
        if (auto op2 = mlir::dyn_cast<mlir::linalg::YieldOp>(op)) {
          assert(op2.getNumOperands() > 0);
          for (unsigned i = 0; i < op2.getNumOperands(); i++) {
            yieldedValues.push_back(op2.getOperand(i));
          }
          return true;
        } else if (auto op2 = mlir::dyn_cast<mlir::tensor::YieldOp>(op)) {
          yieldedValues.push_back(op2.getOperand());
          return true;
        }
        return false;
      },
      [&welldef, &newst](mlir::Operation *op) {
        welldef &= newst.isOpWellDefined(op);
      });

  auto &scope = newst.linalgGenericScopes.top();
  auto outputIndVars = doMap(scope.indVars, outputMap);
  auto tensorSz = addOne(doMap(scope.indVarUpperBounds, outputMap));

  tvec_res.emplace();
  for (unsigned i = 0; i < yieldedValues.size(); i++) {
    // If arg_use_neg_zero is set, convert pos zero to neg zero.
    Expr resExpr = getValueOrNegZero(newst, yieldedValues[i]);
    if (outputValMap)
      resExpr = (*outputValMap)(resExpr, outputIndVars);

    auto il = Tensor::mkInitializedLambda(yieldedValues[i].getType(),
        vector(tensorSz), vector(outputIndVars), resExpr);
    tvec_res->push_back(il);
  }
}

template<class T>
static void encodeConv(State &st, T op, ShapedValue::ConvLayout clayout) {
  vector<Expr> strides, dilations;
  // TODO: The result may not fit in Index::BITS
  for (auto s: op.getStrides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.getDilations())
    dilations.push_back(Index(d.getSExtValue()));

  if (op.hasPureTensorSemantics()) {
    auto t_input = st.regs.get<Tensor>(op.image());
    auto t_filter = st.regs.get<Tensor>(op.filter());
    auto output = st.regs.get<Tensor>(op.getOutputs()[0]);

    auto t_res = t_input
      .conv(t_filter, strides, dilations, clayout, output);
    st.regs.add(op.getResult(0), std::move(t_res));
    st.wellDefined(op, t_input.isFullyInitialized(), "input is initialized");
    st.wellDefined(op, t_filter.isFullyInitialized(), "filter is initialized");
    st.wellDefined(op, output.isFullyInitialized(), "output is initialized");
  } else {
    auto outputTy = op.getOutputs()[0].getType().template cast<mlir::MemRefType>();
    auto elemTy = outputTy.getElementType();
    auto input = st.regs.get<MemRef>(op.image());
    auto filter = st.regs.get<MemRef>(op.filter());
    MemRef output = st.regs.get<MemRef>(op.getOutputs()[0]);

    if (!output.isIdentityMap())
      throw UnsupportedException(op.getOperation(),
          "The output MemRef should have identity layout.");

    auto getInitValue = [&](vector<Expr> &indices) -> optional<Expr> {
      return output.get(indices);
    };
    auto [indices, expr] = input.ShapedValue::conv(
        filter, strides, dilations, clayout, std::move(getInitValue));

    // Check that the output memref is storing initialized values.
    st.wellDefined(op, output.isFullyInitialized(), "output is initialized");

    // we splat results into 1D memory layout
    auto idx = Index::var("outputIdx", VarType::BOUND);
    auto outputIndices = output.getLayout().getInverseIndices(idx);
    auto outputExpr = expr.substitute(indices, outputIndices);
    auto outputTensor = Tensor::mkInitializedLambda(elemTy,
        {output.get1DSize()}, {idx}, outputExpr);

    // store the result to the output reference
    storeTensorTo(st, op, std::move(outputTensor), output, outputTy, true);

    // Input & filter read check
    st.wellDefined(op,
        input.getLiveness() & input.isInBounds(),
        "input read safety (without init check)");
    st.wellDefined(op, input.isFullyInitialized(), "input is initialized");
    st.wellDefined(op,
        filter.getLiveness() & filter.isInBounds(),
        "filter read safety (without init check)");
    st.wellDefined(op,
        filter.isFullyInitialized(), "filter is initialized");
    // No alias checks between output and input/filter
    st.wellDefined(op, output.noalias(input) & output.noalias(filter),
        "output does not alias input and filter");
  }
}

template<> void
encodeOp(State &st, mlir::linalg::DepthwiseConv2DNhwcHwcmOp op,
         bool encodeMemWriteOp) {
  if (!op.hasPureTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  vector<Expr> strides, dilations;

  for (auto s: op.getStrides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.getDilations())
    dilations.push_back(Index(d.getSExtValue()));

  if (op.hasPureTensorSemantics()) {
    auto t_input = st.regs.get<Tensor>(op.image());
    auto t_filter = st.regs.get<Tensor>(op.filter());
    auto t_output = st.regs.get<Tensor>(op.getOutputs()[0]);

    auto t_res = t_input.depthwiseConv2D(t_filter, strides, dilations,
        /* bias */ nullopt, /* output */ t_output);
    st.regs.add(op.getResult(0), std::move(t_res));
    st.wellDefined(op, t_input.isFullyInitialized(), "input is initialized");
    st.wellDefined(op, t_filter.isFullyInitialized(), "filter is initialized");
    st.wellDefined(op, t_output.isFullyInitialized(), "output is initialized");
  } else {
    auto mi = st.regs.get<MemRef>(op.image());
    auto mf = st.regs.get<MemRef>(op.filter());
    auto mo = st.regs.get<MemRef>(op.getOutputs()[0]);
    auto iTy = op.image().getType().cast<mlir::MemRefType>();
    auto fTy = op.filter().getType().cast<mlir::MemRefType>();
    auto oTy = op.getOutputs()[0].getType().cast<mlir::MemRefType>();
    Tensor t_input = loadTensor(st, op, mi, iTy);
    Tensor t_filter = loadTensor(st, op, mf, fTy);
    Tensor t_output = loadTensor(st, op, mo, oTy);
    auto t_res = t_input.depthwiseConv2D(t_filter, strides, dilations,
        /* bias */ nullopt, /* output */ t_output);
    storeTensorTo(st, op, std::move(t_res), mo, oTy, true);
    st.wellDefined(op, mo.noalias(mi) & mo.noalias(mf),
        "output does not alias inputs");
  }
}

template<> void
encodeOp(State &st, mlir::linalg::Conv2DNchwFchwOp op, bool encodeMemWriteOp) {
  if (!op.hasPureTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  encodeConv(st, op, ShapedValue::ConvLayout::NCHW_FCHW);
}

template<> void
encodeOp(State &st, mlir::linalg::Conv2DNhwcHwcfOp op, bool encodeMemWriteOp) {
  if (!op.hasPureTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  encodeConv(st, op, ShapedValue::ConvLayout::NHWC_HWCF);
}

template<>
void encodeOp(State &st, mlir::tensor::CollapseShapeOp op, bool) {
  Tensor t = st.regs.get<Tensor>(op.getOperand());
  mlir::RankedTensorType resTy = op.getResultType();

  auto reassocExprs = op.getReassociationIndices();
  assert(reassocExprs.size() == (size_t)resTy.getRank());

  vector<Expr> newDims;
  if (reassocExprs.size() == 0) {
    newDims.push_back(Index(1));
  } else {
    // If the collapsed size does not match op.getResultType(), it is UB.
    for (unsigned i = 0; i < reassocExprs.size(); ++i) {
      Expr size = Index::one();
      for (auto &idx: reassocExprs[i])
        size = size * t.getDim(idx);

      if (resTy.getDimSize(i) != mlir::ShapedType::kDynamic)
        st.wellDefined(op, size == resTy.getDimSize(i),
            "size check");
      newDims.push_back(std::move(size));
    }
  }

  st.wellDefined(op, t.get1DSize() == smt::get1DSize(newDims),
      "size check");
  st.regs.add(op.getResult(), t.reshape(newDims));
  // Note: tensor_collapse_shape does not look into elements, so initialization
  // check is not necessary.
}

template<>
void encodeOp(State &st, mlir::tensor::ExpandShapeOp op, bool) {
  Tensor t = st.regs.get<Tensor>(op.getSrc());

  // The fresh variables created by ShapedValue::getDims will be ignored
  // by the for loop below.
  auto newdims = ShapedValue::getDims(op.getResultType(), true);
  auto indices = op.getReassociationIndices();

  unsigned i = 0;
  for (unsigned srci = 0; srci < indices.size(); ++srci) {
    auto &ids = indices[srci];
    auto orgdim = (Expr)t.getDim(srci);

    // Allow one '?' only.
    int unknown_dim = -1;
    int64_t const_size = 1;
    for (auto id: ids) {
      if (op.getResultType().getDimSize(id) == mlir::ShapedType::kDynamic) {
        if (unknown_dim != -1)
          throw UnsupportedException(op.getOperation(),
              "it has more than one unknown dimension size in one group");
        unknown_dim = i;
      } else {
        const_size *= op.getResultType().getDimSize(id);
      }
      ++i;
    }

    if (unknown_dim == -1)
      // Nothing to do; it is already well-defined
      continue;

    if (Index::BITS < 64 && (size_t)const_size >= (1ull << Index::BITS))
      throw UnsupportedException(op.getOperation(),
          "tensor size is too large");

    // If the original size isn't divisible, raise UB
    st.wellDefined(op, orgdim.urem(const_size) == 0);
    newdims[unknown_dim] = orgdim.udiv(const_size);
  }

  st.regs.add(op.getResult(), t.reshape(newdims));
  // Note: tensor_expand_shape does not look into elements, so initialization
  // check is not necessary.
}

template<>
void encodeOp(State &st, mlir::linalg::MatmulOp op, bool encodeMemWriteOp) {
  if (!(op.hasPureTensorSemantics() || op.hasPureBufferSemantics()))
    throw UnsupportedException(op.getOperation(),
        "tensor/buffer semantics is supported only");
  if (op.hasPureBufferSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
    throw UnsupportedException(op.getOperation(),
        "unsupported form");

  if (getElemTy(op.getOperand(0)) != getElemTy(op.getOperand(1)) ||
      getElemTy(op.getOperand(0)) != getElemTy(op.getOperand(2)))
    throw UnsupportedException(op.getOperation(),
        "unsupported types");

  if (op.hasPureTensorSemantics()) {
    Tensor a = st.regs.get<Tensor>(op.getOperand(0));
    Tensor b = st.regs.get<Tensor>(op.getOperand(1));
    Tensor c = st.regs.get<Tensor>(op.getOperand(2));
    Tensor result = a.matmul(b, /*transposed*/false, c);

    st.wellDefined(op, a.isFullyInitialized(), "op 0 initialized");
    st.wellDefined(op, b.isFullyInitialized(), "op 1 initialized");
    st.wellDefined(op, c.isFullyInitialized(), "op 2 initialized");
    st.regs.add(op.getResult(0), Tensor(result));
    st.hasQuantifier |= a.isFullyInitialized().hasQuantifier();
    st.hasQuantifier |= b.isFullyInitialized().hasQuantifier();
    st.hasQuantifier |= c.isFullyInitialized().hasQuantifier();
  } else { // Buffer semantics
    auto ma = st.regs.get<MemRef>(op.getOperand(0));
    auto mb = st.regs.get<MemRef>(op.getOperand(1));
    auto mc = st.regs.get<MemRef>(op.getOperand(2));
    auto aTy = op.getOperand(0).getType().cast<mlir::MemRefType>();
    auto bTy = op.getOperand(1).getType().cast<mlir::MemRefType>();
    auto cTy = op.getOperand(2).getType().cast<mlir::MemRefType>();
    Tensor a = loadTensor(st, op, ma, aTy);
    Tensor b = loadTensor(st, op, mb, bTy);
    Tensor c = loadTensor(st, op, mc, cTy);
    Tensor result = a.matmul(b, /*transposed*/false, c);

    storeTensorTo(st, op, std::move(result), mc, cTy, true);
    // No alias checks between input & output
    st.wellDefined(op, mc.noalias(ma) & mc.noalias(mb),
        "output does not alias inputs");
  }
}

template<>
void encodeOp(State &st, mlir::tensor::PadOp op, bool) {
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto &region = op.getRegion();
  if (!region.hasOneBlock())
    throw UnsupportedException(op.getOperation(), "Unsupported region");
  auto &blk = *region.getBlocks().begin();

  vector<Index> padSizeLow = getFromMixedOps<Index>(st, op.getMixedLowPad());
  vector<Index> padSizeHigh = getFromMixedOps<Index>(st, op.getMixedHighPad());

  auto sourceTensor = st.regs.get<Tensor>(op.getSource());
  auto newTensorSize =
      vecAdd(vecAdd(sourceTensor.getDimsAsIndices(), padSizeLow), padSizeHigh);

  State newst = st;
  auto loopUpperBound = vecAddElem(newTensorSize, Index(-1));
  newst.linalgGenericScopes.push(State::LinalgGenericScope{
      std::move(loopUpperBound)});
  auto &indVars = newst.linalgGenericScopes.top().indVars;
  for (int i = 0; i < blk.getNumArguments(); ++i) {
    Expr idxvar = indVars[i];
    newst.regs.add(blk.getArgument(i), Index(idxvar));
  }

  auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
      retty.getRank(), op.getContext());
  auto paddingOrSource = [&](const Expr &pad, const vector<Expr> &indvars) {
    Expr isSource = Expr::mkBool(true);
    assert(indvars.size() == padSizeLow.size() &&
           indvars.size() == padSizeHigh.size());
    vector<Expr> sourceIndices;
    for (unsigned i = 0; i < indvars.size(); ++i) {
      Expr l = padSizeLow[i];
      Expr h = padSizeLow[i] + sourceTensor.getDim(i);
      isSource &= l.ule(indvars[i]) & indvars[i].ult(h);
      sourceIndices.push_back(indvars[i] - l);
    }
    return Expr::mkIte(isSource, sourceTensor.get(sourceIndices), pad);
  };

  optional<vector<Tensor>> tvec_res;

  Expr welldef = Expr::mkBool(true);
  encodeParallelLoopBodyAndOutputs(newst, blk, identityMap, tvec_res, welldef,
      paddingOrSource);

  // pad_tensor has one output.
  welldef = Expr::mkForall(indVars,
      tvec_res->front().isInBounds(indVars).implies(welldef));

  newst.linalgGenericScopes.pop();

  // If pad_tensor's output dimension sizes are known, the padding sizes must
  // match
  if (retty.hasStaticShape()) {
    for (unsigned i = 0; i < retty.getRank(); ++i) {
      st.wellDefined(op,
          tvec_res->front().getDim(i) == retty.getDimSize(i));
    }
  }

  st.regs.add(op.getResult(), std::move(tvec_res->front()));
  st.wellDefined(op, std::move(welldef), "loop body");
  st.wellDefined(op, sourceTensor.isFullyInitialized(),
      "source tensor initialized");
}

template<class T>
static void encodeLinalgPooling(State &st, T op) {
  mlir::DenseIntElementsAttr strideAttr = op.getStrides();
  mlir::DenseIntElementsAttr dilationAttr = op.getDilations();

  if (!strideAttr.isSplat() || !dilationAttr.isSplat())
    throw UnsupportedException(op.getOperation(),
        "Splat elements are supported only");

  auto stride = strideAttr.getSplatValue<mlir::Attribute>()
      .dyn_cast<mlir::IntegerAttr>().getInt();
  auto dilation = dilationAttr.getSplatValue<mlir::Attribute>()
      .dyn_cast<mlir::IntegerAttr>().getInt();

  if (dilation != 1)
    throw UnsupportedException(op.getOperation(),
        "dilation=1 is supported only");

  if (op.hasPureTensorSemantics()) {
    mlir::Type elemTy = getElemTy(op.getResult(0));
    if (!elemTy.isa<mlir::FloatType>())
      throw UnsupportedException(op.getOperation(), "Unsupported type");

    vector<Expr> kernelDims = st.regs.get<Tensor>(op.getInputs()[1]).getDims();
    vector<Expr> strides = {Index(stride), Index(stride)};
    auto input = st.regs.get<Tensor>(op.getInputs()[0]);
    auto output = st.regs.get<Tensor>(op.getOutputs()[0]);
    bool isMaxPool = std::is_same<T, mlir::linalg::PoolingNhwcMaxOp>::value;
    auto result = isMaxPool ? input.maxPool(kernelDims, strides, output)
        : input.sumPool(kernelDims, strides, output);

    st.regs.add(op.getResult(0), std::move(result));
    st.wellDefined(op, input.isFullyInitialized(), "input tensor initialized");
    st.wellDefined(op, output.isFullyInitialized(), "output tensor initialized");
  } else {
    mlir::Type elemTy = op.getOutputs()[0].getType()
                          .template cast<mlir::MemRefType>()
                          .getElementType();
    if (!elemTy.isa<mlir::FloatType>())
      throw UnsupportedException(op.getOperation(), "Unsupported type");

    vector<Expr> kernelDims = st.regs.get<MemRef>(op.getInputs()[1]).getDims();
    vector<Expr> strides = {Index(stride), Index(stride)};
    MemRef minput = st.regs.get<MemRef>(op.getInputs()[0]);
    MemRef moutput = st.regs.get<MemRef>(op.getOutputs()[0]);
    auto inputTy = op.getInputs()[0].getType().template cast<mlir::MemRefType>();
    auto outputTy = op.getOutputs()[0].getType().template cast<mlir::MemRefType>();

    Tensor input = loadTensor(st, op, minput, inputTy);
    Tensor output = loadTensor(st, op, moutput, outputTy);

    bool isMaxPool = std::is_same<T, mlir::linalg::PoolingNhwcMaxOp>::value;
    auto result = isMaxPool ? input.maxPool(kernelDims, strides, output)
        : input.sumPool(kernelDims, strides, output);

    storeTensorTo(st, op, std::move(result), moutput, outputTy, true);
    st.wellDefined(op, moutput.noalias(minput),
        "input and output buffers must not alias");
  }
}

template<>
void encodeOp(State &st, mlir::linalg::PoolingNhwcSumOp op, bool) {
  encodeLinalgPooling(st, op);
}

template<>
void encodeOp(State &st, mlir::linalg::PoolingNhwcMaxOp op, bool) {
  encodeLinalgPooling(st, op);
}

static pair<Expr, Expr> encodeDimOp(
    State &st, vector<Expr> &&dims, mlir::Value index) {
  auto idx = st.regs.get<Index>(index);

  auto res = dims[0];
  for (unsigned i = 1; i < dims.size(); ++i)
    res = Expr::mkIte((Expr)idx == i, dims[i], res);

  return {std::move(res), ((Expr)idx).ult(dims.size())};
}

template<>
void encodeOp(State &st, mlir::tensor::DimOp op, bool) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<Tensor>(op.getSource()).getDims(), op.getIndex());
  st.regs.add(op, Index(res));
  st.wellDefined(op, std::move(wf));
  // DimOp does not look into elements, so initialization check is not necessary
}

template <> void encodeOp(State &st, mlir::tensor::EmptyOp op, bool) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::RankedTensorType>();
  if (!ty || !Tensor::isTypeSupported(ty))
    throw UnsupportedException(op.getOperation(), "Unsupported tensor type");

  vector<Expr> sizes;
  if (ty.getRank() == 0) {
    sizes.push_back(Index(1));
  } else {
    for (unsigned i = 0; i < ty.getRank(); ++i) {
      if (ty.isDynamicDim(i))
        sizes.push_back(st.regs.get<Index>(op.getDynamicSize(i)));
      else
        sizes.push_back(Index(ty.getDimSize(i)));
    }
  }

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  st.regs.add(res, Tensor::var(ty.getElementType(),
                               ("init_tensor#") + to_string(new_var_idx++),
                               sizes, false));
}

template<>
void encodeOp(State &st, mlir::tensor::CastOp op, bool) {
  auto tty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto t = st.regs.get<Tensor>(op.getOperand());
  for (size_t i = 0; i < tty.getRank(); ++i) {
    if (tty.isDynamicDim(i))
      continue;
    st.wellDefined(op, (Expr)t.getDim(i) == tty.getDimSize(i));
  }
  st.regs.add(op, std::move(t));
  // Initialization check is not necessary
}

template<>
void encodeOp(State &st, mlir::tensor::InsertOp op, bool) {
  auto val = st.regs.getExpr(op.getScalar());
  auto dest = st.regs.get<Tensor>(op.getDest());

  vector<Expr> indices;
  for (auto idx0: op.getIndices())
    indices.emplace_back(st.regs.get<Index>(idx0));
  if (indices.empty())
    indices.push_back(Index(0));

  auto [tensor, inbounds] = dest.insert(val, indices);
  st.regs.add(op, std::move(tensor));
  st.wellDefined(op, std::move(inbounds), "inbounds");
}

template<>
void encodeOp(State &st, mlir::tensor::FromElementsOp op, bool) {
  vector<Expr> elems;
  vector<uint64_t> dims;
  auto resTy = op.getType().dyn_cast<mlir::RankedTensorType>();
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    elems.push_back(st.regs.getExpr(op.getOperand(i)));

  if (resTy.getRank() == 0)
    dims.push_back(1);
  for (unsigned i = 0; i < resTy.getRank(); ++i)
    dims.push_back(resTy.getDimSize(i));

  auto elemTy = op.getType().getElementType();
  st.regs.add(op.getResult(), Tensor(elemTy, std::move(elems), dims));
}

template<>
void encodeOp(State &st, mlir::tensor::GenerateOp op, bool) {
  auto exts = op.getDynamicExtents();
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");
  if (op.getBody().getBlocks().size() != 1)
    throw UnsupportedException(op.getOperation(), "Unsupported form");
  auto &blk = op.getBody().getBlocks().front();

  vector<Index> upperbound;
  {
    int j = 0;
    for (int i = 0; i < retty.getRank(); ++i) {
      auto d = retty.getDimSize(i);
      if (d == mlir::ShapedType::kDynamic) {
        auto newd = exts[j++];
        upperbound.push_back(st.regs.get<Index>(newd).ofs(-1));
      } else {
        upperbound.push_back(Index(d).ofs(-1));
      }
    }
  }

  optional<vector<Tensor>> tvec_res;
  Expr welldef = Expr::mkBool(true);
  {
    State newst = st;
    newst.linalgGenericScopes.push(State::LinalgGenericScope{std::move(upperbound)});
    for (int i = 0; i < blk.getNumArguments(); ++i) {
      Expr idxvar = newst.linalgGenericScopes.top().indVars[i];
      newst.regs.add(blk.getArgument(i), Index(idxvar));
    }

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        retty.getRank(), op.getContext());

    encodeParallelLoopBodyAndOutputs(newst, blk, identityMap,
        tvec_res, welldef);

    auto &indVars = newst.linalgGenericScopes.top().indVars;

    // linalg::generate has one result
    welldef = Expr::mkForall(indVars,
        tvec_res->front().isInBounds(indVars).implies(welldef));

    newst.linalgGenericScopes.pop();
  }

  // linalg::generate has one result
  st.regs.add(op.getResult(), std::move(tvec_res->front()));
  st.wellDefined(op, std::move(welldef), "loop body");
}

template<>
void encodeOp(State &st, mlir::tensor::ExtractSliceOp op, bool) {
  vector<Index> offsets, sizes, strides;
  const auto src = st.regs.get<Tensor>(op.getOperand(0));
  auto srcType = op.getOperand(0).getType().dyn_cast<mlir::ShapedType>();
  auto res = op.getResult();
  auto resType = res.getType().dyn_cast<mlir::ShapedType>();

  strides = getFromMixedOps<Index>(st, op.getMixedStrides());
  sizes = getFromMixedOps<Index>(st, op.getMixedSizes());
  offsets = getFromMixedOps<Index>(st, op.getMixedOffsets());

  if (offsets.size() != sizes.size() || sizes.size() != strides.size() ||
      strides.size() != (size_t)srcType.getRank())
    throw UnsupportedException(op.getOperation(), "Unsupported form");

  vector<Expr> dims;

  // push output dimensions to dims
  unsigned j = 0;
  for (unsigned i = 0; i < resType.getRank(); i++) {
    if (!resType.isDynamicDim(i) && resType.getDimSize(i) == 1) {
      dims.push_back(Index(1));
      continue;
    }

    // Find the new size.
    while (true) {
      assert(j < sizes.size());
      auto elem = op.getMixedSizes()[j];
      if (!elem.is<mlir::Attribute>())
        // Matched.
        break;
      auto szval = elem.get<mlir::Attribute>().dyn_cast<mlir::IntegerAttr>();
      if (szval.getInt() != 1)
        break;
      // Ignore the zero size, and look into the next one.
      j++;
    }
    
    // check if output tensor matches size or size is unknown
    dims.push_back(sizes[j]);
    j++;
  }

  // Add out-of-bounds check
  for (unsigned i = 0; i < sizes.size(); ++i) {
    auto dim = src.getDim(i);
    Expr ofs = offsets[i], size = sizes[i];
    Expr cond = ofs.ult(dim) & size.ule(dim) & (ofs + sizes[i]).ule(dim);
    verbose("ExtractSliceOp out-of-bounds check") << cond << "\n";
    st.wellDefined(op, std::move(cond), "inbounds");
  }

  vector<Expr> inIdxs, outIdxs;
  // indices that is going to be read from the output tensor
  inIdxs = Index::boundIndexVars(resType.getRank());

  // map the output tensor indices to source tensor indices
  unsigned idx = 0;
  for (unsigned i = 0; i < srcType.getRank(); i++) {
    uint64_t v;
    bool isDimSizeOne = idx >= resType.getRank() ||
        ((((Expr)sizes[i]).isUInt(v) && v == 1) && resType.getDimSize(idx) != v);

    if (isDimSizeOne) {
      outIdxs.push_back((Expr)offsets[i]);
    } else {
      // sizes operand should match with target tensor dimension
      st.wellDefined(op,  (sizes[i] == dims[idx]));
      outIdxs.push_back((Expr)((inIdxs[idx++] * strides[i])) + offsets[i]);
    }
  }
  st.wellDefined(op, src.isFullyInitialized(), "source is initialized");
  st.regs.add(res,
      Tensor::mkInitializedLambda(src.getElemType(), std::move(dims), std::move(inIdxs),
                       src.get(outIdxs)));
}

template<>
void encodeOp(State &st, mlir::tensor::InsertSliceOp op, bool) {
  
  vector<Index> offsets, sizes, strides;
  auto src = st.regs.get<Tensor>(op.getOperand(0));
  auto tgt = st.regs.get<Tensor>(op.getOperand(1));
  auto res = op.getResult();
  auto rank = op.getOperand(0).getType().dyn_cast<mlir::ShapedType>().getRank();
  if (rank != op.getOperand(1).getType().dyn_cast<mlir::ShapedType>().getRank()
      || rank != res.getType().dyn_cast<mlir::ShapedType>().getRank())
    throw UnsupportedException(op.getOperation(),
        "Unsupported tensor types of src and dest: their ranks do not match");

  strides = getFromMixedOps<Index>(st, op.getMixedStrides());
  sizes = getFromMixedOps<Index>(st, op.getMixedSizes());
  offsets = getFromMixedOps<Index>(st, op.getMixedOffsets());

  assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
         strides.size() == rank);

  vector<Expr> indVars = Index::boundIndexVars(rank);
  vector<Expr> dims = tgt.getDims();
  vector<Expr> srcIdxs;

    // Add out-of-bounds check
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = tgt.getDim(i);
    Expr ofs = offsets[i], size = sizes[i];
    Expr cond = ofs.ult(dim) & size.ule(dim) & (ofs + sizes[i]).ule(dim);
    verbose("InsertSliceOp out-of-bounds check") << cond << "\n";
    st.wellDefined(op, std::move(cond));
  }

  Expr cond = Expr::mkBool(true);

  for (unsigned i = 0; i < rank; i++) {
    srcIdxs.push_back((indVars[i] - offsets[i]).udiv(strides[i]));
    cond &= ((indVars[i] - offsets[i]).urem(strides[i])).isZero() &
            (indVars[i] - offsets[i]).ult(sizes[i] * strides[i]);
    // sizes operand should match with source tensor dimension
    st.wellDefined(op,  (sizes[i] == src.getDim(i)));
  }

  // Picking the value from src1 must not be out of bounds.
  auto srcelem = src.get(srcIdxs);
  auto srcwb   = src.isInBounds(srcIdxs);
  auto tgtelem = tgt.get(indVars);
  auto tgtwb   = tgt.isInBounds(indVars);
  Expr output = Expr::mkIte(cond, std::move(srcelem), std::move(tgtelem));

  // If tgt[indVars] is inbounds and the src[indVars] is to be chosen,
  // src[indVars] must be inbounds as well.
  st.wellDefined(op,
      Expr::mkForall(indVars, (tgtwb & cond).implies(srcwb)));
  // Since we are copying tgt into a new SSA register, tgt must be
  // initialized as well.
  st.wellDefined(op,
      Expr::mkForall(indVars, (tgtwb & !cond).implies(
        tgt.isInitialized(indVars))), "tgt initialized");

  st.regs.add(res, Tensor::mkInitializedLambda(
      src.getElemType(), std::move(dims), std::move(indVars), output));
  st.wellDefined(op, src.isFullyInitialized(), "src initialized");
}

template<>
void encodeOp(State &st, mlir::tosa::AddOp op, bool) {
  auto optys = op.getOperandTypes();
  if (!optys[0].isa<mlir::RankedTensorType>() ||
      !optys[1].isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand types");

  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b); },
      [](auto &&a, auto &&b) { return (Expr)a + (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::SubOp op, bool) {
  auto optys = op.getOperandTypes();
  if (!optys[0].isa<mlir::RankedTensorType>() ||
      !optys[1].isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand types");

  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b.neg()); },
      [](auto &&a, auto &&b) { return (Expr)a - (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::MulOp op, bool) {
  auto optys = op.getOperandTypes();
  if (!optys[0].isa<mlir::RankedTensorType>() ||
      !optys[1].isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(),
        "Unsupported operand types");

  if (op.getShift() != 0)
    throw UnsupportedException(op.getOperation(),
        "Mul with shift is unsupported");

  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.mul(b); },
      [](auto &&a, auto &&b) { return (Expr)a * (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::NegateOp op, bool) {
  auto opty = op.getOperand().getType();
  if (!opty.isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");
  else if (op.getQuantizationInfo())
    throw UnsupportedException(op.getOperation(), "Quantization is unsupported");

  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0,
      [](auto &&a) { return a.neg(); },
      [](auto &&a) { return Expr::mkBV(0, a.bitwidth()) - (Expr)a; });
}

template<>
void encodeOp(State &st, mlir::tosa::ReciprocalOp op, bool) {
  auto opty = op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  if (!opty || !opty.getElementType().isa<mlir::FloatType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  auto elemty = opty.getElementType().cast<mlir::FloatType>();
  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0,
      [elemty](auto &&a) { return Float::one(elemty).div(a); },
      {});
}

template<>
void encodeOp(State &st, mlir::tosa::ExpOp op, bool) {
  auto opty = op.getOperand().getType();
  if (!opty.isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0,
      [](auto &&a) { return Float::exp(a); }, {});
}

template<>
void encodeOp(State &st, mlir::tosa::FullyConnectedOp op, bool) {
  auto input = op.getInput();   // [N, IC]
  auto weight = op.getWeight(); // [OC, IC]
  auto bias = op.getBias();     // [OC]

  if (!input.getType().isa<mlir::RankedTensorType>() ||
      !weight.getType().isa<mlir::RankedTensorType>() ||
      !bias.getType().isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  auto inputTensor = st.regs.get<Tensor>(input);
  auto weightTensor = st.regs.get<Tensor>(weight);
  auto biasTensor = st.regs.get<Tensor>(bias);

  if ((inputTensor.getElemType() != weightTensor.getElemType()) ||
      (weightTensor.getElemType() != biasTensor.getElemType()))
      throw UnsupportedException(op.getOperation(),
        "Operands of different types are unsupported");

  st.wellDefined(op, inputTensor.getDim(1) == weightTensor.getDim(1));
  st.wellDefined(op, weightTensor.getDim(0) == biasTensor.getDim(0));
  st.wellDefined(op, inputTensor.isFullyInitialized(), "input initialized");
  st.wellDefined(op, weightTensor.isFullyInitialized(), "weight initialized");
  st.wellDefined(op, biasTensor.isFullyInitialized(), "bias initialized");

  auto mul = inputTensor.matmul(weightTensor, /*transposed*/true);

  // Output: [N, OC]
  auto idxVars = Index::boundIndexVars(2);
  vector<Expr> sizes = {inputTensor.getDim(0), weightTensor.getDim(0)};
  auto biasBroadcasted = biasTensor.affine(idxVars, {idxVars[1]}, std::move(sizes));

  auto elemTy = getElemTy(input);
  auto res = mul.elementwiseBinOp(biasBroadcasted, elemTy,
      [elemTy](Expr &&a, Expr &&b) -> Expr {
    if (elemTy.isa<mlir::FloatType>()) {
      return Float(a, elemTy).add(Float(b, elemTy));
    } else if (elemTy.isa<mlir::IntegerType>()) {
      return a + b;
    }
    throw UnsupportedException("Unsupported type");
  });

  st.regs.add(op, res);
}

template<>
void encodeOp(State &st, mlir::tosa::ReduceSumOp op, bool) {
  auto input = op.getInput();
  auto inputTy = input.getType().dyn_cast<mlir::RankedTensorType>();
  if (!inputTy)
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  auto t = st.regs.get<Tensor>(input);
  uint64_t axis = op.getAxis();

  st.wellDefined(op.getOperation(), t.isFullyInitialized(),
      "input initialized");
  st.regs.add(op, t.sum(axis));
}

template<>
void encodeOp(State &st, mlir::tosa::ReshapeOp op, bool) {
  auto t = st.regs.get<Tensor>(op.getOperand());
  auto attrs = op.getNewShape();
  vector<Expr> newDims;
  mlir::Operation *oper = op.getOperation();

  for (auto ia: attrs) {
    if (ia == -1)
      throw UnsupportedException(oper, "Dynamic shape is unsupported");
    newDims.push_back(Index(ia));
  }
  st.wellDefined(oper, t.get1DSize() == smt::get1DSize(newDims));
  st.regs.add(op.getResult(), t.reshape(newDims));
  // Reshape does not look into tensor's elements, so init check is not
  // necessary.
}

static MemRef createNewLocalBlk(
    Memory *m, vector<Expr> &&dims, mlir::MemRefType memrefTy, bool writable,
    bool createdByAlloc = false) {
  if (!MemRef::isTypeSupported(memrefTy))
    throw UnsupportedException("unsupported element type");

  auto layout = MemRef::getLayout(memrefTy, dims);
  // Add a new local block
  auto bid = m->addLocalBlock(smt::get1DSize(dims),
      memrefTy.getElementType(), Expr::mkBool(writable), createdByAlloc);
  // Create MemRef which points to the newly created block
  auto memref =
      MemRef(m, memrefTy.getElementType(), bid, Index::zero(), dims,
          std::move(layout), /*is not a view reference*/Expr::mkBool(false));

  return {std::move(memref)};
}

template<class T>
static void encodeAllocLikeOp(State &st, T op) {
  auto memrefTy = op.getType().template cast<mlir::MemRefType>();
  if (!memrefTy.getLayout().isIdentity())
    throw UnsupportedException(op.getOperation(),
        "unsupported memref type for alloc: it has a non-identity layout map");

  auto dsizes = op.getDynamicSizes();
  vector<Expr> dszExprs;
  for (const auto &sz: dsizes) {
    dszExprs.push_back(st.regs.get<Index>(sz));
  }
  auto dims = ShapedValue::getDims(memrefTy, false, std::move(dszExprs));

  auto memref = createNewLocalBlk(st.m.get(), std::move(dims), memrefTy, true,
      std::is_same_v<T, mlir::memref::AllocOp>);
  st.regs.add(op, std::move(memref));
}

template<>
void encodeOp(State &st, mlir::memref::AllocOp op, bool) {
  encodeAllocLikeOp(st, op);
}

template<>
void encodeOp(State &st, mlir::memref::AllocaOp op, bool) {
  encodeAllocLikeOp(st, op);
}

template<>
void encodeOp(State &st, mlir::memref::DimOp op, bool) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<MemRef>(op.getSource()).getDims(), op.getIndex());
  st.regs.add(op, Index(res));
  st.wellDefined(op, std::move(wf));
}

template<>
void encodeOp(State &st, mlir::memref::LoadOp op, bool) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.getIndices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto [val, info] = m.getWithAccessInfo(indices);
  if (auto vt = fromExpr(std::move(val), op.getType())) {
    st.regs.add(op, std::move(*vt));
    st.wellDefined(op, info.checkRead());
  } else
    throw UnsupportedException(op.getOperation(), "unsupported type");
}

template<>
void encodeOp(State &st, mlir::memref::GetGlobalOp op, bool encodeMemWriteOp) {
  auto name = op.getName().str();
  auto bid = Expr::mkBV(st.m->getBidForGlobalVar(name), st.m->getBIDBits());
  auto type = op.getType();
  assert(type.getLayout().isIdentity() &&
      "don't know how to deal with get_global with non-identity layout");
  auto dims = ShapedValue::getDims(type, /*unknown sz is crash*/false);
  MemRef::Layout identityLayout(dims);

  MemRef newref(st.m.get(), type.getElementType(), bid, Index(0), dims,
      identityLayout, Expr::mkBool(false));
  st.regs.add(op, std::move(newref));
}

template<>
void encodeOp(State &st, mlir::memref::StoreOp op, bool encodeMemWriteOp) {
  if (!encodeMemWriteOp)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(1));
  vector<Expr> indices;
  for (auto idx0: op.getIndices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto value = op.getOperand(0);
  if (convertPrimitiveTypeToSort(value.getType()) == nullopt)
    throw UnsupportedException(op.getOperation(), "unsupported type");

  auto valExpr = st.regs.getExpr(value);
  auto success = m.store(valExpr, indices);
  st.wellDefined(op, success.checkWrite());
}

template<>
void encodeOp(State &st, mlir::memref::SubViewOp op, bool) {
  vector<Expr> sizes, offsets, strides;

  for (unsigned i = 0; i < op.getSourceType().getRank(); i++) {
#define ADD(vec, ee) { \
  vec.push_back(op.isDynamic ## ee(i) ? \
      st.regs.get<Index>(op.getDynamic ## ee(i)) : \
      Index(op.getStatic ## ee(i))); \
}
    ADD(offsets, Offset);
    ADD(sizes, Size);
    ADD(strides, Stride);
#undef ADD
  }
  auto src = st.regs.get<MemRef>(op.getSource());
  int rankDiff = op.getSourceType().getRank() - op.getType().getRank();
  assert(rankDiff >= 0); // only reducing rank is allowed

  // This reduction logic mainly from MLIR SubViewOp verify function.
  // See 'Dialect/MemRef/IR/MemRefOps.cpp'.
  auto expectedType = mlir::memref::SubViewOp::inferResultType(
      op.getSourceType(), op.getStaticOffsets(), op.getStaticSizes(), op.getStaticStrides());

  auto originalShapedType = expectedType.cast<mlir::ShapedType>();
  auto candidateReducedShapedType = op.getType().cast<mlir::ShapedType>();
  auto optionalUnusedDimsMask = mlir::computeRankReductionMask(
    originalShapedType.getShape(),
    candidateReducedShapedType.getShape()
  );

  if (!optionalUnusedDimsMask.has_value())
    throw UnsupportedException(op.getOperation(),
        "Subview result size mismatch");

  auto unusedDims = optionalUnusedDimsMask.value();
  auto memref = src.subview(offsets, sizes, strides, unusedDims, rankDiff);
  st.regs.add(op.getResult(), std::move(memref));
}

template<>
void encodeOp(State &st, mlir::bufferization::ToMemrefOp op,
    bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto tensor = st.regs.get<Tensor>(op.getOperand());
  auto memrefTy = op.getMemref().getType().cast<mlir::MemRefType>();
  auto dims = tensor.getDims();

  // Create a read-only block.
  auto memref = createNewLocalBlk(st.m.get(), std::move(dims), memrefTy, false);
  storeTensorTo(st, op.getOperation(), std::move(tensor), memref, memrefTy, false);
  st.regs.add(op.getMemref(), std::move(memref));
}

template<>
void encodeOp(State &st, mlir::bufferization::CloneOp op, bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto src = st.regs.get<MemRef>(op.getOperand());
  auto srcTy = op.getOperand().getType().cast<mlir::MemRefType>();
  auto dims = src.getDims();

  auto tensor = loadTensor(st, op, src, srcTy);

  // Create a read-only block.
  auto memref = createNewLocalBlk(st.m.get(), std::move(dims), srcTy, false);
  storeTensorTo(st, op.getOperation(), std::move(tensor), memref, srcTy, false);
  // Src is not writable as well.
  st.m->setWritable(srcTy.getElementType(), src.getBID(), false);
  st.regs.add(op, std::move(memref));
}

template<>
void encodeOp(State &st, mlir::bufferization::ToTensorOp op,
    bool encodeMemWrite) {
  auto memref = op.getOperand();
  auto memrefTy = memref.getType().cast<mlir::MemRefType>();
  auto m = st.regs.get<MemRef>(memref);
  // Mark the MemBlock pointed by the memref as read-only.
  auto &memory = *(st.m);
  memory.setWritable(memrefTy.getElementType(), m.getBID(), false);

  auto tensor = loadTensor(st, op, m, memrefTy);

  st.regs.add(op.getResult(), tensor);
}

template<>
void encodeOp(State &st, mlir::memref::DeallocOp op, bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto src = st.regs.get<MemRef>(op.getOperand());
  auto srcTy = op.getOperand().getType().cast<mlir::MemRefType>();

  // A dead block cannot be deallocated.
  st.wellDefined(op, src.getLiveness(), "liveness");

  // According to the MLIR specification doc:
  // The dealloc operation should not be called on memrefs which alias an
  // alloc’d memref (e.g. memrefs returned by view operations).
  st.wellDefined(op, !src.isViewReference(), "not a view reference");

  // The deallocating object must have been created by memref.alloc()
  st.wellDefined(op, src.isCreatedByAlloc(), "must be created by memref.alloc");

  // Unlike free(), we don't need to check offset == 0 because MemRef tracks
  // the pointer to the data buffer as allocated, referred to as
  // "allocated pointer". This is useful for deallocating the memref.
  // See: https://mlir.llvm.org/docs/TargetLLVMIR/ , Ranked MemRef Types sec.

  st.m->setLivenessToFalse(srcTy.getElementType(), src.getBID());
}

template<>
void encodeOp(State &st, mlir::memref::ExpandShapeOp op, bool encodeMemWrite) {
  auto srcType = op.getSrc().getType().cast<mlir::MemRefType>();
  auto resType = op.getResult().getType().cast<mlir::MemRefType>();

  if (!srcType.getLayout().isIdentity() || !resType.getLayout().isIdentity())
    throw UnsupportedException(op.getOperation(),
      "We do not support non-identity layout memref");

  MemRef m = st.regs.get<MemRef>(op.getSrc());
  // The fresh variables created by ShapedValue::getDims will be ignored
  // by the for loop below.
  auto newdims = ShapedValue::getDims(op.getResultType(), true);
  auto indices = op.getReassociationIndices();

  unsigned i = 0;
  for (unsigned srci = 0; srci < indices.size(); ++srci) {
    auto &ids = indices[srci];
    auto orgdim = (Expr)m.getDim(srci);

    // Allow one '?' only.
    int unknown_dim = -1;
    int64_t const_size = 1;
    for (auto id: ids) {
      if (op.getResultType().getDimSize(id) == mlir::ShapedType::kDynamic) {
        if (unknown_dim != -1)
          throw UnsupportedException(op.getOperation(),
              "it has more than one unknown dimension size in one group");
        unknown_dim = i;
      } else {
        const_size *= op.getResultType().getDimSize(id);
      }
      ++i;
    }

    if (unknown_dim == -1)
      // Nothing to do; it is already well-defined
      continue;

    if (Index::BITS < 64 && (size_t)const_size >= (1ull << Index::BITS))
      throw UnsupportedException(op.getOperation(),
          "tensor size is too large");

    // If the original size isn't divisible, raise UB
    st.wellDefined(op, orgdim.urem(const_size) == 0);
    newdims[unknown_dim] = orgdim.udiv(const_size);
  }

  st.regs.add(op.getResult(), m.reshape(newdims));
  // Reshape does not look into memref's elements, so init check is not
  // necessary.
}

template<>
void encodeOp(State &st, mlir::memref::CollapseShapeOp op, bool) {
  auto srcType = op.getSrc().getType().cast<mlir::MemRefType>();
  auto resType = op.getResult().getType().cast<mlir::MemRefType>();

  if (!srcType.getLayout().isIdentity() || !resType.getLayout().isIdentity())
    throw UnsupportedException(op.getOperation(),
      "We do not support non-identity layout memref");

  MemRef m = st.regs.get<MemRef>(op.getOperand());
  mlir::ShapedType resTy = op.getResultType();

  auto reassocExprs = op.getReassociationIndices();
  assert(reassocExprs.size() == (size_t)resTy.getRank());

  vector<Expr> newDims;
  if (reassocExprs.size() == 0) {
    newDims.push_back(Index(1));
  } else {
    // If the collapsed size does not match op.getResultType(), it is UB.
    for (unsigned i = 0; i < reassocExprs.size(); ++i) {
      Expr size = Index::one();
      for (auto &idx: reassocExprs[i])
        size = size * m.getDim(idx);

      if (resTy.getDimSize(i) != mlir::ShapedType::kDynamic)
        st.wellDefined(op, size == resTy.getDimSize(i),
            "size check");
      newDims.push_back(std::move(size));
    }
  }

  st.wellDefined(op, m.get1DSize() == smt::get1DSize(newDims),
      "size check");
  st.regs.add(op.getResult(), m.reshape(newDims));
  // Note: tensor_collapse_shape does not look into elements, so initialization
  // check is not necessary.
}

template<>
void encodeOp(State &st, mlir::memref::CopyOp op, bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto *opr = op.getOperation();
  auto mrIn = st.regs.get<MemRef>(op.getSource());
  auto mrOut = st.regs.get<MemRef>(op.getTarget());

  // Src and tgt's shapes & element types must match
  for (unsigned i = 0; i < mrIn.getRank(); ++i)
    st.wellDefined(opr, (Expr)mrIn.getDim(i) == (Expr)mrOut.getDim(i));

  // They must not overlap, according to
  // https://mlir.llvm.org/docs/Dialects/Linalg/#linalgcopy-mlirlinalgcopyop
  st.wellDefined(opr, mrIn.noalias(mrOut), "src and dst does not alias");

  auto loadedTensor = loadTensor(st, op, mrIn,
      op.getSource().getType().cast<mlir::MemRefType>());

  storeTensorTo(st, opr, std::move(loadedTensor), mrOut,
      op.getTarget().getType().cast<mlir::MemRefType>(), true);
}

template<>
void encodeOp(State &st, mlir::linalg::FillOp op, bool encodeMemWrite) {
  if (op.hasPureBufferSemantics() && !encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");
  if (op.getNumResults() > 1)
    throw UnsupportedException(op.getOperation(),
        "it has multiple results");

  Expr elemval = getValueOrNegZero(st, op.getOperand(0));

  auto op1 = op.getOperand(1);
  auto ety = getElemTy(op1);

  if (op.hasPureTensorSemantics()) {
    auto t = st.regs.get<Tensor>(op1);
    auto filled = Tensor(ety, std::move(elemval), t.getDims());
    st.regs.add(op.getResult(0), std::move(filled));
  } else {
    assert(op.hasPureBufferSemantics());
    auto m = st.regs.get<MemRef>(op1);
    auto filled = Tensor(ety, std::move(elemval), m.getDims());
    storeTensorTo(st, op.getOperation(), std::move(filled), m,
        op1.getType().cast<mlir::MemRefType>(), true);
  }
}

template<>
void encodeOp(State &st, mlir::linalg::DotOp op, bool encodeMemWrite) {
  if (!op.hasPureTensorSemantics())
    throw UnsupportedException(op.getOperation(),
        "tensor semantics is supported only");

  auto inputOps = op.getInputs();
  auto outputOps = op.getOutputs();
  auto outputTy = op.getType(0).dyn_cast<mlir::TensorType>();

  // This must be same.
  assert(op.getNumResults() == outputOps.size());

  if (op.getNumResults() != 1)
    throw UnsupportedException(op.getOperation(),
        "it has multiple results");

  auto outputDim = ShapedValue::getDims(outputTy, false);
  if (outputDim.size() != 1)
    throw UnsupportedException(op.getOperation(),
        "unknown dot format; shouldn't the result tensor have one element?");

  if (outputTy.getElementType() !=
      inputOps[0].getType().dyn_cast<mlir::TensorType>()
          .getElementType())
    throw UnsupportedException(op.getOperation(), "casting is not supported");

  auto t1 = st.regs.get<Tensor>(inputOps[0]);
  auto t2 = st.regs.get<Tensor>(inputOps[1]);
  auto t3 = st.regs.get<Tensor>(outputOps[0]);
  st.wellDefined(op, t1.isFullyInitialized());
  st.wellDefined(op, t2.isFullyInitialized());
  st.wellDefined(op, t3.isFullyInitialized());
  st.wellDefined(op, t1.get1DSize() == t2.get1DSize());

  auto res = t1.dot(t2, std::move(t3.get({Index(0)})));
  st.regs.add(op.getResult(0),
      Tensor(t1.getElemType(), std::move(res), std::move(outputDim)));
}

template<>
void encodeOp(State &st, mlir::shape::ToExtentTensorOp op, bool) {
  // TODO: MLIR doc says
  //   If the shape represents an error, this op’s behavior is undefined.
  // Should figure out whether this applies to a Tensor operand as well.
  if (!op.getOperand().getType().isa<mlir::TensorType>())
    throw UnsupportedException(op.getOperation(), "unsupported type");

  auto tt = st.regs.get<Tensor>(op.getOperand());
  assert(tt.getDims().size() ==
         (size_t)op.getType().cast<mlir::TensorType>().getRank());
  st.regs.add(op, tt);
}

template<>
void encodeOp(State &st, mlir::sparse_tensor::ConvertOp op, bool) {
  auto tensor = op.getOperand();
  auto tt = st.regs.get<Tensor>(tensor);
  st.wellDefined(op, tt.isFullyInitialized(), "input initialized");
  st.regs.add(op, std::move(tt));
}

vector<Index> findLoopBounds(State &st, mlir::linalg::GenericOp op) {
  // The size of the loop is calculated (analogous to what
  // LinalgOp::createLoopRanges does).
  // LinalgOp::createLoopRanges relies on the "first" dimension that is
  // matched. If there are multiple matching dimensions, for example:
  //   linalg.generic {
  //      indexing_maps = [affine_map<(n) -> (n)>,
  //                       affine_map<(n) -> (n)>,
  //                       affine_map<(n) -> (n)>] }
  //      ins(%A, %B: <?xf32>, <?xf32>) outs(%C: <?xf32>) { .. }
  // The current algorithm mandates the result to be %A's dimension.

  vector<Index> viewSizes;
  for (auto &opOperand : op.getOperation()->getOpOperands()) {
    unsigned r = op.getRank(&opOperand);
    if (!r)
      continue;

    if (opOperand.get().getType().isa<mlir::TensorType>()) {
      auto t = st.regs.get<Tensor>(opOperand.get());
      for (int64_t i = 0, e = r; i < e; ++i) {
        viewSizes.push_back(t.getDim(i));
      }
    } else if (opOperand.get().getType().isa<mlir::MemRefType>()) {
      auto t = st.regs.get<MemRef>(opOperand.get());
      for (int64_t i = 0, e = r; i < e; ++i) {
        viewSizes.push_back(t.getDim(i));
      }
    }
  }

  if (viewSizes.empty()) {
    // Return [0] if all operands have zero rank, because there exists only
    // one element.
    // This is consistent with what ShapedValue::getDims does.
    return {Index(0)};
  }

  mlir::AffineMap map = op.getLoopsToShapesMap();
  // numDims: # of induction variables
  unsigned numDims = map.getNumDims();
  // numRes: # of output affine Exprs
  // For example, given two affine maps
  //   (i, j, k) -> (i, j)
  //   (i, j, k) -> (i, k)
  //   numDims = 3 (i, j, k), numRes = 4 (i, j, i, k)
  unsigned numRes = map.getNumResults();

  vector<Index> res;
  vector<int> resFilled(numDims);
  std::fill(resFilled.begin(), resFilled.end(), -1);

  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    auto d = result.dyn_cast<mlir::AffineDimExpr>();
    if (!d)
      continue;

    unsigned pos = d.getPosition();
    if (resFilled[pos] != -1)
      continue;
    // If i < N, store N - 1
    // It is to bound e.g., 'i + j <= N - 1 + M - 1'
    resFilled[pos] = res.size();
    res.push_back(viewSizes[idx].ofs(-1));
  }

  vector<Index> res_ordered;
  for (unsigned i = 0; i < numDims; ++i)
    res_ordered.push_back(std::move(res[resFilled[i]]));

  return res_ordered;
}

static void
encodeUBForTensorShapeMatch(State &st, mlir::linalg::GenericOp op,
                            const vector<Index> &indVarBounds) {
  mlir::AffineMap map = op.getLoopsToShapesMap();
  // numRes: # of output affine Exprs
  // For example, given two affine maps
  //   (i, j, k) -> (i, j)
  //   (i, j, k) -> (i, k)
  //   numDims = 3 (i, j, k), numRes = 4 (i, j, i, k)
  unsigned numRes = map.getNumResults();

  vector<Index> viewSizes;
  for (auto &oo : op.getOperation()->getOpOperands()) {
    auto *opOperand = &oo;
    unsigned r = op.getRank(opOperand);
    if (!r)
      continue;

    auto value = st.regs.findOrCrash(opOperand->get());
    ShapedValue *t;
    if (holds_alternative<MemRef>(value)) {
      t = &get<MemRef>(value);
    } else if(holds_alternative<Tensor>(value)) {
      t = &get<Tensor>(value);
    } else {
      throw UnsupportedException(op.getOperation(), "Unsupported ShapedValue");
    }
    for (int64_t i = 0, e = r; i < e; ++i) {
      viewSizes.push_back(t->getDim(i));
    }
  }

  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto ae = encodeAffineExpr(map.getResult(idx), indVarBounds, {});
    if (!ae)
      throw UnsupportedException(op.getOperation(), "unsupported affine Expr");
    // Induction variable's bounds should be matched with
    // other tensor's dimension.
    Expr size = (Expr)viewSizes[idx].ofs(-1);
    st.wellDefined(op, std::move(*ae == size));
  }
}

static void initInputStateForLoopBody(
    State &st, mlir::linalg::GenericOp op,
    map<string, Expr> &welldefs,
    bool isParallelLoop) {
  auto indexingMaps = op.getIndexingMaps().getValue();
  auto &block = *op.getRegion().begin();

  const vector<Expr> &inductionVars = st.linalgGenericScopes.top().indVars;

  auto nInputs = op.getInputs().size();
  auto nOutputs = op.getOutputs().size();
  assert(nInputs + nOutputs == indexingMaps.size());

  // The output variables contain the initial value of the tensor
  //   (see github issue #164)
  // For parallel loops: whole iterations contain the initial value
  // For reduction loops: only the first iteration contains the value
  size_t upperbound = nInputs + nOutputs;

  for (size_t arg_i = 0; arg_i < upperbound; ++arg_i) {
    auto indexMap = indexingMaps[arg_i].cast<mlir::AffineMapAttr>().getValue();
    mlir::Value op_i = op->getOperand(arg_i);
    bool isInput = arg_i < nInputs;
    bool isOutputAndHasUse = !isInput && !block.getArgument(arg_i).use_empty();

    if (op_i.getType().isa<mlir::FloatType>()) {
      // A scalar value.
      Float f_input = st.regs.get<Float>(op_i);
      st.regs.add(block.getArgument(arg_i), f_input);

    } else if (auto tensorty = op_i.getType().dyn_cast<mlir::TensorType>()) {
      // A tensor value.
      auto elemty = tensorty.getElementType();
      Tensor t_input = st.regs.get<Tensor>(op_i);

      if (indexMap.getNumResults() == 0) {
        // A tensor with a single element; e.g. tensor<f32>.
        st.regs.add(block.getArgument(arg_i),
            t_input.get({Index::zero()}), elemty);
        // Reading uninitialized elements is UB.
        // For output variables, encode uninitialized if it syntactically has
        // uses.
        // This is a workaround (overapproximation) for not introducing a
        // 'poison' value.
        if (isInput || isOutputAndHasUse) {
          addToBoolMap(welldefs,
              "op " + to_string(arg_i) + "'s initializedness",
              t_input.isFullyInitialized());
        }
      } else {
        vector<Expr> indices;
        for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
          auto ae_res =
              encodeAffineExpr(indexMap.getResult(i), inductionVars, {});
          if (!ae_res) {
            string msg;
            TO_STRING(msg, "Unsupported affine expr: "<< indexMap.getResult(i));
            throw UnsupportedException(op.getOperation(), std::move(msg));
          }

          indices.emplace_back(std::move(*ae_res));
        }

        // The out-of-bounds checking is done when encoding loop bounds.
        auto t_elem = t_input.get(indices);
        st.regs.add(block.getArgument(arg_i), t_elem, elemty);

        // Reading uninitialized elements is UB.
        // For output variables, encode uninitialized if it syntactically has
        // uses.
        // This is a workaround (overapproximation) for not introducing a
        // 'poison' value.
        if (isInput || isOutputAndHasUse) {
          addToBoolMap(welldefs,
              "op " + to_string(arg_i) + "'s initializedness",
              t_input.isInitialized(indices));
        }
      }

    } else if (auto memrefty = op_i.getType().dyn_cast<mlir::MemRefType>()) {
      // A MemRef value.
      MemRef m_input = st.regs.get<MemRef>(op_i);

      vector<Expr> indices;
      for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
        auto ae_res =
            encodeAffineExpr(indexMap.getResult(i), inductionVars, {});
        if (!ae_res) {
          string msg;
          TO_STRING(msg, "Unsupported affine expr: "<< indexMap.getResult(i));
          throw UnsupportedException(op.getOperation(), std::move(msg));
        }

        indices.emplace_back(std::move(*ae_res));
      }

      // Reading uninitialized elements is UB.
      // For output variables, encode uninitialized if it syntactically has
      // uses.
      // This is a workaround (overapproximation) for not introducing a
      // 'poison' value.
      auto [m_elem, m_welldef] = m_input.getWithAccessInfo(indices);
      if (isInput) {
        addToBoolMap(welldefs, "reading op " + to_string(arg_i) + "'s safety",
            m_welldef.checkRead());
      } else {
        if (isOutputAndHasUse) {
          addToBoolMap(welldefs,
              "reading and writing to op " + to_string(arg_i) + "'s safety",
              m_welldef.checkReadWrite());
        } else {
          addToBoolMap(welldefs,
              "writing to op " + to_string(arg_i) + "'s safety",
              m_welldef.checkWrite());
        }
      }
      mlir::Type elemTy = memrefty.getElementType();
      st.regs.add(block.getArgument(arg_i), m_elem, elemTy);

    } else {
      throw UnsupportedException(op.getOperation(),
          "unsupported block argument type");
    }
  }
}

static void encodeReductionLoopBodyAndOutput(
    State &newst, mlir::Block &block,
    const mlir::ArrayRef<mlir::Attribute> &indexingMaps,
    const mlir::ShapedType &outputType,
    optional<Tensor> &t_res,
    map<string, Expr> &welldefs) {
  // Deal with simple reduction loops.
  // TODO: support more kinds of reduction loops!
  string errmsg = "permutated output map or simple reduction form is"
                  " supported only";
  mlir::Operation *the_op = block.getParentOp();

  auto &ops = block.getOperations();
  int instcount = ops.size();

  using mlir::m_Op;
  using mlir::matchers::m_Any;
  using mlir::matchers::m_Val;
  // Support this form:
  //   ...
  //   %sum = op %v, %arg_out or  %sum = op %arg_out, %v
  //      where op = addf, addi
  //   yield %sum
  auto lastarg = block.getArgument(block.getNumArguments() - 1);

  auto p1 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::arith::AddFOp>(m_Val(lastarg), m_Any()));
  auto p2 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::arith::AddFOp>(m_Any(), m_Val(lastarg)));
  auto p3 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::arith::AddIOp>(m_Val(lastarg), m_Any()));
  auto p4 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::arith::AddIOp>(m_Any(), m_Val(lastarg)));

  unsigned idx;
  if (p1.match(&ops.back()) || p3.match(&ops.back()))      idx = 1;
  else if (p2.match(&ops.back()) || p4.match(&ops.back())) idx = 0;
  else
    throw UnsupportedException(the_op, std::move(errmsg));

  auto sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(idx);

  // TODO: deal with merging memories
  Expr opsWelldef = Expr::mkBool(true);
  encodeBlock(newst, block, /*print ops*/false, /*encode mem writes*/false,
      [instcount, &lastarg, &the_op](
          mlir::Operation *op, int opindex) {
        if (opindex >= instcount - 2)
          // Don't directly encode %sum and yield
          return true;

        auto op_operands = op->getOperands();
        for (const auto &opop: op_operands) {
          if (lastarg == opop) {
            string msg;
            TO_STRING(msg, "Unsupported reduction form because it contains "
                << *op);
            throw UnsupportedException(the_op, std::move(msg));
          }
        }
        return false;
      },
      [&opsWelldef, &newst](mlir::Operation *op) {
        opsWelldef &= newst.isOpWellDefined(op);
      });
  addToBoolMap(welldefs, "loop body", std::move(opsWelldef));

  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();

  auto &linalgInfo = newst.linalgGenericScopes.top();

  // Represent %v as an element of a tensor.
  Tensor t_v = Tensor::mkInitializedLambda(
      sumvar.getType(),
      addOne(vector(linalgInfo.indVarUpperBounds)),
      vector(linalgInfo.indVars),
      newst.regs.getExpr(sumvar));

  if (llvm::all_of(outputMap.getResults(), [](const mlir::AffineExpr &Expr) {
    auto ac = Expr.dyn_cast<mlir::AffineConstantExpr>();
    return ac && ac.getValue() == 0;
  })) {
    // in:  (i, j) -> (i, j)
    // out: (i, j) -> (0)
    // =>
    // t_res[0] = sum(\i. t_input[i / n][i % n] , i < m * n)

    // Define this as a splat tensor (num. elems is 1 anyway)
    // TODO(aqjune): Support memref cases (memref.isFullyInitialized)
    auto outTensor = newst.regs.get<Tensor>(the_op->getOperands().back());
    auto initElem = outTensor.get({Index(0)});
    t_res = Tensor(t_v.getElemType(), t_v.sum(std::move(initElem)),
          makeCube(Index(1), outputType.getRank()));
    addToBoolMap(welldefs, "output tensor is initialized",
        outTensor.isFullyInitialized());
  } else {
    // in:  (i, j) -> (i, j)
    // out: (i, j) -> (i)
    // =>
    // t_res[i] = sum(\j. t_input[i][j] , j < m)

    // Gather affine vars that are unused in the output (e.g. j) first.
    vector<bool> isInputIdxUsed(outputMap.getNumInputs());
    for (unsigned j = 0; j < outputMap.getNumResults(); ++j) {
      auto Expr = outputMap.getResult(j);

      if (auto ade = Expr.dyn_cast<mlir::AffineDimExpr>()) {
        isInputIdxUsed[ade.getPosition()] = true;
      } else {
        // Output map has an unknown form
        throw UnsupportedException(the_op, std::move(errmsg));
      }
    }

    vector<Expr> boundsForRes;
    vector<Expr> indVarsForRes;
    for (unsigned j = 0; j < isInputIdxUsed.size(); ++j) {
      if (!isInputIdxUsed[j]) {
        boundsForRes.push_back(linalgInfo.indVarUpperBounds[j]);
        indVarsForRes.push_back(linalgInfo.indVars[j]);
      }
    }

    auto outputIndVars = doMap(linalgInfo.indVars, outputMap);

    optional<Expr> initElem;
    mlir::Value outOp = the_op->getOperands().back();
    if (outOp.getType().isa<mlir::TensorType>()) {
      Tensor outTensor = newst.regs.get<Tensor>(outOp);
      initElem = outTensor.get(outputIndVars);
      addToBoolMap(welldefs, "output tensor is initialized",
          outTensor.isFullyInitialized());
    } else {
      MemRef outMemRef = newst.regs.get<MemRef>(outOp);
      auto [v, ainfo] = outMemRef.getWithAccessInfo(outputIndVars);
      initElem.emplace(std::move(v));
      addToBoolMap(welldefs, "output tensor is initialized",
          Expr::mkForall(outputIndVars,
            ainfo.inbounds.implies(ainfo.initialized)));
    }

    auto tensorSz = addOne(doMap(linalgInfo.indVarUpperBounds, outputMap));
    auto t_sum = Tensor::mkInitializedLambda(
          t_v.getElemType(),
          addOne(std::move(boundsForRes)),
          std::move(indVarsForRes),
          t_v.get(linalgInfo.indVars))
        .sum(std::move(*initElem));

    t_res = Tensor::mkInitializedLambda(
        t_v.getElemType(), std::move(tensorSz), std::move(outputIndVars), t_sum);
  }
}

template<>
void encodeOp(State &st, mlir::linalg::GenericOp op, bool encodeMemWriteOp) {
  if (!(op.hasPureTensorSemantics() || op.hasPureBufferSemantics()))
    throw UnsupportedException(op.getOperation(),
        "tensor/buffer semantics is supported only");

  else if (op.hasPureBufferSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto &region = op.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(op.getOperation(),
        "a single block is supported only");

  auto &block = region.front();
  if (!std::all_of(block.args_begin(), block.args_end(),
      [](auto &arg) { return arg.getType().isSignlessIntOrFloat(); }))
    throw UnsupportedException(op.getOperation(),
        "unsupported block arguments");

  if (llvm::any_of(op.getIteratorTypesArray(), [](auto itrty) {
    return itrty != mlir::utils::IteratorType::parallel &&
           itrty != mlir::utils::IteratorType::reduction;
  }))
    throw UnsupportedException(op.getOperation(),
        "unsupported iterator type");

  // Find the inclusive upper bounds
  auto loopBounds = findLoopBounds(st, op);

  encodeUBForTensorShapeMatch(st, op, loopBounds);

  // Start from newst
  optional<vector<Tensor>> tvec_res;
  // reason -> WB (= !UB)
  map<string, Expr> welldefs;
  {
    State newst = st;
    newst.linalgGenericScopes.push(State::LinalgGenericScope{loopBounds});

    auto indexingMaps = op.getIndexingMaps().getValue();
    auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
    bool isParallelLoop = outputMap.isPermutation();

    initInputStateForLoopBody(newst, op, welldefs, isParallelLoop);

    auto &indVars = newst.linalgGenericScopes.top().indVars;

    if (isParallelLoop) {
      Expr welldef = Expr::mkBool(true);
      encodeParallelLoopBodyAndOutputs(newst, block, outputMap,
          tvec_res, welldef);
      addToBoolMap(welldefs, "loop body", std::move(welldef));

    } else {
      // Reduction loops returning multiple values is not supported by MLIR-TV
      // yet.
      if (op.getOutputs().size() > 1)
        throw UnsupportedException(op.getOperation(),
            "unsupported reduction form");

      optional<Tensor> t_res;
      auto outputType = op.getOutputs().front().getType().cast<mlir::ShapedType>();
      // Reduction loops returning memref is not supported by MLIR-TV yet.
      if (outputType.isa<mlir::MemRefType>())
        throw UnsupportedException(op.getOperation(),
            "using memref as a reduction loop output is not yet supported");
      
      encodeReductionLoopBodyAndOutput(newst, block,
            indexingMaps, outputType, t_res, welldefs);
      tvec_res = {*t_res};
    }

    for(unsigned i = 0; i < tvec_res->size(); i++) {
      assert(tvec_res->at(i).getDims().size() != 0);
    }

    // Encode UB of linalg.generic.
    // For all induction vars' values, there must be no UB.
    Expr inbounds = Expr::mkBool(true);
    for (int i = 0; i < indVars.size(); ++i) {
      inbounds &= indVars[i].ult(loopBounds[i] + 1);
    }

    // Encode well-definedness.
    for (auto &[itm, wdef]: welldefs) {
      st.wellDefined(op, Expr::mkForall(indVars, inbounds.implies(wdef)),
          string(itm));
    }
  }

  if (op.hasPureTensorSemantics()) {
    for(unsigned i = 0; i < tvec_res->size(); i++) {
      // NOTE: op's output tensor (op.getOutputOperand()[0]->get())
      // isn't updated;
      // aqjune talked with mlir people and confirmed
      st.regs.add(op.getResult(i), std::move(tvec_res->at(i)));
    }
  } else if (op.hasPureBufferSemantics()) {
    unsigned i = 0;
    assert(op.getOutputs().size() == tvec_res->size());

    for(auto opi: op.getOutputs()) {
      // unsigned i = 0; i < tvec_res->size(); i++
      auto m_res = st.regs.get<MemRef>(opi);
      storeTensorTo(st, op, std::move(tvec_res->at(i)), m_res,
          opi.getType().cast<mlir::MemRefType>(), true);

      // Noalias with input operands
      for (auto opj: op.getInputs()) {
        if (!opj.getType().isa<mlir::MemRefType>()) continue;

        auto input = st.regs.get<MemRef>(opj);
        st.wellDefined(op, input.noalias(m_res));
      }
      // Noalias with other output operands
      unsigned j = 0;
      for (auto opj: op.getOutputs()) {
        if (j >= i) break;
        if (!opj.getType().isa<mlir::MemRefType>()) continue;

        auto output = st.regs.get<MemRef>(opj);
        st.wellDefined(op, output.noalias(m_res));
        ++j;
      }
      ++i;
    }
  } else {
    llvm_unreachable("Unknown linalg::generic semantics");
  }
}


#define ENCODE(st, op, ty, encodeMemWriteOps) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    try { \
      encodeOp(st, op2, encodeMemWriteOps); \
    } catch (UnsupportedException ue) { \
      if (arg_assign_random_to_unsupported_ops.getValue()) { \
        assignRandomValue(st, &op, printOps); \
      } else { \
        if (std::holds_alternative<mlir::Operation *>(ue.getObject())) { \
          auto *op_ue = std::get<mlir::Operation *>(ue.getObject()); \
          if (!op_ue) \
            throw UnsupportedException(&op, ue.getReason()); \
        } \
        throw ue; \
      } \
    } \
    if (callbackAfterEnc) callbackAfterEnc(&op); \
    continue; \
  }

static void assignRandomValue(State &st, mlir::Operation *op, bool printOp) {
  if (printOp) {
    llvm::outs() << "    Assigning any value to this op ("
        << op->getName() << ")..\n";
  }

  for (auto r: op->getResults()) {
    auto ty = r.getType();
    if (auto ity = ty.dyn_cast<mlir::IntegerType>()) {
      Integer i(0, ity.getIntOrFloatBitWidth());
      st.regs.add(r, std::move(i));

    } else if (auto fty = ty.dyn_cast<mlir::FloatType>()) {
      st.regs.add(r, *getIdentity(fty), fty);

    } else if (auto tty = ty.dyn_cast<mlir::RankedTensorType>()) {
      auto dims = ShapedValue::getDims(tty);
      auto elemTy = tty.getElementType();
      Tensor t(elemTy, *getIdentity(elemTy), std::move(dims));
      st.regs.add(r, std::move(t));

    } else {
      // TODO: support memref
      throw UnsupportedException(op, "Cannot assign random value");
    }
  }
}

static void encodeBlock(
    State &st, mlir::Block &block, bool printOps, bool encodeMemWriteOps,
    // checkBeforeEnc: return true if the op is to be ignored (e.g. yield)
    function<bool(mlir::Operation *, int)> checkBeforeEnc,
    function<void(mlir::Operation *)> callbackAfterEnc) {

  int index = -1;
  for (auto &op: block) {
    index++;
    if (printOps)
      llvm::outs() << "  " << op << "\n";

    if (checkBeforeEnc && checkBeforeEnc(&op, index)) continue;

    // Encode ops. Alphabetically sorted.
    ENCODE(st, op, mlir::affine::AffineApplyOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::arith::AddFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::AddIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::CmpFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::CmpIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantFloatOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantIndexOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantIntOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::DivFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ExtFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ExtSIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ExtUIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::IndexCastOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::MulFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::MulIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::NegFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SelectOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ShLIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ShRSIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ShRUIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SIToFPOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SubFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SubIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::TruncFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::TruncIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::XOrIOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::bufferization::CloneOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::bufferization::ToMemrefOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::bufferization::ToTensorOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::func::CallOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::func::ReturnOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::math::AbsFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::math::AbsIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::math::ExpOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::memref::AllocOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::AllocaOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::CollapseShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::CopyOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::DeallocOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::DimOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::ExpandShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::GetGlobalOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::LoadOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::StoreOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::SubViewOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::linalg::DepthwiseConv2DNhwcHwcmOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::Conv2DNchwFchwOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::Conv2DNhwcHwcfOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::DotOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::FillOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::GenericOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::IndexOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::MatmulOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::PoolingNhwcMaxOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::PoolingNhwcSumOp, encodeMemWriteOps);
    
    ENCODE(st, op, mlir::shape::ShapeOfOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::shape::ToExtentTensorOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::sparse_tensor::ConvertOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::tensor::CastOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::CollapseShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::DimOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::EmptyOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExpandShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::InsertOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExtractOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExtractSliceOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::FromElementsOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::GenerateOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::InsertSliceOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::PadOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::tosa::AbsOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::AddOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::AvgPool2dOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::BitwiseAndOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::BitwiseNotOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::BitwiseOrOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::BitwiseXorOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ClampOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ConcatOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ConstOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::Conv2DOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::DepthwiseConv2DOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ExpOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::FullyConnectedOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::GatherOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::MaxPool2dOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::MulOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::NegateOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ReciprocalOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ReduceSumOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ReshapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::ReverseOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::SubOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::TileOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::TransposeOp, encodeMemWriteOps);

    if (arg_assign_random_to_unsupported_ops.getValue()) {
      assignRandomValue(st, &op, printOps);
    } else {
      throw UnsupportedException(&op);
    }
  }
  if (printOps)
    llvm::outs() << "\n";
}

void encode(State &st, mlir::func::FuncOp &fn, bool printOps) {
  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  auto &block = region.front();

  encodeBlock(st, block, printOps, true/*allow mem ops*/, {}, {});
}
