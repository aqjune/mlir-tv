#include "encode.h"
#include "abstractops.h"
#include "opts.h"
#include "utils.h"
#include "debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"

#include <functional>
#include <map>
#include <sstream>
#include <variant>
#include <vector>
#include <optional>

using namespace smt;
using namespace std;

llvm::cl::opt<bool> arg_assign_random_to_unsupported_ops(
      "assign-random-to-unsupported-ops",
  llvm::cl::desc("Assign a random value to the result of unsupported ops. "
      "Note that this option is purely for debugging purpose. This flag will "
      "make the validation result meaningless."),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));


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

template<class ValTy>
vector<ValTy> getFromMixedOps(
    const State &st, const llvm::SmallVector<mlir::OpFoldResult> &mixedOps) {
  vector<ValTy> vec;
  for (auto s: mixedOps) {
    vec.push_back(s.is<mlir::Value>() ?
      st.regs.get<ValTy>(s.get<mlir::Value>()) :
      Index(s.get<mlir::Attribute>().dyn_cast<mlir::IntegerAttr>().getInt()));
  }
  return vec;
}


template<class ValTy>
vector<Expr> getFromArrayAttr(const mlir::ArrayAttr &attr) {
  vector<Expr> vec;
  for (auto s: attr) {
    vec.push_back(ValTy(s.dyn_cast<mlir::IntegerAttr>().getInt()));
  }
  return vec;
}


template<class T>
optional<Expr> encodeAffineExpr(
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
    auto ade = ae.dyn_cast<mlir::AffineDimExpr>();
    auto id = ade.getPosition();
    assert(id < dimvars.size());
    return dimvars[id];
  }
  case mlir::AffineExprKind::SymbolId: {
    auto ade = ae.dyn_cast<mlir::AffineSymbolExpr>();
    auto id = ade.getPosition();
    assert(id < symbolvars.size());
    return symbolvars[id];
  }
  case mlir::AffineExprKind::Constant: {
    auto ac = ae.dyn_cast<mlir::AffineConstantExpr>();
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
  return v.getType().dyn_cast<mlir::ShapedType>().getElementType();
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

    bool dyn0 = d1 == mlir::ShapedType::kDynamicSize;
    bool dyn1 = d2 == mlir::ShapedType::kDynamicSize;
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
      move(resDims0), move(inVars0), t0.get(outVars0));

  auto m1 = Tensor::mkInitializedLambda(t1.getElemType(),
      move(resDims1), move(inVars1), t1.get(outVars1));

  return {{m0, m1}};
}

template<class OpTy>
static void
encodeBinaryOp(State &st, OpTy op, mlir::Value arg0, mlir::Value arg1,
    function<Float(Float &&e1, Float &&e2)> f_float,
    function<Integer(Integer &&e1, Integer &&e2)> f_int) {

  mlir::Operation *opr = op.getOperation();

  if (arg0.getType().isa<mlir::FloatType>()) {
    auto a = st.regs.get<Float>(arg0);
    auto b = st.regs.get<Float>(arg1);
    st.regs.add(op, f_float(move(a), move(b)));

  } else if (arg0.getType().isa<mlir::IntegerType>()) {
    auto a = st.regs.get<Integer>(arg0);
    auto b = st.regs.get<Integer>(arg1);
    st.regs.add(op, f_int(move(a), move(b)));

  } else if (arg0.getType().isa<mlir::IndexType>()) {
    auto a = st.regs.get<Index>(arg0);
    auto b = st.regs.get<Index>(arg1);
    st.regs.add(op, Index::fromInteger(f_int(a.asInteger(), b.asInteger())));

  } else if (auto tty = arg0.getType().dyn_cast<mlir::RankedTensorType>()) {
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
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()));
    st.wellDefined(op, a.isFullyInitialized());
    st.wellDefined(op, b.isFullyInitialized());

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
    auto a = st.regs.get<Float>(arg);
    st.regs.add(op, f_float(move(a)));

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
    st.regs.add(op, a.elementwiseUnaryOp(elemty, f));
    st.wellDefined(op, a.isFullyInitialized());

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
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()));
    st.wellDefined(op, a.isFullyInitialized());
    st.wellDefined(op, b.isFullyInitialized());

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
    st.wellDefined(op, listsEqual(a.getDims(), b.getDims()));
    st.wellDefined(op, a.isFullyInitialized());
    st.wellDefined(op, b.isFullyInitialized());

  } else if (op1Type.isa<mlir::IntegerType>() &&
              op2Type.isa<mlir::IntegerType>()) {
    auto a = st.regs.get<Integer>(op.getOperand(0));
    auto b = st.regs.get<Integer>(op.getOperand(1));
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

    st.regs.add(op, move(te));

  } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    st.regs.add(op, attrToValueTy(intAttr));

  } else {
    throw UnsupportedException(op.getOperation(), "Unsupported constant");
  }
}

enum class FPPrecision {
  // F16,
  F32,
  F64
};

static FPPrecision getPrecision(mlir::Type &type) {
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

template<>
void encodeOp(State &st, mlir::arith::ExtFOp op, bool) {
  auto op_type = op.getType();
  FPPrecision tgt_prec = getPrecision(op_type);

  auto operand_type = op.getOperand().getType();
  FPPrecision src_prec = getPrecision(operand_type);

  if (src_prec == tgt_prec) {
    st.regs.add(op.getResult(), st.regs.get<Float>(op.getOperand()));
    return; // extending into identical type is a no-op
  } else if (src_prec > tgt_prec) {
    throw UnsupportedException(op.getOperation(),
      "cannot ExtF into lower precision type!");
  }

  auto arg = op.getOperand();
  encodeUnaryOp(st, op, arg, [op_type](auto &&a) { return a.extend(op_type); },
      {});
}

template<>
void encodeOp(State &st, mlir::arith::TruncFOp op, bool) {
  auto op_type = op.getType();
  FPPrecision tgt_prec = getPrecision(op_type);

  auto operand_type = op.getOperand().getType();
  FPPrecision src_prec = getPrecision(operand_type);

  if (src_prec == tgt_prec) {
    st.regs.add(op.getResult(), st.regs.get<Float>(op.getOperand()));
    return; // truncating into identical type is a no-op
  } else if (src_prec < tgt_prec) {
    throw UnsupportedException(op.getOperation(),
      "cannot TruncF into higher precision type!");
  }

  auto arg = op.getOperand();
  encodeUnaryOp(st, op, arg,
      [op_type](auto &&a) { return a.truncate(op_type); },
      {});
}

template<>
void encodeOp(State &st, mlir::linalg::IndexOp op, bool) {
  uint64_t i = op.dim();
  assert(i < st.linalgGenericScopes.top().indVars.size());
  Expr idxvar = st.linalgGenericScopes.top().indVars[i];
  st.regs.add(op, Index(idxvar));
}

template<>
void encodeOp(State &st, mlir::math::AbsOp op, bool) {
  mlir::Value arg0 = op.getOperand();

  encodeUnaryOp(st, op, arg0, [](auto &&a) { return a.abs(); }, {});
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
          dst_elemty, move(e));
    });
    st.regs.add(op, move(res));
    st.wellDefined(op, src.isFullyInitialized());

  } else {
    auto src = st.regs.getExpr(op.getOperand());
    auto res = evalIndexCastOp(srcty, dstty, move(src));
    if (dstty.isIndex())
      st.regs.add(op, Index(res));
    else
      st.regs.add(op, Integer(res));
  }
}

template<>
void encodeOp(State &st, mlir::AffineApplyOp op, bool) {
  auto m = op.getAffineMap();
  if (m.getNumResults() != 1)
    throw UnsupportedException(
        op.getOperation(), "num results is larger than one");

  auto dimOperands = op.mapOperands().take_front(m.getNumDims());
  auto symbolOperands = op.mapOperands().take_back(m.getNumSymbols());

  vector<Index> indices, symbols;
  for (auto arg: dimOperands)
    indices.push_back(st.regs.get<Index>(arg));
  for (auto symbol: symbolOperands)
    symbols.push_back(st.regs.get<Index>(symbol));

  auto res = encodeAffineExpr(m.getResult(0), indices, symbols);
  if (!res)
    throw UnsupportedException(op.getOperation(), "unsupported affine Expr");
  st.regs.add(op, Index(move(*res)));
}

template<>
void encodeOp(State &st, mlir::ReturnOp op, bool) {
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    st.retValues.push_back(st.regs.findOrCrash(op.getOperand(i)));
}

template<>
void encodeOp(State &st, mlir::SelectOp op, bool) {
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
    st.wellDefined(op, move(welldef));
    // Operands must be initialized.
    st.wellDefined(op, trueValue.isFullyInitialized());
    st.wellDefined(op, falseValue.isFullyInitialized());

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
    st.wellDefined(op, listsEqual(trueValue.getDims(), falseValue.getDims()));

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
  st.wellDefined(op, t.isFullyInitialized());
}

template<>
void encodeOp(State &st, mlir::tosa::ConcatOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  uint64_t axis = op.axis();
  auto t = st.regs.get<Tensor>(op.getOperand(0));
  st.wellDefined(op, t.isFullyInitialized());

  for (auto tensor: op.getOperands().drop_front()) {
    auto t2 = st.regs.get<Tensor>(tensor);
    st.wellDefined(op, t2.isFullyInitialized());
    for (unsigned i = 0; i < t2.getRank(); ++i) {
      if (i != axis)
        st.wellDefined(op, t.getDim(i) == t2.getDim(i));
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

  auto input = st.regs.get<Tensor>(op.input());

  auto unaryFn = [elemTy, &op](smt::Expr &&elem0) -> smt::Expr {
    // In TOSA 0.23:
    // apply_clip := apply_min(apply_max(value, minval), maxval)
    // apply_max: (a >= b) ? a : b
    // apply_min: (a < b)  ? a : b

    if (elemTy.isa<mlir::IntegerType>()) {
      Integer minval(op.min_int(), elemTy.getIntOrFloatBitWidth());
      Integer maxval(op.max_int(), elemTy.getIntOrFloatBitWidth());
      Integer elem(elem0);
      elem = Expr::mkIte(((Expr)elem).sge(minval), elem, minval);
      elem = Expr::mkIte(((Expr)elem).slt(maxval), elem, maxval);

      return elem;
    } else {
      Float minval = Float::constant(op.min_fp(), elemTy);
      Float maxval = Float::constant(op.max_fp(), elemTy);
      Float elem(elem0, elemTy);
      auto olt = mlir::arith::CmpFPredicate::OLT;
      auto one = Expr::mkBV(1, 1);
      // NOTE: strictly speaking, this isn't
      // apply_min(apply_max(value, minval), maxval) because the results are
      // different if value is NaN!
      // But the definition makes validation of tosa-to-linalg lowering fail.
      // Needs a discussion about this.
      Expr e1 = Float(Expr::mkIte(
          (Expr)elem.cmp(olt, minval) == one, minval, elem), elemTy);
      Expr e2 = Float(Expr::mkIte(
          (Expr)maxval.cmp(olt, elem) == one, maxval, e1), elemTy);

      return e2;
    }
  };

  auto output = input.elementwiseUnaryOp(elemTy, unaryFn);
  
  st.wellDefined(op, input.isFullyInitialized());
  st.regs.add(op, output);
}

template<>
void encodeOp(State &st, mlir::tosa::ConstOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
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

  auto t = st.regs.get<Tensor>(op.input());
  auto axis = op.axis();

  st.regs.add(op, t.reverse(axis));
  st.wellDefined(op, t.isFullyInitialized());
}

template<>
void encodeOp(State &st, mlir::tosa::TileOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto t = st.regs.get<Tensor>(op.input1());
  vector<unsigned> repeat;
  for (mlir::Attribute val: op.multiples())
    repeat.push_back(val.cast<mlir::IntegerAttr>().getValue().getSExtValue());

  st.regs.add(op, t.tile(repeat));
  st.wellDefined(op, t.isFullyInitialized());
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseAndOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.input1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.input2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type"); 
  
  mlir::Value i1 = op.input1();
  mlir::Value i2 = op.input2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a & (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseNotOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.input1()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");

  mlir::Value i1 = op.input1();

  encodeUnaryOp(st, op, i1,
      nullptr,
      [](auto &&a) { return ~(Expr)a; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseOrOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.input1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.input2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type"); 
  
  mlir::Value i1 = op.input1();
  mlir::Value i2 = op.input2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a | (Expr)b; });
}

template<>
void encodeOp(State &st, mlir::tosa::BitwiseXorOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  if(!getElemTy(op.input1()).isa<mlir::IntegerType>() ||
      !getElemTy(op.input2()).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");
  
  mlir::Value i1 = op.input1();
  mlir::Value i2 = op.input2();

  encodeBinaryOp(st, op, i1, i2,
      nullptr,
      [](auto &&a, auto &&b) { return (Expr)a ^ (Expr)b; });
}

static Tensor getPaddedTensor2D(mlir::Type elemTy, 
                                Tensor input, 
                                mlir::ArrayAttr padding) {
  if (!llvm::all_of(padding, [](mlir::Attribute a) {
      return a.cast<mlir::IntegerAttr>().getInt() == 0; })) {

    // pad = [top, bottom, left, right], filled with zero
    vector<Expr> pad = getFromArrayAttr<Index>(padding);
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

    // TOSA pad operands fill padded area as +0.0
    auto zero = *getZero(elemTy);
    Expr padVal = Expr::mkIte(cond, input.get(srcInd), zero);

    return Tensor::mkInitializedLambda(
                    elemTy, move(padDims), move(padInd), padVal);

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
            elemTy, move(dims), move(ind), 
            tf.add(biasf)
          );
}

template<>
void encodeOp(State &st, mlir::tosa::DepthwiseConv2DOp op, bool) {
  // input's dim sizes = [N, H, W, C]
  auto input = st.regs.get<Tensor>(op.input());
  // weight's dim sizes = [H, W, C, M]
  auto weight = st.regs.get<Tensor>(op.weight());
  // bias: a 1-dim array whose size is C * M
  auto bias = st.regs.get<Tensor>(op.bias());
  // strides = [strides_y, strides_x]
  vector<Expr> strides = getFromArrayAttr<Index>(op.stride());
  // dilations = [dilations_y, dilations_x]
  vector<Expr> dilations = getFromArrayAttr<Index>(op.dilation());

  auto elemTy = getElemTy(op.getResult());

  auto C = weight.getDim(2);
  auto M = weight.getDim(3);

  // Check whether C is identical
  st.wellDefined(op, input.getDim(3) == C);
  // Check whether C * M is identical
  st.wellDefined(op, bias.getDim(0) == (C * M));

  auto paddedTensor = getPaddedTensor2D(elemTy, input, op.pad());

  auto output = paddedTensor.depthwiseConv2D(weight, strides, dilations, bias);
  
  st.wellDefined(op, input.isFullyInitialized());
  st.wellDefined(op, weight.isFullyInitialized());
  st.wellDefined(op, bias.isFullyInitialized());

  st.regs.add(op, output);

}

template<>
void encodeOp(State &st, mlir::tosa::Conv2DOp op, bool) {
  // input's dim sizes = [N, H, W, C]
  auto input = st.regs.get<Tensor>(op.input());
  // weight's dim sizes = [F, H, W, C]
  auto weight = st.regs.get<Tensor>(op.weight());
  // bias: a 1-dim array whose size is F
  auto bias = st.regs.get<Tensor>(op.bias());
  // strides = [strides_y, strides_x]
  vector<Expr> strides = getFromArrayAttr<Index>(op.stride());
  // dilations = [dilations_y, dilations_x]
  vector<Expr> dilations = getFromArrayAttr<Index>(op.dilation());

  // Check whether C is identical
  st.wellDefined(op, input.getDim(3) == weight.getDim(3));
  // Check whether F is identical
  st.wellDefined(op, weight.getDim(0) == bias.getDim(0));

  assert(strides.size() == 2 && dilations.size() == 2);  

  auto elemTy = getElemTy(op.getResult());
  if (!elemTy.isa<mlir::FloatType>())
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto paddedTensor = getPaddedTensor2D(elemTy, input, op.pad());


  auto acc = paddedTensor.conv(weight,
                      strides, dilations, ShapedValue::ConvLayout::NHWC_FHWC);

  vector<Expr> outDims = acc.getDims();
  auto output = addBias2D(elemTy, outDims, acc, bias);

  st.wellDefined(op, input.isFullyInitialized());
  st.wellDefined(op, weight.isFullyInitialized());
  st.wellDefined(op, bias.isFullyInitialized());

  st.regs.add(op, output);

}

template<>
void encodeOp(State &st, mlir::tosa::TransposeOp op, bool) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  mlir::Value i = op.input1();
  mlir::Value p = op.perms();

  auto ity = i.getType().dyn_cast<mlir::RankedTensorType>();
  auto pty = p.getType().dyn_cast<mlir::RankedTensorType>();
  if(!getElemTy(p).isa<mlir::IntegerType>())
    throw UnsupportedException(op.getOperation(), "Unsupported element type");

  assert(pty.getRank() == 1 && pty.getDimSize(0) == ity.getRank());

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

  st.wellDefined(op, input.isFullyInitialized());
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
                    move(dims), move(indVars), output, Expr::mkBool(true)));

}

template<>
void encodeOp(State &st, mlir::tosa::GatherOp op, bool) {
  // values, output - 3D dimension, indices - 2D dimension.
  // These were checked by default MLIR verifier

  // input's dim sizes = [N, K, C]
  auto values = st.regs.get<Tensor>(op.values());
  // indices's dim sizes = [N, W]
  auto indices = st.regs.get<Tensor>(op.indices());
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
      inBounds.implies(move(idxInBounds) & move(inputInBounds))));
  st.wellDefined(op, Expr::mkForall(indVars,
      inBounds.implies(move(isInitialized))));
  st.wellDefined(op, indices.isFullyInitialized());

  st.regs.add(op, Tensor::mkInitializedLambda(
      values.getElemType(), move(outputDims), move(indVars),
      move(outputValue)));
}


template<>
void encodeOp(State &st, mlir::tensor::ExtractOp op, bool) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.

  auto t = st.regs.get<Tensor>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));
  if (indices.empty())
    // Deal with the zero-rank tensor case
    indices.push_back(Index(0));

  auto elem = t.get(indices);
  if (auto v = fromExpr(move(elem), op.getType()))
    st.regs.add(op, move(*v));
  else
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  st.wellDefined(op, t.isInBounds(indices));
  st.wellDefined(op, t.isInitialized(indices));
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
    Expr resExpr = newst.regs.getExpr(yieldedValues[i]);
    if (outputValMap)
      resExpr = (*outputValMap)(resExpr, outputIndVars);

    tvec_res->push_back(Tensor::mkInitializedLambda(yieldedValues[i].getType(),
        vector(tensorSz), vector(outputIndVars), resExpr));
  }
}

template<class T>
static void encodeConv(State &st, T op, ShapedValue::ConvLayout clayout) {
  vector<Expr> strides, dilations;
  // TODO: The result may not fit in Index::BITS
  for (auto s: op.strides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.dilations())
    dilations.push_back(Index(d.getSExtValue()));

  if (op.hasTensorSemantics()) {
    auto t_input = st.regs.get<Tensor>(op.image());
    auto t_filter = st.regs.get<Tensor>(op.filter());

    auto t_res = t_input.conv(t_filter, strides, dilations, clayout);
    st.regs.add(op.getResult(0), move(t_res));
    st.wellDefined(op, t_input.isFullyInitialized());
    st.wellDefined(op, t_filter.isFullyInitialized());
  } else {
    auto outputTy = op.outputs()[0].getType().template cast<mlir::MemRefType>();
    auto elemTy = outputTy.getElementType();
    auto input = st.regs.get<MemRef>(op.image());
    auto filter = st.regs.get<MemRef>(op.filter());
    auto output = st.regs.get<MemRef>(op.outputs()[0]);

    if (!output.isIdentityMap())
      throw UnsupportedException(op.getOperation(),
          "The output MemRef should have identity layout.");

    auto [indices, expr] = input.ShapedValue::conv(filter, strides, dilations,
        clayout);

    // we splat results into 1D memory layout
    auto idx = Index::var("outputIdx", VarType::BOUND);
    auto outputIndices = output.getLayout().getInverseIndices(idx);
    auto outputExpr = expr.substitute(indices, outputIndices);
    auto outputTensor = Tensor::mkInitializedLambda(elemTy,
        {output.get1DSize()}, {idx}, outputExpr);

    // store the result to the output reference
    storeTensorTo(st, op, move(outputTensor), output, outputTy, true);

    // Input & filter read check
    st.wellDefined(op,
        input.getLiveness() & input.isInBounds() & filter.getLiveness() &
        filter.isInBounds());
    // No alias checks between output and input/filter
    st.wellDefined(op, output.noalias(input) & output.noalias(filter));
  }
}

template<> void
encodeOp(State &st, mlir::linalg::DepthwiseConv2DNhwcHwcmOp op,
         bool encodeMemWriteOp) {
  if (!op.hasTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  vector<Expr> strides, dilations;

  for (auto s: op.strides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.dilations())
    dilations.push_back(Index(d.getSExtValue()));

  auto t_input = st.regs.get<Tensor>(op.image());
  auto t_filter = st.regs.get<Tensor>(op.filter());

  st.wellDefined(op, t_input.getDim(3) == t_input.getDim(3));
  st.wellDefined(op, t_filter.isFullyInitialized());

  auto t_res = t_input.depthwiseConv2D(t_filter, strides, dilations);
  st.regs.add(op.getResult(0), move(t_res));
  st.wellDefined(op, t_input.isFullyInitialized());
  st.wellDefined(op, t_filter.isFullyInitialized());
}

template<> void
encodeOp(State &st, mlir::linalg::Conv2DNchwFchwOp op, bool encodeMemWriteOp) {
  if (!op.hasTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  encodeConv(st, op, ShapedValue::ConvLayout::NCHW_FCHW);
}

template<> void
encodeOp(State &st, mlir::linalg::Conv2DNhwcHwcfOp op, bool encodeMemWriteOp) {
  if (!op.hasTensorSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation());

  encodeConv(st, op, ShapedValue::ConvLayout::NHWC_HWCF);
}

template<>
void encodeOp(State &st, mlir::linalg::InitTensorOp op, bool) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::RankedTensorType>();
  if (!ty || !Tensor::isTypeSupported(ty))
    throw UnsupportedException(op.getOperation(), "Unsupported tensor type");

  vector<Expr> sizes;
  if (ty.getRank() == 0) {
    sizes.push_back(Index(1));
  } else {
    for (unsigned i = 0; i < ty.getRank(); ++i) {
      if (op.isDynamicSize(i))
        sizes.push_back(st.regs.get<Index>(op.getDynamicSize(i)));
      else
        sizes.push_back(Index(op.getStaticSize(i)));
    }
  }

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  st.regs.add(res,
      Tensor::var(ty.getElementType(),
          ("init_tensor#") + to_string(new_var_idx++), sizes, false));
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

      if (resTy.getDimSize(i) != mlir::TensorType::kDynamicSize)
        st.wellDefined(op, size == resTy.getDimSize(i));
      newDims.push_back(move(size));
    }
  }

  st.wellDefined(op, t.get1DSize() == smt::get1DSize(newDims));
  st.regs.add(op.getResult(), t.reshape(newDims));
  // Note: tensor_collapse_shape does not look into elements, so initialization
  // check is not necessary.
}

template<>
void encodeOp(State &st, mlir::tensor::ExpandShapeOp op, bool) {
  Tensor t = st.regs.get<Tensor>(op.getOperand());

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
      if (op.getResultType().getDimSize(id) == mlir::TensorType::kDynamicSize) {
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
void encodeOp(State &st, mlir::linalg::MatmulOp op, bool) {
  if (!op.hasTensorSemantics())
    throw UnsupportedException(op.getOperation(),
        "tensor semantics is supported only");

  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    throw UnsupportedException(op.getOperation(),
        "unsupported form");

  if (getElemTy(op.getOperand(0)) != getElemTy(op.getOperand(1)) ||
      getElemTy(op.getOperand(0)) != getElemTy(op.getResult(0)))
    throw UnsupportedException(op.getOperation(),
        "unsupported types");

  Tensor a = st.regs.get<Tensor>(op.getOperand(0));
  Tensor b = st.regs.get<Tensor>(op.getOperand(1));
  Tensor result = a.matmul(b);
  st.wellDefined(op, a.isFullyInitialized());
  st.wellDefined(op, b.isFullyInitialized());
  st.regs.add(op.getResult(0), Tensor(result));
}

template<>
void encodeOp(State &st, mlir::linalg::PadTensorOp op, bool) {
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");

  auto &region = op.getRegion();
  if (!region.hasOneBlock())
    throw UnsupportedException(op.getOperation(), "Unsupported region");
  auto &blk = *region.getBlocks().begin();

  vector<Index> padSizeLow = getFromMixedOps<Index>(st, op.getMixedLowPad());
  vector<Index> padSizeHigh = getFromMixedOps<Index>(st, op.getMixedHighPad());

  auto sourceTensor = st.regs.get<Tensor>(op.source());
  auto newTensorSize =
      vecAdd(vecAdd(sourceTensor.getDimsAsIndices(), padSizeLow), padSizeHigh);

  State newst = st;
  auto loopUpperBound = vecAddElem(newTensorSize, Index(-1));
  newst.linalgGenericScopes.push(State::LinalgGenericScope{
      move(loopUpperBound)});
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

  st.regs.add(op.getResult(), move(tvec_res->front()));
  st.wellDefined(op, move(welldef));
  st.wellDefined(op, sourceTensor.isFullyInitialized());
}

static pair<Expr, Expr> encodeDimOp(
    State &st, vector<Expr> &&dims, mlir::Value index) {
  auto idx = st.regs.get<Index>(index);

  auto res = dims[0];
  for (unsigned i = 1; i < dims.size(); ++i)
    res = Expr::mkIte((Expr)idx == i, dims[i], res);

  return {move(res), ((Expr)idx).ult(dims.size())};
}

template<>
void encodeOp(State &st, mlir::tensor::DimOp op, bool) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<Tensor>(op.source()).getDims(), op.index());
  st.regs.add(op, Index(res));
  st.wellDefined(op, move(wf));
  // DimOp does not look into elements, so initialization check is not necessary
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
  st.regs.add(op, move(t));
  // Initialization check is not necessary
}

template<>
void encodeOp(State &st, mlir::tensor::InsertOp op, bool) {
  auto val = st.regs.getExpr(op.scalar());
  auto dest = st.regs.get<Tensor>(op.dest());

  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));
  if (indices.empty())
    indices.push_back(Index(0));

  auto [tensor, inbounds] = dest.insert(val, indices);
  st.regs.add(op, move(tensor));
  st.wellDefined(op, move(inbounds));
}

template<>
void encodeOp(State &st, mlir::tensor::FromElementsOp op, bool) {
  vector<Expr> elems;
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    elems.push_back(st.regs.getExpr(op.getOperand(i)));

  auto elemTy = op.getType().getElementType();
  st.regs.add(op.getResult(), Tensor(elemTy, move(elems)));
}

template<>
void encodeOp(State &st, mlir::tensor::GenerateOp op, bool) {
  auto exts = op.dynamicExtents();
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    throw UnsupportedException(op.getOperation(), "Unsupported type");
  auto *blk = op.getBody();
  if (!blk)
    throw UnsupportedException(op.getOperation(), "Unsupported form");

  vector<Index> upperbound;
  {
    int j = 0;
    for (int i = 0; i < retty.getRank(); ++i) {
      auto d = retty.getDimSize(i);
      if (d == mlir::ShapedType::kDynamicSize) {
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
    newst.linalgGenericScopes.push(State::LinalgGenericScope{move(upperbound)});
    for (int i = 0; i < blk->getNumArguments(); ++i) {
      Expr idxvar = newst.linalgGenericScopes.top().indVars[i];
      newst.regs.add(blk->getArgument(i), Index(idxvar));
    }

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        retty.getRank(), op.getContext());

    encodeParallelLoopBodyAndOutputs(newst, *blk, identityMap,
        tvec_res, welldef);

    auto &indVars = newst.linalgGenericScopes.top().indVars;

    // linalg::generate has one result
    welldef = Expr::mkForall(indVars,
        tvec_res->front().isInBounds(indVars).implies(welldef));

    newst.linalgGenericScopes.pop();
  }

  // linalg::generate has one result
  st.regs.add(op.getResult(), move(tvec_res->front()));
  st.wellDefined(op, move(welldef));
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
    st.wellDefined(op, move(cond));
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
  st.wellDefined(op, src.isFullyInitialized());
  st.regs.add(res,
      Tensor::mkInitializedLambda(src.getElemType(), move(dims), move(inIdxs),
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
    st.wellDefined(op, move(cond));
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
  Expr output = Expr::mkIte(cond, move(srcelem), move(tgtelem));

  // If tgt[indVars] is inbounds and the src[indVars] is to be chosen,
  // src[indVars] must be inbounds as well.
  st.wellDefined(op,
      Expr::mkForall(indVars, (tgtwb & cond).implies(srcwb)));
  // Since we are copying tgt into a new SSA register, tgt must be
  // initialized as well.
  st.wellDefined(op,
      Expr::mkForall(indVars, (tgtwb & !cond).implies(
        tgt.isInitialized(indVars))));

  st.regs.add(res, Tensor::mkInitializedLambda(
      src.getElemType(), move(dims), move(indVars), output));
  st.wellDefined(op, src.isFullyInitialized());
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

  if (op.shift() != 0)
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
  else if (op.quantization_info())
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
  auto input = op.input();   // [N, IC]
  auto weight = op.weight(); // [OC, IC]
  auto bias = op.bias();     // [OC]
  if (!input.getType().isa<mlir::RankedTensorType>() ||
      !weight.getType().isa<mlir::RankedTensorType>() ||
      !bias.getType().isa<mlir::RankedTensorType>())
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  auto inputTensor = st.regs.get<Tensor>(input);
  auto weightTensor = st.regs.get<Tensor>(weight);
  auto biasTensor = st.regs.get<Tensor>(bias);

  st.wellDefined(op, inputTensor.getDim(1) == weightTensor.getDim(1));
  st.wellDefined(op, weightTensor.getDim(0) == biasTensor.getDim(0));
  st.wellDefined(op, inputTensor.isFullyInitialized());
  st.wellDefined(op, weightTensor.isFullyInitialized());
  st.wellDefined(op, biasTensor.isFullyInitialized());

  auto mul = inputTensor.matmul(weightTensor, /*transposed*/true);

  // Output: [N, OC]
  auto idxVars = Index::boundIndexVars(2);
  vector<Expr> sizes = {inputTensor.getDim(0), weightTensor.getDim(0)};
  auto biasBroadcasted = biasTensor.affine(idxVars, {idxVars[1]}, move(sizes));

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
  auto input = op.input();
  auto inputTy = input.getType().dyn_cast<mlir::RankedTensorType>();
  if (!inputTy)
    throw UnsupportedException(op.getOperation(), "Unsupported operand type");

  auto t = st.regs.get<Tensor>(input);
  uint64_t axis = op.axis();

  st.wellDefined(op.getOperation(), t.isFullyInitialized());
  st.regs.add(op, t.sum(axis));
}

template<>
void encodeOp(State &st, mlir::tosa::ReshapeOp op, bool) {
  auto t = st.regs.get<Tensor>(op.getOperand());
  auto attrs = op.new_shape();
  vector<Expr> newDims;
  mlir::Operation *oper = op.getOperation();

  for (auto a: attrs) {
    auto ia = a.cast<mlir::IntegerAttr>();
    if (ia.getInt() == -1)
      throw UnsupportedException(oper, "Dynamic shape is unsupported");
    newDims.push_back(Index(ia.getInt()));
  }
  st.wellDefined(oper, t.get1DSize() == smt::get1DSize(newDims));
  st.regs.add(op.getResult(), t.reshape(newDims));
  // Reshape does not look into tensor's elements, so init check is not
  // necessary.
}

static MemRef createNewLocalBlk(
    Memory *m, vector<Expr> &&dims, mlir::MemRefType memrefTy, bool writable) {
  if (!MemRef::isTypeSupported(memrefTy))
    throw UnsupportedException("unsupported element type");

  auto layout = MemRef::getLayout(memrefTy, dims);
  // Add a new local block
  auto bid = m->addLocalBlock(smt::get1DSize(dims),
      memrefTy.getElementType(), Expr::mkBool(writable));
  // Create MemRef which points to the newly created block
  auto memref =
      MemRef(m, memrefTy.getElementType(), bid, Index::zero(), dims,
          move(layout), /*is not a view reference*/Expr::mkBool(false));

  return {move(memref)};
}

template<>
void encodeOp(State &st, mlir::memref::AllocOp op, bool) {
  auto memrefTy = op.getType().cast<mlir::MemRefType>();
  if (!memrefTy.getLayout().isIdentity())
    throw UnsupportedException(op.getOperation(),
        "unsupported memref type for alloc: it has a non-identity layout map");

  auto dsizes = op.dynamicSizes();
  vector<Expr> dszExprs;
  for (const auto &sz: dsizes) {
    dszExprs.push_back(st.regs.get<Index>(sz));
  }
  auto dims = ShapedValue::getDims(memrefTy, false, move(dszExprs));

  auto memref = createNewLocalBlk(st.m.get(), move(dims), memrefTy, true);
  st.regs.add(op, move(memref));
}

template<>
void encodeOp(State &st, mlir::memref::DimOp op, bool) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<MemRef>(op.source()).getDims(), op.index());
  st.regs.add(op, Index(res));
  st.wellDefined(op, move(wf));
}

template<>
void encodeOp(State &st, mlir::memref::LoadOp op, bool) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto [val, info] = m.getWithAccessInfo(indices);
  if (auto vt = fromExpr(move(val), op.getType())) {
    st.regs.add(op, move(*vt));
    st.wellDefined(op, info.checkRead());
  } else
    throw UnsupportedException(op.getOperation(), "unsupported type");
}

template<>
void encodeOp(State &st, mlir::memref::GetGlobalOp op, bool encodeMemWriteOp) {
  auto name = op.name().str();
  auto bid = Expr::mkBV(st.m->getBidForGlobalVar(name), st.m->getBIDBits());
  auto type = op.getType();
  assert(type.getLayout().isIdentity() &&
      "don't know how to deal with get_global with non-identity layout");
  auto dims = ShapedValue::getDims(type, /*unknown sz is crash*/false);
  MemRef::Layout identityLayout(dims);

  MemRef newref(st.m.get(), type.getElementType(), bid, Index(0), dims,
      identityLayout, Expr::mkBool(false));
  st.regs.add(op, move(newref));
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
  for (auto idx0: op.indices())
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
  auto src = st.regs.get<MemRef>(op.source());
  int rankDiff = op.getSourceType().getRank() - op.getType().getRank();
  assert(rankDiff >= 0); // only reducing rank is allowed

  // This reduction logic mainly from MLIR SubViewOp verify function.
  // See 'Dialect/MemRef/IR/MemRefOps.cpp'.
  auto expectedType = mlir::memref::SubViewOp::inferResultType(
      op.getSourceType(), extractFromI64ArrayAttr(op.static_offsets()),
      extractFromI64ArrayAttr(op.static_sizes()),
      extractFromI64ArrayAttr(op.static_strides()));

  auto originalShapedType = expectedType.cast<mlir::ShapedType>();
  auto candidateReducedShapedType = op.getType().cast<mlir::ShapedType>();
  auto optionalUnusedDimsMask = mlir::computeRankReductionMask(
    originalShapedType.getShape(),
    candidateReducedShapedType.getShape()
  );

  if (!optionalUnusedDimsMask.hasValue())
    throw UnsupportedException(op.getOperation(),
        "Subview result size mismatch");

  auto unusedDims = optionalUnusedDimsMask.getValue();
  auto memref = src.subview(offsets, sizes, strides, unusedDims, rankDiff);
  st.regs.add(op.getResult(), move(memref));
}

static void storeTensorTo(
    State &st, mlir::Operation *op, Tensor &&tensor, const MemRef &memref,
    mlir::MemRefType memrefTy, bool ubIfReadOnly) {
  // Accessing uninitialized elem is UB.
  st.wellDefined(op, tensor.isFullyInitialized());

  if (memrefTy.getLayout().isIdentity()) {
    // memref with identity map
    auto success = st.m->storeArray(memrefTy.getElementType(),
        tensor.asArray(), memref.getBID(), memref.getOffset(),
        tensor.get1DSize());
    st.wellDefined(op, success.checkWrite(!ubIfReadOnly));

  } else {
    // TODO: can we further optimize this if we know that memref is a
    // freshly created block?
    // We may not need to preserve the 'previous' bytes.

    vector<Expr> idxs = Index::boundIndexVars(memrefTy.getRank());
    auto tVal = tensor.get(idxs);
    auto tInBounds = tensor.isInBounds(idxs);
    auto tInit = tensor.isInitialized(idxs);
    auto [mVal, mInfo] = memref.getWithAccessInfo(idxs);

    st.wellDefined(op, Expr::mkForall(idxs, tInBounds.implies(tInit)));
    st.wellDefined(op, Expr::mkForall(idxs,
        tInBounds.implies(mInfo.checkWrite())));

    // NOTE: this will be always false if mVal and tVal store unequal constants.
    // Therefore, we can't encode this condition as UB because
    // (1) If they are in src: src always becomes UB (false negative)
    // (2) If they are in tgt: tgt always becomes UB (false positive)
    // Therefore, encode this as precondition; precondition does not introduce
    // false negative, if properly handled (not implemented yet; see how Alive2
    // does it).
    st.addPrecondition(Expr::mkForall(idxs,
        tInBounds.implies(mVal == tVal)));
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

    auto idx = Index::var("loadidx", VarType::BOUND);
    return Tensor::mkLambdaFrom1D(elemTy, memref.getDims(),
        move(idx), arr.select(idx), Expr::mkBool(true));

  } else {
    vector<Expr> idxs = Index::boundIndexVars(memrefTy.getRank());
    auto [val, info] = memref.getWithAccessInfo(idxs);

    st.wellDefined(op, Expr::mkForall(idxs,
        fitsInDims(idxs, memref.getDims()).implies(info.checkRead())));
    st.hasQuantifier = true;

    return Tensor::mkInitializedLambda(elemTy, memref.getDims(),
        move(idxs), move(val));
  }
}
template<>
void encodeOp(State &st, mlir::bufferization::ToMemrefOp op,
    bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto tensor = st.regs.get<Tensor>(op.getOperand());
  auto memrefTy = op.memref().getType().cast<mlir::MemRefType>();
  auto dims = tensor.getDims();

  // Create a read-only block.
  auto memref = createNewLocalBlk(st.m.get(), move(dims), memrefTy, false);
  storeTensorTo(st, op.getOperation(), move(tensor), memref, memrefTy, false);
  st.regs.add(op.memref(), move(memref));
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
  auto memref = createNewLocalBlk(st.m.get(), move(dims), srcTy, false);
  storeTensorTo(st, op.getOperation(), move(tensor), memref, srcTy, false);
  // Src is not writable as well.
  st.m->setWritable(srcTy.getElementType(), src.getBID(), false);
  st.regs.add(op, move(memref));
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
  st.wellDefined(op, src.getLiveness());

  // According to the MLIR specification doc:
  // The dealloc operation should not be called on memrefs which alias an
  // alloc’d memref (e.g. memrefs returned by view operations).
  st.wellDefined(op, !src.isViewReference());

  // Unlike free(), we don't need to check offset == 0 because MemRef tracks
  // the pointer to the data buffer as allocated, referred to as
  // "allocated pointer". This is useful for deallocating the memref.
  // See: https://mlir.llvm.org/docs/TargetLLVMIR/ , Ranked MemRef Types sec.

  st.m->setLivenessToFalse(srcTy.getElementType(), src.getBID());
}

template<>
void encodeOp(State &st, mlir::memref::TensorStoreOp op, bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto t = st.regs.get<Tensor>(op.tensor());
  auto m = st.regs.get<MemRef>(op.memref());

  // Src and tgt's shapes & element types must match
  // Memref may have its layout, though.
  for (unsigned i = 0; i < t.getRank(); ++i)
    st.wellDefined(op, (Expr)t.getDim(i) == (Expr)m.getDim(i));

  storeTensorTo(st, op.getOperation(), move(t), m,
      op.memref().getType().cast<mlir::MemRefType>(), true);
}

template<>
void encodeOp(State &st, mlir::linalg::CopyOp op, bool encodeMemWrite) {
  if (!encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");
  else if (op.inputPermutation() || op.outputPermutation())
    // Well, this might be straightforward...
    throw UnsupportedException(op.getOperation(),
        "linalg.copy with permutations is not supported");

  auto *opr = op.getOperation();
  auto mrIn = st.regs.get<MemRef>(op.input());
  auto mrOut = st.regs.get<MemRef>(op.output());

  // Src and tgt's shapes & element types must match
  for (unsigned i = 0; i < mrIn.getRank(); ++i)
    st.wellDefined(opr, (Expr)mrIn.getDim(i) == (Expr)mrOut.getDim(i));

  // They must not overlap, according to
  // https://mlir.llvm.org/docs/Dialects/Linalg/#linalgcopy-mlirlinalgcopyop
  st.wellDefined(opr, mrIn.noalias(mrOut));

  auto loadedTensor = loadTensor(st, op, mrIn,
      op.input().getType().cast<mlir::MemRefType>());

  storeTensorTo(st, opr, move(loadedTensor), mrOut,
      op.output().getType().cast<mlir::MemRefType>(), true);
}

template<>
void encodeOp(State &st, mlir::linalg::FillOp op, bool encodeMemWrite) {
  if (op.hasBufferSemantics() && !encodeMemWrite)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");
  if (op.getNumResults() > 1)
    throw UnsupportedException(op.getOperation(),
        "it has multiple results");

  auto elemval = st.regs.getExpr(op.getOperand(0));
  auto op1 = op.getOperand(1);
  auto ety = getElemTy(op1);

  if (op.hasTensorSemantics()) {
    auto t = st.regs.get<Tensor>(op1);
    auto filled = Tensor(ety, move(elemval), t.getDims());
    st.regs.add(op.getResult(0), move(filled));
  } else {
    assert(op.hasBufferSemantics());
    auto m = st.regs.get<MemRef>(op1);
    auto filled = Tensor(ety, move(elemval), m.getDims());
    storeTensorTo(st, op.getOperation(), move(filled), m,
        op1.getType().cast<mlir::MemRefType>(), true);
  }
}

template<>
void encodeOp(State &st, mlir::linalg::DotOp op, bool encodeMemWrite) {
  if (!op.hasTensorSemantics())
    throw UnsupportedException(op.getOperation(),
        "tensor semantics is supported only");

  if (op.getNumResults() != 1)
    throw UnsupportedException(op.getOperation(),
        "it has multiple results");

  auto inputOps = op.getInputOperands();
  auto outputTy = op.getType(0).dyn_cast<mlir::TensorType>();

  auto outputDim = ShapedValue::getDims(outputTy, false);
  if (outputDim.size() != 1)
    throw UnsupportedException(op.getOperation(),
        "unknown dot format; shouldn't the result tensor have one element?");

  if (outputTy.getElementType() !=
      inputOps[0]->get().getType().dyn_cast<mlir::TensorType>()
          .getElementType())
    throw UnsupportedException(op.getOperation(), "casting is not supported");

  auto t1 = st.regs.get<Tensor>(inputOps[0]->get());
  auto t2 = st.regs.get<Tensor>(inputOps[1]->get());
  st.wellDefined(op, t1.isFullyInitialized());
  st.wellDefined(op, t2.isFullyInitialized());
  st.wellDefined(op, t1.get1DSize() == t2.get1DSize());

  auto res = t1.dot(t2);
  st.regs.add(op.getResult(0),
      Tensor(t1.getElemType(), move(res), move(outputDim)));
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
  st.wellDefined(op, tt.isFullyInitialized());
  st.regs.add(op, move(tt));
}

vector<Index> findLoopBounds(State &st, mlir::linalg::GenericOp op) {
  // The size of the loop is calculated (analogous to what
  // LinalgOp::createLoopRanges does).
  // The process of getting the size of the loop seems fishy;
  // LinalgOp::createLoopRanges relies on the "first" dimension that is
  // matched, and it isn't clear what happens if there are multiple matching
  // dimensions. For example,
  //   linalg.generic {
  //      indexing_maps = [affine_map<(n) -> (n)>,
  //                       affine_map<(n) -> (n)>,
  //                       affine_map<(n) -> (n)>] }
  //      ins(%A, %B: <?xf32>, <?xf32>) outs(%C: <?xf32>) { .. }
  // The size of the loop is either %A, %B, or %C's dimension, but the current
  // algorithm mandates the result to be %A's dimension.

  vector<Index> viewSizes;
  for (auto *opOperand : op.getInputAndOutputOperands()) {
    unsigned r = op.getRank(opOperand);
    if (!r)
      continue;

    if (opOperand->get().getType().isa<mlir::TensorType>()) {
      auto t = st.regs.get<Tensor>(opOperand->get());
      for (int64_t i = 0, e = r; i < e; ++i) {
        viewSizes.push_back(t.getDim(i));
      }
    } else if (opOperand->get().getType().isa<mlir::MemRefType>()) {
      auto t = st.regs.get<MemRef>(opOperand->get());
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
    res_ordered.push_back(move(res[resFilled[i]]));

  return res_ordered;
}

static void
encodeUBForTensorShapeMatch(State &st, mlir::linalg::GenericOp op,
                            const vector<Index> &indVarBounds) {
  mlir::AffineMap map = op.getLoopsToShapesMap();
  unsigned numRes = map.getNumResults();

  vector<Index> viewSizes;
  for (auto *opOperand : op.getInputAndOutputOperands()) {
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
    st.wellDefined(op, move(*ae == size));
  }
}

static void initInputStateForLoopBody(
    State &st, mlir::linalg::GenericOp op, Expr &welldef,
    bool isParallelLoop) {
  auto indexingMaps = op.indexing_maps().getValue();
  auto &block = *op.region().begin();

  const vector<Expr> &inductionVars = st.linalgGenericScopes.top().indVars;

  assert(op.getInputOperands().size() + op.getNumOutputs() ==
         indexingMaps.size());
  assert(op.getNumInputs() == op.getInputOperands().size());

  // The output variables contain the initial value of the tensor
  //   (see github issue #164)
  // For parallel loops: whole iterations contain the initial value
  // For reduction loops: only the first iteration contains the value
  size_t upperbound = op.getNumInputs() + op.getNumOutputs();

  for (size_t arg_i = 0; arg_i < upperbound; ++arg_i) {
    auto indexMap = indexingMaps[arg_i].cast<mlir::AffineMapAttr>().getValue();
    mlir::Value op_i = arg_i >= op.getNumInputs() ?
        op.getOutputOperand(arg_i - op.getNumInputs())->get() :
        op.getInputOperand(arg_i)->get();
    bool isInput = arg_i < op.getNumInputs();
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
        if (isInput || isOutputAndHasUse)
          welldef &= t_input.isFullyInitialized();
      } else {
        vector<Expr> indices;
        for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
          auto ae_res =
              encodeAffineExpr(indexMap.getResult(i), inductionVars, {});
          if (!ae_res) {
            string msg;
            TO_STRING(msg, "Unsupported affine expr: "<< indexMap.getResult(i));
            throw UnsupportedException(op.getOperation(), move(msg));
          }

          indices.emplace_back(move(*ae_res));
        }

        // The out-of-bounds checking is done when encoding loop bounds.
        auto t_elem = t_input.get(indices);
        st.regs.add(block.getArgument(arg_i), t_elem, elemty);

        // Reading uninitialized elements is UB.
        // For output variables, encode uninitialized if it syntactically has
        // uses.
        // This is a workaround (overapproximation) for not introducing a
        // 'poison' value.
        if (isInput || isOutputAndHasUse)
          welldef &= t_input.isInitialized(indices);
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
          throw UnsupportedException(op.getOperation(), move(msg));
        }

        indices.emplace_back(move(*ae_res));
      }

      // Reading uninitialized elements is UB.
      // For output variables, encode uninitialized if it syntactically has
      // uses.
      // This is a workaround (overapproximation) for not introducing a
      // 'poison' value.
      auto [m_elem, m_welldef] = m_input.getWithAccessInfo(indices);
      if (isInput)
        welldef &= m_welldef.checkRead();
      else
        welldef &= isOutputAndHasUse ?
            m_welldef.checkReadWrite() : m_welldef.checkWrite();
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
    Expr &welldef) {
  // Deal with simple reduction loops.
  // TODO: support more kinds of reduction loops!
  string errmsg = "permutated output map or simple reduction form is"
                  " supported only";
  mlir::Operation *the_op = block.getParentOp();

  auto &ops = block.getOperations();
  int instcount = ops.size();
  mlir::Value yieldedValue;

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
    throw UnsupportedException(the_op, move(errmsg));

  auto sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(idx);

  // TODO: deal with merging memories
  encodeBlock(newst, block, /*print ops*/false, /*encode mem writes*/false,
      [&yieldedValue, instcount, &lastarg, &the_op](
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
            throw UnsupportedException(the_op, move(msg));
          }
        }

        return false;
      },
      [&welldef, &newst](mlir::Operation *op) {
        welldef &= newst.isOpWellDefined(op);
      });

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
    t_res = Tensor(t_v.getElemType(), t_v.sum(),
          makeCube(Index(1), outputType.getRank()));
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
        throw UnsupportedException(the_op, move(errmsg));
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

    auto tensorSz = addOne(doMap(linalgInfo.indVarUpperBounds, outputMap));
    auto t_sum = Tensor::mkInitializedLambda(
          t_v.getElemType(),
          addOne(move(boundsForRes)),
          move(indVarsForRes),
          t_v.get(linalgInfo.indVars))
        .sum();

    auto outputIndVars = doMap(linalgInfo.indVars, outputMap);
    t_res = Tensor::mkInitializedLambda(
        t_v.getElemType(), move(tensorSz), move(outputIndVars), t_sum);
  }
}

template<>
void encodeOp(State &st, mlir::linalg::GenericOp op, bool encodeMemWriteOp) {
  if (!(op.hasTensorSemantics() || op.hasBufferSemantics()))
    throw UnsupportedException(op.getOperation(),
        "tensor/buffer semantics is supported only");

  else if (op.hasBufferSemantics() && !encodeMemWriteOp)
    throw UnsupportedException(op.getOperation(),
        "We do not support memory writes in this scope");

  auto &region = op.region();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(op.getOperation(),
        "a single block is supported only");

  auto &block = region.front();
  if (!std::all_of(block.args_begin(), block.args_end(),
      [](auto &arg) { return arg.getType().isSignlessIntOrFloat(); }))
    throw UnsupportedException(op.getOperation(),
        "unsupported block arguments");

  if (llvm::any_of(op.iterator_types(), [](mlir::Attribute attr) {
    auto str = attr.cast<mlir::StringAttr>().getValue();
    return str != mlir::getParallelIteratorTypeName() &&
           str != mlir::getReductionIteratorTypeName() &&
           str != mlir::getWindowIteratorTypeName();
  }))
    throw UnsupportedException(op.getOperation(),
        "unsupported iterator type");

  // Find the inclusive upper bounds
  auto loopBounds = findLoopBounds(st, op);

  encodeUBForTensorShapeMatch(st, op, loopBounds);

  // Start from newst
  optional<vector<Tensor>> tvec_res;
  optional<Expr> t_welldef;
  {
    Expr welldef = Expr::mkBool(true);
    State newst = st;
    newst.linalgGenericScopes.push(State::LinalgGenericScope{loopBounds});

    auto indexingMaps = op.indexing_maps().getValue();
    auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
    bool isParallelLoop = outputMap.isPermutation();

    initInputStateForLoopBody(newst, op, welldef, isParallelLoop);

    auto &indVars = newst.linalgGenericScopes.top().indVars;

    if (isParallelLoop) {
      encodeParallelLoopBodyAndOutputs(newst, block, outputMap,
          tvec_res, welldef);

    } else {
      // Reduction loops returning multiple values is not supported by MLIR-TV
      // yet.
      if (op.getNumOutputs() > 1)
        throw UnsupportedException(op.getOperation(),
            "unsupported reduction form");

      optional<Tensor> t_res;
      auto outputType = op.getOutputOperand(0)->get().getType()
          .cast<mlir::ShapedType>();
      encodeReductionLoopBodyAndOutput(newst, block,
            indexingMaps, outputType, t_res, welldef);
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
    t_welldef = Expr::mkForall(indVars, inbounds.implies(welldef));
  }


  st.wellDefined(op, move(*t_welldef));

  if (op.hasTensorSemantics()) {
    for(unsigned i = 0; i < tvec_res->size(); i++) {
      // NOTE: op's output tensor (op.getOutputOperand()[0]->get())
      // isn't updated;
      // aqjune talked with mlir people and confirmed
      st.regs.add(op.getResult(i), move(tvec_res->at(i)));
    }
  } else if (op.hasBufferSemantics()) {
    for(unsigned i = 0; i < tvec_res->size(); i++) {
      auto opi = op.getOutputOperand(i)->get();
      auto m_res = st.regs.get<MemRef>(opi);
      storeTensorTo(st, op, move(tvec_res->at(i)), m_res,
          opi.getType().cast<mlir::MemRefType>(), true);
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
      st.regs.add(r, move(i));

    } else if (auto fty = ty.dyn_cast<mlir::FloatType>()) {
      st.regs.add(r, *getIdentity(fty), fty);

    } else if (auto tty = ty.dyn_cast<mlir::RankedTensorType>()) {
      auto dims = ShapedValue::getDims(tty);
      auto elemTy = tty.getElementType();
      Tensor t(elemTy, *getIdentity(elemTy), move(dims));
      st.regs.add(r, move(t));

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

    ENCODE(st, op, mlir::AffineApplyOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::SelectOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::ReturnOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::arith::AddFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::AddIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::CmpFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::CmpIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantFloatOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantIndexOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantIntOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ConstantOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::ExtFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::IndexCastOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::MulFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::MulIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::DivFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::NegFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SubFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::SubIOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::TruncFOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::arith::XOrIOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::bufferization::CloneOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::bufferization::ToMemrefOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::bufferization::ToTensorOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::math::AbsOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::math::ExpOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::memref::AllocOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::DeallocOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::DimOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::LoadOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::GetGlobalOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::StoreOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::SubViewOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::memref::TensorStoreOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::linalg::DepthwiseConv2DNhwcHwcmOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::Conv2DNchwFchwOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::Conv2DNhwcHwcfOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::CopyOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::DotOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::FillOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::GenericOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::IndexOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::InitTensorOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::MatmulOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::linalg::PadTensorOp, encodeMemWriteOps);
    
    ENCODE(st, op, mlir::shape::ShapeOfOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::shape::ToExtentTensorOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::sparse_tensor::ConvertOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::tensor::CastOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::CollapseShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::DimOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExpandShapeOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::InsertOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExtractOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::ExtractSliceOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::FromElementsOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::GenerateOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tensor::InsertSliceOp, encodeMemWriteOps);

    ENCODE(st, op, mlir::tosa::AbsOp, encodeMemWriteOps);
    ENCODE(st, op, mlir::tosa::AddOp, encodeMemWriteOps);
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

void encode(State &st, mlir::FuncOp &fn, bool printOps) {
  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  auto &block = region.front();

  encodeBlock(st, block, printOps, true/*allow mem ops*/, {}, {});
}
