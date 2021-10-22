#include "encode.h"
#include "utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
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


#define RET_STR(V) { \
  string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << V; \
  rso.flush(); \
  return msg; \
}
#define RET_STR_WITH_PREFIX(PREFIX, V) { \
  string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << PREFIX << V; \
  rso.flush(); \
  return msg; \
}


static optional<ValueTy> attrToValueTy(mlir::Attribute a) {
  auto ty = a.getType();
  if (ty.isa<mlir::FloatType>()) {
    return Float::constant(a.dyn_cast<mlir::FloatAttr>().getValue(), ty);
  } else if (ty.isa<mlir::IntegerType>()) {
    if (64 < ty.getIntOrFloatBitWidth())
      // size is too large
      return {};

    return Integer(a.dyn_cast<mlir::IntegerAttr>().getValue());
  } else if (ty.isa<mlir::IndexType>()) {
    llvm::APInt i = a.dyn_cast<mlir::IntegerAttr>().getValue();
    assert(i.getBitWidth() == 64);
    int64_t ii = i.getSExtValue();
    assert(-2147483648ll <= ii && ii <= 2147483647ll);
    return Index(ii);
  }
  return {};
}

static optional<ValueTy> fromExpr(Expr &&e, mlir::Type ty) {
  if (ty.isa<mlir::IndexType>())
    return Index(e);
  else if (ty.isa<mlir::FloatType>())
    return Float(e, ty);
  else if (ty.isa<mlir::IntegerType>()) {
    assert(e.sort().bitwidth() == ty.getIntOrFloatBitWidth());
    return Integer(e);
  }
  return {};
}

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

static Expr evalIndexCastOp(mlir::Type src, mlir::Type tgt, Expr &&val) {
  assert(val.sort().isBV());

  unsigned srcWidth = val.sort().bitwidth();

  unsigned destWidth = 0;
  if (auto dstty = tgt.dyn_cast<mlir::IntegerType>())
    destWidth = dstty.getWidth();
  else {
    assert(tgt.isa<mlir::IndexType>());
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

static mlir::Type getTensorElemTy(mlir::Value v) {
  return v.getType().dyn_cast<mlir::TensorType>().getElementType();
}


static optional<pair<Tensor, Tensor>>
broadcastTensors(State &st, mlir::Value arg0, mlir::Value arg1) {
  // reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
  auto ty0 = arg0.getType().cast<mlir::RankedTensorType>();
  auto ty1 = arg1.getType().cast<mlir::RankedTensorType>();
  auto ty0rank = ty0.getRank(), ty1rank = ty1.getRank();

  auto resRank = max(ty0rank, ty1rank);
  auto inVars0 = Index::boundIndexVars(resRank);
  auto inVars1 = Index::boundIndexVars(resRank);
  Expr izero = Index(0);

  vector<Expr> outVars0, outVars1, resDims;
  for (int64_t i = 0; i < min(ty0rank, ty1rank); i++) {
    int64_t idx0 = ty0rank - 1 - i;
    int64_t idx1 = ty1rank - 1 - i;

    auto d1 = ty0.getDimSize(idx0);
    auto d2 = ty1.getDimSize(idx1);

    bool dyn0 = d1 == mlir::ShapedType::kDynamicSize;
    bool dyn1 = d2 == mlir::ShapedType::kDynamicSize;
    if (dyn0 ^ dyn1)
      return nullopt;

    assert(d1 == 1 || d2 == 1 || d1 == d2);
    resDims.insert(resDims.begin(), Index(max(d1,d2)));
    outVars0.insert(outVars0.begin(), d1 == 1 ? izero : inVars0[idx0]);
    outVars1.insert(outVars1.begin(), d2 == 1 ? izero : inVars1[idx1]);
  }

  if (ty0rank < ty1rank) {
    for (int64_t i = ty1rank - ty0rank - 1; i >= 0; --i) {
      resDims.insert(resDims.begin(), Index(ty1.getDimSize(i)));
      outVars1.insert(outVars1.begin(), inVars1[i]);
    }
  } else if (ty1rank < ty0rank) {
    for (int64_t i = ty0rank - ty1rank - 1; i >= 0; --i) {
      resDims.insert(resDims.begin(), Index(ty0.getDimSize(i)));
      outVars0.insert(outVars0.begin(), inVars0[i]);
    }
  }

  auto resDims2 = resDims;
  auto t0 = st.regs.get<Tensor>(arg0);
  auto t1 = st.regs.get<Tensor>(arg1);
  auto m0 = Tensor::mkLambda(t0.getElemType(), move(resDims), move(inVars0),
                              t0.get(outVars0).first);

  auto m1 = Tensor::mkLambda(t1.getElemType(), move(resDims2), move(inVars1),
                              t1.get(outVars1).first);

  return {{m0, m1}};
}

template<class OpTy>
static optional<string>
encodeBinaryOp(State &st, OpTy op, mlir::Value arg0, mlir::Value arg1,
    function<Float(Float &&e1, Float &&e2)> f_float,
    function<Integer(Integer &&e1, Integer &&e2)> f_int) {

  if (arg0.getType().isa<mlir::FloatType>()) {
    auto a = st.regs.get<Float>(arg0);
    auto b = st.regs.get<Float>(arg1);
    st.regs.add(op, f_float(move(a), move(b)));

  } else if (auto tty = arg0.getType().dyn_cast<mlir::RankedTensorType>()) {
    auto elemty = tty.getElementType();
    if (!elemty.isIntOrFloat())
      return "Unsupported element type";

    auto bts = broadcastTensors(st, arg0, arg1);
    if (!bts)
      return "Unsupported broadcast form";
    auto [a, b] = *bts;

    auto f = [&](smt::Expr &&a, smt::Expr &&b) -> smt::Expr {
      if (elemty.isa<mlir::FloatType>()) {
        return f_float(Float(a, elemty), Float(b, elemty));
      } else if (elemty.isa<mlir::IntegerType>()) {
        return f_int(Integer(a), Integer(b));
      }
      llvm_unreachable("Unreachable case");
    };
    auto [res, welldef] = a.elementwiseBinOp(b, f);
    st.regs.add(op, move(res));
    st.wellDefined(op.getOperation(), move(welldef));

  } else {
    return "Unsupported type";
  }
  return {};
}

#define ENCODE(st, op, ty) { \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    try { \
      auto errmsg = encodeOp(st, op2); \
      if (errmsg) { \
        RET_STR("Unknown op (" << op.getName() << "): " << op << "\n\t" << *errmsg << "\n") \
      } \
    } catch (UnsupportedException& e) { \
      RET_STR("Unknown op (" << op.getName() << "): " << op << "\n\t" << e.what() << "\n") \
    } \
    continue; \
  } \
}

template<class T>
static optional<string> encodeOp(State &st, T op);

template<>
optional<string> encodeOp(State &st, mlir::arith::AddFOp op) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b); },
      [](auto &&a, auto &&b) { return (Expr)a + (Expr)b; });
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::MulFOp op) {
  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.mul(b); },
      [](auto &&a, auto &&b) { return (Expr)a * (Expr)b; });
  return {};
}

static void addIntOrIndex(
    State &st, mlir::Value res, const Expr &e, bool isIndex) {
  if (isIndex)
    st.regs.add(res, Index(e));
  else
    st.regs.add(res, Integer(e));
}

template<>
optional<string> encodeOp(State &st, mlir::arith::AddIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a + b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::SubIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a - b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::MulIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a * b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::ConstantIndexOp op) {
  st.regs.add(op, Index(op.value()));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::ConstantIntOp op) {
  st.regs.add(op, Integer(op.value(), op.getType().getIntOrFloatBitWidth()));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::ConstantFloatOp op) {
  if (Float::sort(op.getType()) == nullopt)
    return "unsupported constant type";

  auto fp = op.value();
  st.regs.add(op, Float::constant(fp, op.getType()));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::ConstantOp op) {
  auto attr = op.value();
  if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    // splat
    if (!denseAttr.isSplat())
      return "a fp splat constant tensor is supported only";
    if (!op.getType().isa<mlir::TensorType>())
      return "unsupported constant type";

    // A constant tensor's type cannot have unknown dimensions
    auto tensorty = op.getType().cast<mlir::TensorType>();
    auto dims = ShapedValue::getDims(tensorty, false);
    auto v = attrToValueTy(denseAttr.getSplatValue());
    if (!v)
      return "unsupported constant";

    st.regs.add(op,
        Tensor(tensorty.getElementType(), getExpr(*v), move(dims)));
    return {};

  } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    auto v = attrToValueTy(intAttr);
    if (!v)
      return "unsupported constant";

    st.regs.add(op, move(*v));
    return {};

  } else if (auto sparseAttr = attr.dyn_cast<mlir::SparseElementsAttr>()) {
    mlir::ShapedType sparseType = sparseAttr.getType();
    if (!sparseType.isa<mlir::TensorType>())
      return "unsupported type";

    auto sparseIndexValues = sparseAttr.getIndices().getValues<uint64_t>();
    auto elemTy = sparseType.getElementType();
    auto rank = sparseType.getRank();
    vector<uint64_t> dims;
    for (unsigned i = 0; i < rank; ++i)
      dims.push_back(sparseType.getDimSize(i));

    // Unspecified locations are filled with zero.
    auto zero = getZero(elemTy);
    if (!zero)
      return "unsupported element type";

    vector<vector<uint64_t>> sparseIndices;
    vector<Expr> sparseValues;

    auto sparseIndBeg = sparseIndexValues.begin();
    while (sparseIndBeg != sparseIndexValues.end()) {
      vector<uint64_t> curIndices;
      for (unsigned i = 0; i < rank; ++i) {
        curIndices.push_back(*sparseIndBeg);
        sparseIndBeg++;
      }

      auto value = sparseAttr.getValue(curIndices);
      sparseIndices.push_back(move(curIndices));

      auto e = attrToValueTy(value);
      if (!e)
        return "unsupported element";
      sparseValues.push_back(getExpr(*e));
    }
    st.hasConstArray = true;
    st.regs.add(op, Tensor(elemTy, sparseIndices, sparseValues, dims, *zero));
    return {};
  }
  return "unsupported constant";
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::IndexOp op) {
  uint64_t i = op.dim();
  assert(i < st.linalgGenericScopes.top().indVars.size());
  Expr idxvar = st.linalgGenericScopes.top().indVars[i];
  st.regs.add(op, Index(idxvar));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::math::AbsOp op) {
  auto f = st.regs.get<Float>(op.getOperand());
  st.regs.add(op.getResult(), f.abs());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::arith::IndexCastOp op) {
  auto srcty = op.getOperand().getType();
  auto dstty = op.getType();

  if (auto src_tensorty = srcty.dyn_cast<mlir::TensorType>()) {
    auto dst_tensorty = dstty.dyn_cast<mlir::TensorType>();
    if (!dst_tensorty)
      return "Unknown type";

    auto src = st.regs.get<Tensor>(op.getOperand());
    auto dst_elemty = dst_tensorty.getElementType();
    auto res = src.elementwiseUnaryOp(dst_elemty, [&](auto &&e) {
      return evalIndexCastOp(src_tensorty.getElementType(),
          dst_elemty, move(e));
    });
    st.regs.add(op, move(res));

  } else {
    auto src = st.regs.getExpr(op.getOperand());
    st.regs.add(op, Integer(evalIndexCastOp(srcty, dstty, move(src))));
  }
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::AffineApplyOp op) {
  auto m = op.getAffineMap();
  if (m.getNumResults() != 1)
    return "num results is larger than one";

  auto dimOperands = op.mapOperands().take_front(m.getNumDims());
  auto symbolOperands = op.mapOperands().take_back(m.getNumSymbols());

  vector<Index> indices, symbols;
  for (auto arg: dimOperands)
    indices.push_back(st.regs.get<Index>(arg));
  for (auto symbol: symbolOperands)
    symbols.push_back(st.regs.get<Index>(symbol));

  auto res = encodeAffineExpr(m.getResult(0), indices, symbols);
  if (!res)
    return "unsupported affine Expr";
  st.regs.add(op, Index(move(*res)));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ReturnOp op) {
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    st.retValues.push_back(st.regs.findOrCrash(op.getOperand(i)));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::shape::ShapeOfOp op) {
  if (!op.getType().isa<mlir::TensorType>())
    return "unsupported type";

  auto tensor = op.getOperand();
  if (!tensor.getType().isa<mlir::TensorType>())
    return "unsupported type";

  auto tt = st.regs.get<Tensor>(tensor);
  st.regs.add(op, Tensor(getTensorElemTy(op.getResult()), tt.getDims()));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tosa::AbsOp op) {
  auto dty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!dty)
    return "Unsupported type";
  auto t = st.regs.get<Tensor>(op.getOperand());
  auto ety = dty.getElementType();
  st.regs.add(op.getResult(), t.elementwiseUnaryOp(ety, [&](auto &&e) {
    return Float(e, ety).abs();
  }));

  return {};
}

// Encode operations that do not change the current state except the definition
// of a new register. They can raise UB however. This list is emphirically
// determined.
#define ENCODE_SCALAR_OPS(st, op) \
    ENCODE(st, op, mlir::arith::AddFOp); \
    ENCODE(st, op, mlir::arith::AddIOp); \
    ENCODE(st, op, mlir::arith::ConstantFloatOp); \
    ENCODE(st, op, mlir::arith::ConstantIndexOp); \
    ENCODE(st, op, mlir::arith::ConstantIntOp); \
    ENCODE(st, op, mlir::arith::ConstantOp); \
    ENCODE(st, op, mlir::arith::IndexCastOp); \
    ENCODE(st, op, mlir::arith::MulFOp); \
    ENCODE(st, op, mlir::arith::MulIOp); \
    ENCODE(st, op, mlir::arith::SubIOp); \
    ENCODE(st, op, mlir::AffineApplyOp); \
    ENCODE(st, op, mlir::linalg::IndexOp); \
    ENCODE(st, op, mlir::math::AbsOp); \
    ENCODE(st, op, mlir::shape::ShapeOfOp); \
    ENCODE(st, op, mlir::tosa::AbsOp);


static optional<string> encodeParallelLoopBodyAndOutput(
    State &newst, mlir::Block &block, const mlir::AffineMap &outputMap,
    const mlir::ShapedType &outputType, optional<Tensor> &t_res,
    // (yielded value, ind var) -> newly mapped value
    optional<function<Expr(const Expr&, const vector<Expr>&)>>
        outputValMap = nullopt) {
  // Encode the loop body
  // TODO: deal with merging UBs and memorys
  auto &ops = block.getOperations();
  mlir::Value yieldedValue;
  for (auto &op: ops) {
    auto op_operands = op.getOperands();
    for (const auto &opop: op_operands) {
      if (!newst.regs.contains(opop)) {
        RET_STR("This is a bug in mlir-tv or the loop is ill-formed: "
          "the result of a block in a parallel loop depends on the output"
          " variable: " << opop);
      }
    }

    ENCODE_SCALAR_OPS(newst, op);

    if (auto op2 = mlir::dyn_cast<mlir::linalg::YieldOp>(op)) {
      yieldedValue = op2.getOperand(0);
      break;
    } else if (auto op2 = mlir::dyn_cast<mlir::tensor::YieldOp>(op)) {
      yieldedValue = op2.getOperand();
      break;
    }

    RET_STR("has an unsupported operation: '" << op << "'");
  }

  auto &scope = newst.linalgGenericScopes.top();
  auto outputIndVars = doMap(scope.indVars, outputMap);
  auto tensorSz = addOne(doMap(scope.indVarUpperBounds, outputMap));
  auto yieldedExpr = newst.regs.getExpr(yieldedValue);
  if (outputValMap)
    yieldedExpr = (*outputValMap)(yieldedExpr, outputIndVars);

  t_res = Tensor::mkLambda(yieldedValue.getType(),
      move(tensorSz), move(outputIndVars), yieldedExpr);

  return {};
}



template<>
optional<string>
encodeOp(State &st, mlir::linalg::Conv2DNhwcHwcfOp op) {
  vector<Expr> strides, dilations;
  // TODO: The result may not fit in Index::BITS
  for (auto s: op.strides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.dilations())
    dilations.push_back(Index(d.getSExtValue()));

  if (op.hasTensorSemantics()) {
    auto t_input = st.regs.get<Tensor>(op.image());
    auto t_filter = st.regs.get<Tensor>(op.filter());

    auto t_res = t_input.conv(t_filter, strides, dilations);
    st.regs.add(op.getResult(0), move(t_res));
  } else {
    auto input = st.regs.get<MemRef>(op.image());
    auto filter = st.regs.get<MemRef>(op.filter());
    auto output = st.regs.get<MemRef>(op.outputs()[0]);

    if (!output.isIdentityMap())
      return "Currently output MemRef should have identity layout..";

    auto success = output.conv(input, filter, strides, dilations);
    // add well defined
    st.wellDefined(op, move(success));
  }

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::InitTensorOp op) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::RankedTensorType>();
  if (!ty || !Tensor::isTypeSupported(ty))
    return "Unsupported tensor type";

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
      Tensor(ty.getElementType(), ("init_tensor_") + to_string(new_var_idx++),
             sizes));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorCollapseShapeOp op) {
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
        st.wellDefined(op.getOperation(), size == resTy.getDimSize(i));
      newDims.push_back(move(size));
    }
  }

  st.wellDefined(op.getOperation(), t.get1DSize() == smt::get1DSize(newDims));
  st.regs.add(op.getResult(), t.reshape(newDims));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorExpandShapeOp op) {
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
          return "it has more than one unknown dimension size in one group";
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
      return "tensor size is too large";

    // If the original size isn't divisible, raise UB
    st.wellDefined(op, orgdim.mod(const_size) == 0);
    newdims[unknown_dim] = orgdim.udiv(const_size); 
  }

  st.regs.add(op.getResult(), t.reshape(newdims));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::MatmulOp op) {
  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";

  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    return "unsupported form";

  if (getTensorElemTy(op.getOperand(0)) != getTensorElemTy(op.getOperand(1)) ||
      getTensorElemTy(op.getOperand(0)) != getTensorElemTy(op.getResult(0)))
    return "unsupported types";

  // NOTE: op's output tensor (op.getOutputOperand()[0]->get()) isn't updated;
  // aqjune talked with mlir people and it is confirmed by them

  Tensor a = st.regs.get<Tensor>(op.getOperand(0));
  Tensor b = st.regs.get<Tensor>(op.getOperand(1));
  Tensor result = a.matmul(b);
  st.regs.add(op.getResult(0), Tensor(result));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::PadTensorOp op) {
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    return "Unsupported type";

  auto &region = op.getRegion();
  if (!region.hasOneBlock())
    return "Unsupported region";
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
  for (int i = 0; i < blk.getNumArguments(); ++i) {
    Expr idxvar = newst.linalgGenericScopes.top().indVars[i];
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
    return Expr::mkIte(isSource, sourceTensor.get(sourceIndices).first, pad);
  };

  optional<Tensor> t_res;
  if (auto msg = encodeParallelLoopBodyAndOutput(newst, blk,
      identityMap, retty, t_res, paddingOrSource))
    return *msg;

  newst.linalgGenericScopes.pop();

  st.regs.add(op.getResult(), move(*t_res));
  return {};
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
optional<string> encodeOp(State &st, mlir::tensor::DimOp op) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<Tensor>(op.source()).getDims(), op.index());
  st.regs.add(op, Index(res));
  st.wellDefined(op.getOperation(), move(wf));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::CastOp op) {
  auto tty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tty)
    return "Unsupported type";

  auto t = st.regs.get<Tensor>(op.getOperand());
  for (size_t i = 0; i < tty.getRank(); ++i) {
    if (tty.isDynamicDim(i))
      continue;
    st.wellDefined(op.getOperation(), (Expr)t.getDim(i) == tty.getDimSize(i));
  }
  st.regs.add(op, move(t));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::InsertOp op) {
  auto val = st.regs.get<Float>(op.scalar());
  auto dest = st.regs.get<Tensor>(op.dest());

  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto [tensor, inbounds] = dest.insert(val, indices);
  st.regs.add(op, move(tensor));
  st.wellDefined(op.getOperation(), move(inbounds));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::ExtractOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.

  auto t = st.regs.get<Tensor>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto [elem, inbounds] = t.get(indices);
  if (auto v = fromExpr(move(elem), op.getType()))
    st.regs.add(op, move(*v));
  else
    return "unsupported type";

  st.wellDefined(op.getOperation(), move(inbounds));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::FromElementsOp op) {
  vector<Expr> elems;
  for (unsigned i = 0; i < op.getNumOperands(); ++i)
    elems.push_back(st.regs.getExpr(op.getOperand(i)));

  auto res = Tensor(op.getType().getElementType(), move(elems));
  st.regs.add(op.getResult(), move(res));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::GenerateOp op) {
  auto exts = op.dynamicExtents();
  auto retty = op.getType().dyn_cast<mlir::RankedTensorType>();
  if (!retty)
    return "Unsupported type";
  auto *blk = op.getBody();
  if (!blk)
    return "Unsupported form";

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

  State newst = st;
  newst.linalgGenericScopes.push(State::LinalgGenericScope{move(upperbound)});
  for (int i = 0; i < blk->getNumArguments(); ++i) {
    Expr idxvar = newst.linalgGenericScopes.top().indVars[i];
    newst.regs.add(blk->getArgument(i), Index(idxvar));
  }

  auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
      retty.getRank(), op.getContext());
  optional<Tensor> t_res;
  if (auto msg = encodeParallelLoopBodyAndOutput(newst, *blk,
      identityMap, retty, t_res))
    return *msg;

  newst.linalgGenericScopes.pop();

  st.regs.add(op.getResult(), move(*t_res));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::ExtractSliceOp op) {
  vector<Index> offsets, sizes, strides;
  auto src = st.regs.get<Tensor>(op.getOperand(0));
  auto srcType = op.getOperand(0).getType().dyn_cast<mlir::ShapedType>();
  auto res = op.getResult();
  auto resType = res.getType().dyn_cast<mlir::ShapedType>();

  strides = getFromMixedOps<Index>(st, op.getMixedStrides());
  sizes = getFromMixedOps<Index>(st, op.getMixedSizes());
  offsets = getFromMixedOps<Index>(st, op.getMixedOffsets());

  if (offsets.size() != sizes.size() || sizes.size() != strides.size() ||
      strides.size() != (size_t)srcType.getRank())
    return "Unsupported form";

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

  vector<Expr> inIdxs, outIdxs;
  // indices that is going to be read from the output tensor
  inIdxs = Index::boundIndexVars(resType.getRank());

  // map the output tensor indices to source tensor indices
  unsigned idx = 0;
  for (unsigned i = 0; i < srcType.getRank(); i++) {
    uint64_t v;
    bool isDimSizeOne = idx >= resType.getRank() ||
        ((((Expr)sizes[i]).isUInt(v) && v == 1) &&
          resType.getDimSize(idx) != -1);
    outIdxs.push_back(isDimSizeOne ?
        (Expr)offsets[i] : (Expr)((inIdxs[idx++] * strides[i])) + offsets[i]);
  }
  st.regs.add(res,
      Tensor::mkLambda(src.getElemType(), move(dims), move(inIdxs),
                       src.get(outIdxs).first));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::InsertSliceOp op) {
  vector<Index> offsets, sizes, strides;
  auto src1 = st.regs.get<Tensor>(op.getOperand(0));
  auto src2 = st.regs.get<Tensor>(op.getOperand(1));
  auto res = op.getResult();
  auto rank = op.getOperand(0).getType().dyn_cast<mlir::ShapedType>().getRank();
  if (rank != op.getOperand(1).getType().dyn_cast<mlir::ShapedType>().getRank()
      || rank != res.getType().dyn_cast<mlir::ShapedType>().getRank())
    return "Unsupported tensor types of src and dest: their ranks do not match";

  strides = getFromMixedOps<Index>(st, op.getMixedStrides());
  sizes = getFromMixedOps<Index>(st, op.getMixedSizes());
  offsets = getFromMixedOps<Index>(st, op.getMixedOffsets());

  assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
         strides.size() == rank);

  vector<Expr> indVars = Index::boundIndexVars(rank);
  vector<Expr> dims = src2.getDims();
  vector<Expr> src1Idxs;

  Expr cond = Expr::mkBool(true);

  for (unsigned i = 0; i < rank; i++) {
    src1Idxs.push_back((indVars[i] - offsets[i]).udiv(strides[i]));
    cond &= ((indVars[i] - offsets[i]) % strides[i]).isZero() &
            (indVars[i] - offsets[i]).ult((Expr)sizes[i] * strides[i]);
  }

  // Picking the value from src1 must not be out of bounds.
  auto [src1elem, src1wb] = src1.get(src1Idxs);
  Expr output = Expr::mkIte(cond, move(src1elem), src2.get(indVars).first);

  st.wellDefined(op.getOperation(), move(src1wb));
  st.regs.add(res, Tensor::mkLambda(
      src1.getElemType(), move(dims), move(indVars), output));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tosa::AddOp op) {
  auto optys = op.getOperandTypes();
  if (!optys[0].isa<mlir::RankedTensorType>() ||
      !optys[1].isa<mlir::RankedTensorType>())
    return "Unsupported operand types";

  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  return encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.add(b); },
      [](auto &&a, auto &&b) { return (Expr)a + (Expr)b; });
}

template<>
optional<string> encodeOp(State &st, mlir::tosa::MulOp op) {
  auto optys = op.getOperandTypes();
  if (!optys[0].isa<mlir::RankedTensorType>() ||
      !optys[1].isa<mlir::RankedTensorType>())
    return "Unsupported operand types";
  if (op.shift() != 0)
    return "Mul with shift is unsupported";

  mlir::Value arg0 = op.getOperand(0);
  mlir::Value arg1 = op.getOperand(1);

  return encodeBinaryOp(st, op, arg0, arg1,
      [](auto &&a, auto &&b) { return a.mul(b); },
      [](auto &&a, auto &&b) { return (Expr)a * (Expr)b; });
}

static variant<string, MemRef> createNewLocalBlk(
    Memory *m, vector<Expr> &&dims, mlir::MemRefType memrefTy, bool writable) {
  if (!MemRef::isTypeSupported(memrefTy))
    return "unsupported element type";

  auto layout = MemRef::getLayout(memrefTy, dims);
  // Add a new local block
  auto bid = m->addLocalBlock(smt::get1DSize(dims), Expr::mkBool(writable));
  // Create MemRef which points to the newly created block
  auto memref =
      MemRef(m, memrefTy.getElementType(), bid, Index::zero(), dims,
          move(layout));

  return {move(memref)};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::AllocOp op) {
  auto memrefTy = op.getType().cast<mlir::MemRefType>();
  if (!memrefTy.getLayout().isIdentity())
    return "unsupported memref type for alloc: it has a non-identity layout map";

  auto dsizes = op.dynamicSizes();
  vector<Expr> dszExprs;
  for (const auto &sz: dsizes) {
    dszExprs.push_back(st.regs.get<Index>(sz));
  }
  auto dims = ShapedValue::getDims(memrefTy, false, move(dszExprs));

  auto memrefOrErr = createNewLocalBlk(st.m.get(), move(dims), memrefTy, true);
  if (holds_alternative<string>(memrefOrErr))
    return get<0>(move(memrefOrErr));
  auto memref = get<1>(move(memrefOrErr));

  st.regs.add(op, move(memref));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::DimOp op) {
  auto [res, wf] = encodeDimOp(
      st, st.regs.get<MemRef>(op.source()).getDims(), op.index());
  st.regs.add(op, Index(res));
  st.wellDefined(op.getOperation(), move(wf));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::LoadOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(0));
  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  auto [Expr, success] = m.get(indices);
  if (auto vt = fromExpr(move(Expr), op.getType())) {
    st.regs.add(op, move(*vt));
    st.wellDefined(op.getOperation(), move(success));
  } else
    return "unsupported type";

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::StoreOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(1));
  vector<Expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  if (op.getOperand(0).getType().isa<mlir::Float32Type>()) {
    auto val = st.regs.get<Float>(op.getOperand(0));
    auto success = m.store(val, indices);
    st.wellDefined(op.getOperation(), move(success));
  } else {
    // Currently we support only f32 memory type
    return "unsupported type";
  }

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::SubViewOp op) {
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
    return "Subview result size mismatch";

  auto unusedDims = optionalUnusedDimsMask.getValue();
  auto memref = src.subview(offsets, sizes, strides, unusedDims, rankDiff);
  st.regs.add(op.getResult(), move(memref));
  return {};
}

static void storeTensorTo(
    State &st, mlir::Operation *op, Tensor &&tensor, const MemRef &memref,
    mlir::MemRefType memrefTy) {
  if (memrefTy.getLayout().isIdentity()) {
    // memref with identity map
    auto success = memref.storeArray(tensor.asArray(), Index::zero(),
        tensor.get1DSize(), false);
    st.wellDefined(op, move(success));

  } else {
    // TODO: can we further optimize this if we know that memref is a
    // freshly created block?
    // We may not need to preserve the 'previous' bytes.

    vector<Expr> idxs = Index::boundIndexVars(memrefTy.getRank());
    auto [tVal, tSuccess] = tensor.get(idxs);
    auto [mVal, mSuccess] = memref.get(idxs);
    auto success = tSuccess & mSuccess;

    // TODO: clarify whether this is precondition or UB.
    st.wellDefined(op, Expr::mkForall(idxs, success.implies(mVal == tVal)));
    st.hasQuantifier = true;
  }
}

template<>
optional<string> encodeOp(State &st, mlir::memref::BufferCastOp op) {
  auto tensor = st.regs.get<Tensor>(op.getOperand());
  auto memrefTy = op.memref().getType().cast<mlir::MemRefType>();
  auto dims = tensor.getDims();

  // Create a read-only block.
  auto memrefOrErr = createNewLocalBlk(st.m.get(), move(dims), memrefTy, false);
  if (holds_alternative<string>(memrefOrErr))
    return get<0>(move(memrefOrErr));
  auto memref = get<1>(move(memrefOrErr));

  storeTensorTo(st, op.getOperation(), move(tensor), memref, memrefTy);
  st.regs.add(op.memref(), move(memref));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::TensorLoadOp op) {
  auto m = st.regs.get<MemRef>(op.getOperand());
  // Step 1. Mark the MemBlock pointed by the memref as read-only.
  auto &memory = *(st.m);
  memory.setWritable(m.getBID(), false);

  // Step 2. Create a new Tensor using Tensor::mkLambda
  auto dims = m.getDims();
  vector<Expr> idxs = Index::boundIndexVars(dims.size());
  auto expr = m.get(idxs).first;
  Tensor t_res = Tensor::mkLambda(getTensorElemTy(op.getResult()),
      move(dims), move(idxs), expr);

  st.regs.add(op.getResult(), t_res);
  st.wellDefined(op.getOperation(), m.isInBounds());

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::TensorStoreOp op) {
  auto t = st.regs.get<Tensor>(op.tensor());
  auto m = st.regs.get<MemRef>(op.memref());

  // Src and tgt's shapes & element types must match
  // Memref may have its layout, though.
  for (unsigned i = 0; i < t.getRank(); ++i)
    st.wellDefined(op.getOperation(), (Expr)t.getDim(i) == (Expr)m.getDim(i));

  storeTensorTo(st, op.getOperation(), move(t), m,
      op.memref().getType().cast<mlir::MemRefType>());

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::FillOp op) {
  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";
  if (op.getNumResults() != 1)
    return "it has multiple results";

  auto t = st.regs.get<Tensor>(op.getOperand(1));
  auto res = Tensor(t.getElemType(),
      st.regs.getExpr(op.getOperand(0)), t.getDims());
  st.regs.add(op.getResult(0), move(res));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::DotOp op) {
  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";

  if (op.getNumResults() != 1)
    return "it has multiple results";

  auto inputOps = op.getInputOperands();
  auto outputTy = op.getType(0).dyn_cast<mlir::TensorType>();

  auto outputDim = ShapedValue::getDims(outputTy, false);
  if (outputDim.size() != 1)
    return "unknown dot format; isn't the result tensor having one element?";

  if (outputTy.getElementType() !=
      inputOps[0]->get().getType().dyn_cast<mlir::TensorType>()
          .getElementType())
    return "casting is not supported";

  auto t1 = st.regs.get<Tensor>(inputOps[0]->get());
  auto t2 = st.regs.get<Tensor>(inputOps[1]->get());
  st.wellDefined(op.getOperation(), t1.get1DSize() == t2.get1DSize());

  auto res = t1.dot(t2);
  st.regs.add(op.getResult(0),
      Tensor(t1.getElemType(), move(res), move(outputDim)));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::shape::ToExtentTensorOp op) {
  // TODO: MLIR doc says
  //   If the shape represents an error, this op’s behavior is undefined.
  // Should figure out whether this applies to a Tensor operand as well.
  if (!op.getOperand().getType().isa<mlir::TensorType>())
    return "unsupported type";

  auto tt = st.regs.get<Tensor>(op.getOperand());
  assert(tt.getDims().size() ==
         (size_t)op.getType().cast<mlir::TensorType>().getRank());
  st.regs.add(op, tt);
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::sparse_tensor::ConvertOp op) {
  auto tensor = op.getOperand();
  auto tt = st.regs.get<Tensor>(tensor);
  st.regs.add(op, move(tt));
  return {};
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
  fill(resFilled.begin(), resFilled.end(), -1);

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

static optional<string>
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
      return "Unsupported ShapedValue";
    }
    for (int64_t i = 0, e = r; i < e; ++i) {
      viewSizes.push_back(t->getDim(i));
    }
  }

  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto ae = encodeAffineExpr(map.getResult(idx), indVarBounds, {});
    if (!ae)
      return "unsupported affine Expr";

    Expr size = (Expr)viewSizes[idx];
    Expr inbounds = size.isNonZero().implies(ae->ult(size));
    st.wellDefined(op.getOperation(), move(inbounds));
  }

  return {};
}

static optional<string> initInputStateForLoopBody(
    State &st, mlir::linalg::GenericOp op) {
  // TODO: Currently we do not encode UB in loop body. How to deal with UB properly?
  auto indexingMaps = op.indexing_maps().getValue();
  auto &block = *op.region().begin();

  const vector<Expr> &inductionVars = st.linalgGenericScopes.top().indVars;

  // Fill in args
  assert(op.getInputOperands().size() + op.getNumOutputs() ==
         indexingMaps.size());

  // Output variables are not encoded! Reduction loops are dealt specially
  size_t numout = (size_t)op.getNumOutputs();
  for (size_t arg_i = 0; arg_i + numout < indexingMaps.size(); ++arg_i) {
    auto inputMap = indexingMaps[arg_i].cast<mlir::AffineMapAttr>().getValue();
    auto op_i = op.getInputOperand(arg_i)->get();

    if (op_i.getType().isa<mlir::FloatType>()) {
      // A scalar value.
      Float f_input = st.regs.get<Float>(op_i);
      st.regs.add(block.getArgument(arg_i), f_input);

    } else if (auto tensorty = op_i.getType().dyn_cast<mlir::TensorType>()) {
      // A tensor value.
      auto elemty = tensorty.getElementType();
      Tensor t_input = st.regs.get<Tensor>(op_i);

      if (inputMap.getNumResults() == 0) {
        // A tensor with a single element; e.g. tensor<f32>.
        st.regs.add(block.getArgument(arg_i), t_input.get({Index::zero()}).first, elemty);
      } else {
        vector<Expr> affine_Exprs;
        for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
          auto ae_res = encodeAffineExpr(inputMap.getResult(i), inductionVars, {});
          if (!ae_res)
            RET_STR_WITH_PREFIX("unsupported affine Expr ",
                                inputMap.getResult(i));

          affine_Exprs.emplace_back(move(*ae_res));
        }

        auto t_elem = t_input.get(affine_Exprs).first;
        st.regs.add(block.getArgument(arg_i), t_elem, elemty);
      }
    } else if (auto memrefty = op_i.getType().dyn_cast<mlir::MemRefType>()) {
      // A MemRef value.
      // TODO: currently we support float32 element type
      MemRef m_input = st.regs.get<MemRef>(op_i);

      vector<Expr> affine_Exprs;
      for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
        auto ae_res = encodeAffineExpr(inputMap.getResult(i), inductionVars, {});
        if (!ae_res)
          RET_STR_WITH_PREFIX("unsupported affine expr ",
                              inputMap.getResult(i));

        affine_Exprs.emplace_back(move(*ae_res));
      }

      // TODO: We do not encode UB in loops currently. How to deal with this?
      auto m_elem = m_input.get(affine_Exprs).first;
      st.regs.add(block.getArgument(arg_i), 
          Float(m_elem, memrefty.getElementType()));
    } else {
      return "unsupported block argument type";
    }
  }

  return {};
}

static optional<string> encodeReductionLoopBodyAndOutput(
    State &newst, mlir::Block &block,
    const mlir::ArrayRef<mlir::Attribute> &indexingMaps,
    const mlir::ShapedType &outputType, optional<Tensor> &t_res) {
  // Deal with simple reduction loops.
  // TODO: support more kinds of reduction loops!
  string errmsg = "permutated output map or simple reduction form is"
                  " supported only";

  // TODO: deal with merging UBs and memorys
  auto &ops = block.getOperations();
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
  assert(!newst.regs.contains(lastarg));

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
    return errmsg;

  auto sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(idx);

  unsigned cnt = 0;
  for (auto &op: ops) {
    if (cnt++ == ops.size() - 2)
      // Don't directly encode %sum
      break;

    ENCODE_SCALAR_OPS(newst, op);
    RET_STR("has an unsupported operation" << op);
  }

  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();

  auto &linalgInfo = newst.linalgGenericScopes.top();

  // Represent %v as an element of a tensor.
  Tensor t_v = Tensor::mkLambda(
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
    return {};
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
        return errmsg;
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
    auto t_sum = Tensor::mkLambda(
          t_v.getElemType(),
          addOne(move(boundsForRes)),
          move(indVarsForRes),
          t_v.get(linalgInfo.indVars).first)
        .sum();

    auto outputIndVars = doMap(linalgInfo.indVars, outputMap);
    t_res = Tensor::mkLambda(
        t_v.getElemType(), move(tensorSz), move(outputIndVars), t_sum);
    return {};
  }
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::GenericOp op) {
  if (!(op.hasTensorSemantics() || op.hasBufferSemantics()))
    return "tensor/buffer semantics is supported only";

  if (op.getNumOutputs() != 1)
    return "a single output is supported only";

  auto &region = op.region();
  if (!llvm::hasSingleElement(region))
    return "a single block is supported only";

  auto &block = region.front();
  if (!std::all_of(block.args_begin(), block.args_end(),
      [](auto &arg) { return arg.getType().isSignlessIntOrFloat(); }))
    return "unsupported block arguments";

  if (llvm::any_of(op.iterator_types(), [](mlir::Attribute attr) {
    auto str = attr.cast<mlir::StringAttr>().getValue();
    return str != mlir::getParallelIteratorTypeName() &&
           str != mlir::getReductionIteratorTypeName() &&
           str != mlir::getWindowIteratorTypeName();
  }))
    return "unsupported iterator type";

  auto loopBounds = findLoopBounds(st, op);

  if (auto errmsg = encodeUBForTensorShapeMatch(st, op, loopBounds))
    return errmsg;

  // Start from newst
  State newst = st;
  newst.linalgGenericScopes.push(State::LinalgGenericScope{move(loopBounds)});

  if (auto msg = initInputStateForLoopBody(newst, op))
    return msg;

  optional<Tensor> t_res;
  auto indexingMaps = op.indexing_maps().getValue();
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  auto outputType = op.getOutputOperand(0)->get().getType()
      .cast<mlir::ShapedType>();

  if (outputMap.isPermutation()) {
    if (auto errmsg = encodeParallelLoopBodyAndOutput(newst, block, outputMap,
          outputType, t_res))
      return errmsg;

  } else {
    if (auto errmsg = encodeReductionLoopBodyAndOutput(newst, block,
          indexingMaps, outputType, t_res))
      return errmsg;
  }

  assert(t_res->getDims().size() != 0);
  newst.linalgGenericScopes.pop();

  if (op.hasTensorSemantics()) {
    if (op.getNumResults() != 0) {
      // NOTE: op's output tensor (op.getOutputOperand()[0]->get()) isn't updated;
      // aqjune talked with mlir people and confirmed
      assert(op.getNumResults() == 1);
      st.regs.add(op.getResult(0), move(*t_res));
    }
    return {};
  } else if (op.hasBufferSemantics()) {
    auto m_res = st.regs.get<MemRef>(op.getOutputOperand(0)->get());
    auto success = m_res.storeArray(t_res->asArray(), Index::zero(), t_res->get1DSize());
    st.wellDefined(op, move(success));
    return {};
  }
  llvm_unreachable("Unknown linalg::genric semantics");
}


static optional<string> encodeRegion(
    State &st, mlir::Region &region, bool printOps) {
  if (!llvm::hasSingleElement(region))
    return "Only a region with one block is supported";

  auto &block = region.front();
  for (auto &op: block) {
    if (printOps)
      llvm::outs() << "  " << op << "\n";

    ENCODE_SCALAR_OPS(st, op);

    ENCODE(st, op, mlir::ReturnOp);

    ENCODE(st, op, mlir::tensor::CastOp);
    ENCODE(st, op, mlir::tensor::DimOp);
    ENCODE(st, op, mlir::tensor::InsertOp);
    ENCODE(st, op, mlir::tensor::ExtractOp);
    ENCODE(st, op, mlir::tensor::ExtractSliceOp);
    ENCODE(st, op, mlir::tensor::FromElementsOp);
    ENCODE(st, op, mlir::tensor::GenerateOp);
    ENCODE(st, op, mlir::tensor::InsertSliceOp);

    ENCODE(st, op, mlir::tosa::AddOp);
    ENCODE(st, op, mlir::tosa::MulOp);

    ENCODE(st, op, mlir::memref::AllocOp);
    ENCODE(st, op, mlir::memref::BufferCastOp);
    ENCODE(st, op, mlir::memref::DimOp);
    ENCODE(st, op, mlir::memref::LoadOp);
    ENCODE(st, op, mlir::memref::StoreOp);
    ENCODE(st, op, mlir::memref::SubViewOp);
    ENCODE(st, op, mlir::memref::TensorLoadOp);
    ENCODE(st, op, mlir::memref::TensorStoreOp);

    ENCODE(st, op, mlir::linalg::Conv2DNhwcHwcfOp);
    ENCODE(st, op, mlir::linalg::DotOp);
    ENCODE(st, op, mlir::linalg::FillOp);
    ENCODE(st, op, mlir::linalg::GenericOp);
    ENCODE(st, op, mlir::linalg::InitTensorOp);
    ENCODE(st, op, mlir::linalg::MatmulOp);
    ENCODE(st, op, mlir::linalg::PadTensorOp);
    ENCODE(st, op, mlir::linalg::TensorCollapseShapeOp);
    ENCODE(st, op, mlir::linalg::TensorExpandShapeOp);
    
    ENCODE(st, op, mlir::shape::ToExtentTensorOp);

    ENCODE(st, op, mlir::sparse_tensor::ConvertOp);

    RET_STR("Unknown op (" << op.getName() << "): " << op);
  }
  if (printOps)
    llvm::outs() << "\n";
  return {};
}

optional<string> encode(State &st, mlir::FuncOp &fn, bool printOps) {
  return encodeRegion(st, fn.getRegion(), printOps);
}
