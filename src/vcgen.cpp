#include "abstractops.h"
#include "value.h"
#include "smt.h"
#include "state.h"
#include "vcgen.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/TensorOps.h.inc"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "z3++.h"
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <variant>
#include <vector>

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

namespace {
class Defer {
private:
  function<void()> fn;
public:
  Defer(function<void()> &&fn): fn(fn) {}
  ~Defer() { fn(); }
};

class ValidationInput {
public:
  mlir::FuncOp src, tgt;
  string dumpSMTPath;

  MemEncoding encoding;
  unsigned int numBlocks;
};

enum VerificationStep {
  UB,
  RetValue,
  Memory
};
};

static optional<string> checkFunctionSignatures(mlir::FuncOp src, mlir::FuncOp tgt) {
  if (src.getNumArguments() != tgt.getNumArguments())
    RET_STR("The source and target program has different number of arguments.");

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto srcArgTy = src.getArgument(i).getType();
    auto tgtArgTy = tgt.getArgument(i).getType();
    if (srcArgTy != tgtArgTy)
      RET_STR("The source and target argument type is different.\n"
          << "Src: " << srcArgTy << ", Tgt: " << tgtArgTy);
  }

  return {};
}

static variant<string, State>
createInputState(mlir::FuncOp fn, unsigned int numBlocks, MemEncoding encoding, ArgInfo &args) {
  State s(numBlocks, encoding);
  s.isWellDefined = ctx.bool_val(true);
  unsigned n = fn.getNumArguments();

  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    auto argty = arg.getType();

    if (auto value = args.get(i)) {
      // Use identical arguments from source when encoding target.
      if (holds_alternative<MemRef>(*value)) {
        auto memref = get<MemRef>(*value);
        memref.setMemory(s.m.get());
        s.regs.add(arg, memref);
      } else {
        s.regs.add(arg, move(*value));
      }
    } else {
      // Encode each arguments of source.
      if (auto ty = argty.dyn_cast<mlir::TensorType>()) {
        auto dimsAndElemTy = Tensor::getDimsAndElemTy(ty);
        if (!dimsAndElemTy)
          RET_STR("Unsupported Tensor element type: " << arg.getType());
        s.regs.add(arg, Tensor("arg" + to_string(arg.getArgNumber()),
                              dimsAndElemTy->first,
                              dimsAndElemTy->second));

      } else if (auto ty = argty.dyn_cast<mlir::MemRefType>()) {
        auto dimsAndElemTy = MemRef::getDimsAndElemTy(ty);
        if (!dimsAndElemTy)
          RET_STR("Unsupported MemRef element type: " << arg.getType());
        // TODO : out of bounds pointer is allowed?
        s.regs.add(arg, MemRef(s.m.get(), "arg" + to_string(arg.getArgNumber()),
          dimsAndElemTy->first,
          dimsAndElemTy->second));

      } else if (auto ty = argty.dyn_cast<mlir::IndexType>()) {
        s.regs.add(arg, Index("arg" + to_string(arg.getArgNumber())));

      } else if (auto ty = argty.dyn_cast<mlir::FloatType>()) {
        s.regs.add(arg, Float("arg" + to_string(arg.getArgNumber())));

      } else {
        RET_STR("Unsupported type: " << arg.getType());
      }
      args.add(i, s.regs.findOrCrash(arg));
    }
  }
  return s;
}


template<class T>
optional<z3::expr> encodeAffineExpr(
    mlir::AffineExpr ae, const vector<T> &dimvars
) {
  switch (ae.getKind()) {
  case mlir::AffineExprKind::Add:
  case mlir::AffineExprKind::Mul: {
    auto aboe = ae.dyn_cast<mlir::AffineBinaryOpExpr>();
    auto lhs = encodeAffineExpr(aboe.getLHS(), dimvars);
    auto rhs = encodeAffineExpr(aboe.getRHS(), dimvars);
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


#define ENCODE(st, op, ty) { \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    auto errmsg = encodeOp(st, op2); \
    if (errmsg) { \
      RET_STR("Unknown op: " << op << "\n\t" << *errmsg << "\n") \
    } \
    continue; \
  } \
}

template<class T>
static optional<string> encodeOp(State &st, T op);

template<>
optional<string>
encodeOp(State &st, mlir::linalg::ConvInputNHWCFilterHWCFOp op) {
  if (!llvm::all_of(op.dilations(), [](auto i) { return i == 1; }))
    return "dilation isn't one\n";
  else if (!llvm::all_of(op.strides(), [](auto i) { return i == 1; }))
    return "strides isn't one\n";

  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";

  auto inputs = op.getInputTensorOperands();
  assert(inputs.size() == 2);
  auto input = inputs[0]->get();
  auto filter = inputs[1]->get();

  // NOTE: conv's output tensor (op.getOutputTensorOperands()[0]->get())
  // aqjune talked with mlir people and it is confirmed by them

  auto t_input = st.regs.get<Tensor>(input);
  auto t_filter = st.regs.get<Tensor>(filter);

  auto t_res = t_input.conv(t_filter);
  st.regs.add(op.getResult(0), move(t_res));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::InitTensorOp op) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::TensorType>();
  assert(ty);

  vector<z3::expr> sizes;
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

  auto elemTy = Tensor::getElemTy(ty);
  if (!elemTy)
    return "Unsupported tensor type";

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  auto name = string("init_tensor_") + to_string(new_var_idx++);
  st.regs.add(res, Tensor(name, sizes, *elemTy));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorCollapseShapeOp op) {
  Tensor t = st.regs.get<Tensor>(op.getOperand());
  auto res = Tensor::getDimsAndElemTy(op.getResultType());
  if (!res)
    return "unsupported type";

  st.regs.add(op.getResult(), t.reshape(res->first));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorExpandShapeOp op) {
  Tensor t = st.regs.get<Tensor>(op.getOperand());

  auto res = Tensor::getDimsAndElemTy(op.getResultType());
  if (!res)
    return "unsupported type";
  auto newdims = move(res->first);
  auto indices = op.getReassociationIndices();

  unsigned i = 0;
  for (unsigned srci = 0; srci < indices.size(); ++srci) {
    auto &ids = indices[srci];
    auto orgdim = (z3::expr)t.getDim(srci);

    // Allow one '?' only.
    int unknown_dim = -1;
    int64_t const_size = 1;
    for (auto id: ids) {
      if (op.getResultType().getDimSize(id) == -1) {
        if (unknown_dim != -1)
          return "has more than one unknown size in one group";
        unknown_dim = i;
      } else {
        const_size *= op.getResultType().getDimSize(id);
      }
      ++i;
    }

    if (unknown_dim == -1)
      // Nothing to do
      continue;

    if (const_size >= (1ull << Index::BITS))
      return "tensor size is too large";

    // If the original size isn't divisible, raise UB
    st.wellDefined(z3::mod(orgdim, const_size) == 0);
    newdims[unknown_dim] = z3::udiv(orgdim, const_size); 
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
optional<string> encodeOp(State &st, mlir::tensor::DimOp op) {
  auto tensor = op.source();
  if (!tensor.getType().isa<mlir::TensorType>())
    return "tensor type is supported only";
  auto t = st.regs.get<Tensor>(tensor);

  if (auto idx = op.getConstantIndex())
    st.regs.add(op, t.getDim(*idx));
  else {
    // TODO: if-then-else needed
    return "variable index not implemented yet";
  }
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::tensor::ExtractOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.

  auto t = st.regs.get<Tensor>(op.getOperand(0));
  vector<z3::expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  if (op.getType().isa<mlir::IndexType>())
    st.regs.add(op, Index(t.get(indices)));
  else if (op.getType().isa<mlir::Float32Type>())
    st.regs.add(op, Float(t.get(indices)));
  else
    // TODO: how to do this well?
    return "unsupported type";

  for (unsigned i = 0; i < indices.size(); ++i)
    // TODO: revisit this; may not be axis-wise
    st.wellDefined(z3::ult(indices[i], t.getDim(i)));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::LoadOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(0));
  vector<z3::expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  if (op.getType().isa<mlir::Float32Type>()) {
    auto [expr, success] = m.load(indices);
    st.regs.add(op, Float(expr));
    st.wellDefined(success);
  }
  else
    // TODO: how to do this well?
    return "unsupported type";

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::StoreOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.
  auto m = st.regs.get<MemRef>(op.getOperand(1));
  vector<z3::expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  if (op.getOperand(0).getType().isa<mlir::Float32Type>()) {
    auto val = st.regs.get<Float>(op.getOperand(0));
    auto success = m.store(val, indices);
    st.wellDefined(success);
  } else {
    // Currently we support only f32 memory type
    return "unsupported type";
  }

  return {};
}


template<>
optional<string> encodeOp(State &st, mlir::memref::TensorLoadOp op) {
  auto m = st.regs.get<MemRef>(op.getOperand());
  // step1. MemBlock which contains source memref marks as not writable.
  auto &memory = *(st.m);
  memory.setWritable(m.getBID(), false);

  // step2. create new Tensor that alias origin memref using Tensor::mkLambda
  auto dims = m.getDims();
  vector<z3::expr> idxs;
  for (int i = 0; i < dims.size(); i ++) {
    idxs.push_back(Index("Index_" + std::to_string(i)));
  }
  auto [expr, success] = m.load(idxs);
  Tensor t_res = Tensor::mkLambda(move(dims), move(idxs), expr);

  // step3. add result tensor to register
  st.regs.add(op.getResult(), t_res);
  st.wellDefined(m.isInBounds());

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::IndexOp op) {
  uint64_t i = op.dim();
  assert(i < st.linalgGenericScopes.top().indVars.size());
  z3::expr idxvar = st.linalgGenericScopes.top().indVars[i];
  st.regs.add(op, Index(idxvar));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::FillOp op) {
  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";
  if (op.getNumResults() != 1)
    return "it has multiple results";

  auto t = st.regs.get<Tensor>(op.getOperand(1));
  auto res = Tensor(st.regs.getZ3Expr(op.getOperand(0)), t.getDims());
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
  if (outputTy.getElementType() !=
      inputOps[0]->get().getType().dyn_cast<mlir::TensorType>()
          .getElementType())
    return "casting is not supported";

  auto resty = Tensor::getDimsAndElemTy(outputTy);
  if (!resty)
    return "unsupported type";

  auto t1 = st.regs.get<Tensor>(inputOps[0]->get());
  auto t2 = st.regs.get<Tensor>(inputOps[1]->get());
  st.wellDefined(t1.get1DSize() == t2.get1DSize());
  auto res = t1.dot(t2);
  st.regs.add(op.getResult(0), Tensor(res, resty->first));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::AddFOp op) {
  auto a = st.regs.get<Float>(op.getOperand(0));
  auto b = st.regs.get<Float>(op.getOperand(1));
  st.regs.add(op, a.add(b));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::MulFOp op) {
  auto a = st.regs.get<Float>(op.getOperand(0));
  auto b = st.regs.get<Float>(op.getOperand(1));
  st.regs.add(op, a.mul(b));
  return {};
}

static void addIntOrIndex(
    State &st, mlir::Value res, const z3::expr &e, bool isIndex) {
  if (isIndex)
    st.regs.add(res, Index(e));
  else
    st.regs.add(res, Integer(e));
}

template<>
optional<string> encodeOp(State &st, mlir::AddIOp op) {
  auto a = st.regs.getZ3Expr(op.getOperand(0));
  auto b = st.regs.getZ3Expr(op.getOperand(1));
  addIntOrIndex(st, op, a + b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::SubIOp op) {
  auto a = st.regs.getZ3Expr(op.getOperand(0));
  auto b = st.regs.getZ3Expr(op.getOperand(1));
  addIntOrIndex(st, op, a - b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::MulIOp op) {
  auto a = st.regs.getZ3Expr(op.getOperand(0));
  auto b = st.regs.getZ3Expr(op.getOperand(1));
  addIntOrIndex(st, op, a * b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::IndexCastOp op) {
  auto src = st.regs.getZ3Expr(op.getOperand());
  assert(src.is_bv());
  unsigned srcWidth = src.get_sort().bv_size();

  unsigned destWidth = 0;
  if (auto dstty = op.getType().dyn_cast<mlir::IntegerType>())
    destWidth = dstty.getWidth();
  else {
    assert(op.getType().isa<mlir::IndexType>());
    destWidth = Index::BITS;
  }

  z3::expr casted = src;
  if (srcWidth > destWidth)
    casted = src.extract(destWidth - 1, 0);
  else if (srcWidth < destWidth)
    casted = z3::concat(ctx.bv_val(0, destWidth - srcWidth), casted);
  st.regs.add(op, Integer(casted));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::AffineApplyOp op) {
  auto m = op.getAffineMap();
  if (m.getNumResults() != 1)
    return "num results is larger than one";

  vector<Index> indices;
  for (auto arg: op.mapOperands()) {
    indices.push_back(st.regs.get<Index>(arg));
  }
  auto res = encodeAffineExpr(m.getResult(0), indices);
  if (!res)
    return "unsupported affine expr";
  st.regs.add(op, Index(move(*res)));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ReturnOp op) {
  if (op.getNumOperands() == 0)
    return {};
  st.retValue = st.regs.findOrCrash(op.getOperand(0));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ConstantIndexOp op) {
  st.regs.add(op, Index(op.getValue()));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ConstantFloatOp op) {
  auto fp = op.getValue();
  st.regs.add(op, Float(fp));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ConstantOp op) {
  auto attr = op.getValue();
  if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    if (!denseAttr.isSplat())
      return "a fp splat constant tensor is supported only";

    auto splatfval = denseAttr.getSplatValue().dyn_cast<mlir::FloatAttr>();
    if (!splatfval)
      return "a fp splat constant tensor is supported only";

    auto resty = Tensor::getDimsAndElemTy(
        op.getType().cast<mlir::TensorType>());
    if (!resty)
      return "unsupported type";
    st.regs.add(op, Tensor(Float(splatfval.getValueAsDouble()), resty->first));
    return {};
  } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    llvm::APInt i = intAttr.getValue();
    unsigned bw = i.getBitWidth();
    if (bw > 64)
      return "size is too large";

    st.regs.add(op, Integer(i.getSExtValue(), bw));
    return {};
  }
  return "unsupported constant";
}


template<>
optional<string> encodeOp(State &st, mlir::shape::ShapeOfOp op) {
  if (!op.getType().isa<mlir::TensorType>())
    return "unsupported type";

  auto tensor = op.getOperand();
  if (!tensor.getType().isa<mlir::TensorType>())
    return "unsupported type";

  auto tt = st.regs.get<Tensor>(tensor);
  st.regs.add(op, Tensor(tt.getDims()));
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
         op.getType().cast<mlir::TensorType>().getRank());
  st.regs.add(op, tt);
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

    auto t = st.regs.get<Tensor>(opOperand->get());
    for (int64_t i = 0, e = r; i < e; ++i) {
      viewSizes.push_back(t.getDim(i));
    }
  }

  mlir::AffineMap map = op.getLoopsToShapesMap();
  // numDims: # of induction variables
  unsigned numDims = map.getNumDims();
  // numRes: # of output affine exprs
  // For example, given two affine maps
  //   (i, j, k) -> (i, j)
  //   (i, j, k) -> (i, k)
  //   numDims = 3 (i, j, k), numRes = 4 (i, j, i, k)
  unsigned numRes = map.getNumResults();

  vector<Index> res(numDims);
  vector<bool> resFilled(numDims);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    auto d = result.dyn_cast<mlir::AffineDimExpr>();
    if (!d)
      continue;

    unsigned pos = d.getPosition();
    if (resFilled[pos])
      continue;
    // If i < N, store N - 1
    // It is to bound e.g., 'i + j <= N - 1 + M - 1'
    res[pos] = viewSizes[idx].ofs(-1);
    resFilled[pos] = true;
  }

  return res;
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

    auto t = st.regs.get<Tensor>(opOperand->get());
    for (int64_t i = 0, e = r; i < e; ++i) {
      viewSizes.push_back(t.getDim(i));
    }
  }

  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto ae = encodeAffineExpr(map.getResult(idx), indVarBounds);
    if (!ae)
      return "unsupported affine expr";

    z3::expr inbounds = z3::ult(*ae, (z3::expr)viewSizes[idx]);
    st.wellDefined(inbounds);
  }

  return {};
}

static optional<string> initInputStateForLoopBody(
    State &st, mlir::linalg::GenericOp op) {
  auto indexingMaps = op.indexing_maps().getValue();
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  auto &block = *op.region().begin();

  const vector<z3::expr> &inductionVars = st.linalgGenericScopes.top().indVars;

  // Fill in args
  assert(op.getInputOperands().size() + op.getNumOutputs() ==
         indexingMaps.size());

  // Output variables are not encoded! Reduction loops are dealt specially
  for (unsigned arg_i = 0; arg_i + op.getNumOutputs() < indexingMaps.size();
       ++arg_i) {
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
        st.regs.add(block.getArgument(arg_i), t_input.get({Index::zero()}),
                    elemty);
      } else {
        vector<z3::expr> affine_exprs;
        for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
          auto ae_res = encodeAffineExpr(inputMap.getResult(i), inductionVars);
          if (!ae_res)
            RET_STR_WITH_PREFIX("unsupported affine expr ",
                                inputMap.getResult(i));

          affine_exprs.emplace_back(move(*ae_res));
        }

        auto t_elem = t_input.get(affine_exprs);
        st.regs.add(block.getArgument(arg_i), t_elem, elemty);
      }
    } else {
      return "unsupported block argument type";
    }
  }

  return {};
}

// map := (i, j, k) -> (j, k, i)
// input := [a, b, c]
// output := [b, c, a]
static vector<z3::expr> doMap(
    const vector<z3::expr> &input, const mlir::AffineMap &map) {
  if (map.isIdentity())
    return input;

  vector<z3::expr> output;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    auto ade = map.getResult(i).dyn_cast<mlir::AffineDimExpr>();
    output.push_back(input[ade.getPosition()]);
  }
  return output;
}

static vector<z3::expr> addOne(vector<z3::expr> &&vec) {
  for (unsigned i = 0; i < vec.size(); ++i) {
    uint64_t v;
    if (vec[i].is_bv() && vec[i].is_numeral_u64(v))
      vec[i] = ctx.bv_val(v + 1, vec[i].get_sort().bv_size());
    else
      vec[i] = vec[i] + 1;
  }
  return vec;
}

static optional<string> encodeParallelLoopBodyAndOutput(
    State &newst, mlir::Block &block, const mlir::AffineMap &outputMap,
    const mlir::TensorType &outputType, Tensor &t_res) {
  // Encode the loop body
  // TODO: deal with merging UBs and memorys
  auto &ops = block.getOperations();
  mlir::Value yieldedValue;
  for (auto &op: ops) {
    ENCODE(newst, op, mlir::AddFOp);
    ENCODE(newst, op, mlir::MulFOp);
    ENCODE(newst, op, mlir::AddIOp);
    ENCODE(newst, op, mlir::SubIOp);
    ENCODE(newst, op, mlir::MulIOp);
    ENCODE(newst, op, mlir::IndexCastOp);
    ENCODE(newst, op, mlir::AffineApplyOp);
    ENCODE(newst, op, mlir::linalg::IndexOp);
    if (auto op2 = mlir::dyn_cast<mlir::linalg::YieldOp>(op)) {
      yieldedValue = op2.getOperand(0);
      break;
    }
    RET_STR("has an unsupported operation: '" << op << "'");
  }

  auto &scope = newst.linalgGenericScopes.top();
  auto outputIndVars = doMap(scope.indVars, outputMap);
  auto tensorSz = addOne(doMap(scope.indVarUpperBounds, outputMap));
  t_res = Tensor::mkLambda(move(tensorSz), move(outputIndVars),
      newst.regs.getZ3Expr(yieldedValue));

  return {};
}

static optional<string> encodeReductionLoopBodyAndOutput(
    State &newst, mlir::Block &block,
    const mlir::ArrayRef<mlir::Attribute> &indexingMaps,
    const mlir::TensorType &outputType, Tensor &t_res) {
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
  //   %sum = addf %v, %arg_out
  //   yield %sum
  auto lastarg = block.getArgument(block.getNumArguments() - 1);
  assert(!newst.regs.contains(lastarg));

  auto p = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::AddFOp>(m_Any(), m_Val(lastarg)));
  if (!p.match(&ops.back()))
    return errmsg;
  auto sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(0);

  unsigned cnt = 0;
  for (auto &op: ops) {
    if (cnt++ == ops.size() - 2)
      // Don't directly encode %sum
      break;

    ENCODE(newst, op, mlir::AddFOp);
    ENCODE(newst, op, mlir::MulFOp);
    RET_STR("has an unsupported operation" << op);
  }

  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();

  auto &linalgInfo = newst.linalgGenericScopes.top();

  // Represent %v as an element of a tensor.
  Tensor t_v = Tensor::mkLambda(
      addOne(vector(linalgInfo.indVarUpperBounds)),
      vector(linalgInfo.indVars),
      newst.regs.getZ3Expr(sumvar));

  if (llvm::all_of(outputMap.getResults(), [](const mlir::AffineExpr &expr) {
    auto ac = expr.dyn_cast<mlir::AffineConstantExpr>();
    return ac && ac.getValue() == 0;
  })) {
    // in:  (i, j) -> (i, j)
    // out: (i, j) -> (0)
    // =>
    // t_res[0] = sum(\i. t_input[i / n][i % n] , i < m * n)

    // Define this as a splat tensor (num. elems is 1 anyway)
    vector<z3::expr> tensorSz(1, Index(1));
    for (unsigned i = 1; i < outputType.getRank(); ++i)
      tensorSz.push_back(Index(1));
    t_res = Tensor(t_v.sum(), tensorSz);
    return {};
  } else {
    // in:  (i, j) -> (i, j)
    // out: (i, j) -> (i)
    // =>
    // t_res[i] = sum(\j. t_input[i][j] , j < m)

    // Gather affine vars that are unused in the output (e.g. j) first.
    vector<bool> isInputIdxUsed(outputMap.getNumInputs());
    for (unsigned j = 0; j < outputMap.getNumResults(); ++j) {
      auto expr = outputMap.getResult(j);

      if (auto ade = expr.dyn_cast<mlir::AffineDimExpr>()) {
        isInputIdxUsed[ade.getPosition()] = true;
      } else {
        // Output map has an unknown form
        return errmsg;
      }
    }

    vector<z3::expr> boundsForRes;
    vector<z3::expr> indVarsForRes;
    for (unsigned j = 0; j < isInputIdxUsed.size(); ++j) {
      if (!isInputIdxUsed[j]) {
        boundsForRes.push_back(linalgInfo.indVarUpperBounds[j]);
        indVarsForRes.push_back(linalgInfo.indVars[j]);
      }
    }

    auto tensorSz = addOne(doMap(linalgInfo.indVarUpperBounds, outputMap));
    auto t_sum = Tensor::mkLambda(
          addOne(move(boundsForRes)),
          move(indVarsForRes),
          t_v.get(linalgInfo.indVars))
        .sum();

    auto outputIndVars = doMap(linalgInfo.indVars, outputMap);
    t_res = Tensor::mkLambda(move(tensorSz), move(outputIndVars), t_sum);
    return {};
  }
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::GenericOp op) {
  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";

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
           str != mlir::getReductionIteratorTypeName();
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

  Tensor t_res;
  auto indexingMaps = op.indexing_maps().getValue();
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  auto outputType = op.getOutputOperand(0)->get().getType()
      .cast<mlir::TensorType>();

  if (outputMap.isPermutation()) {
    if (auto errmsg = encodeParallelLoopBodyAndOutput(newst, block, outputMap,
          outputType, t_res))
      return errmsg;

  } else {
    if (auto errmsg = encodeReductionLoopBodyAndOutput(newst, block,
          indexingMaps, outputType, t_res))
      return errmsg;
  }

  assert(t_res.getDims().size() != 0);
  newst.linalgGenericScopes.pop();

  if (op.getNumResults() != 0) {
    // NOTE: op's output tensor (op.getOutputOperand()[0]->get()) isn't updated;
    // aqjune talked with mlir people and confirmed
    assert(op.getNumResults() == 1);
    st.regs.add(op.getResult(0), move(t_res));
  }
  return {};
}


static optional<string> encodeRegion(
    State &st, mlir::Region &region, bool printOps) {
  if (!llvm::hasSingleElement(region))
    return "Only a region with one block is supported";

  auto &block = region.front();
  for (auto &op: block) {
    if (printOps)
      llvm::outs() << "  " << op << "\n";
    ENCODE(st, op, mlir::ConstantIndexOp);
    ENCODE(st, op, mlir::ConstantFloatOp);
    ENCODE(st, op, mlir::ConstantOp);

    ENCODE(st, op, mlir::AddFOp);
    ENCODE(st, op, mlir::AddIOp);
    ENCODE(st, op, mlir::IndexCastOp);
    ENCODE(st, op, mlir::MulFOp);
    ENCODE(st, op, mlir::MulIOp);
    ENCODE(st, op, mlir::ReturnOp);
    ENCODE(st, op, mlir::SubIOp);

    ENCODE(st, op, mlir::AffineApplyOp);

    ENCODE(st, op, mlir::tensor::DimOp);
    ENCODE(st, op, mlir::tensor::ExtractOp);

    ENCODE(st, op, mlir::memref::LoadOp);
    ENCODE(st, op, mlir::memref::StoreOp);
    ENCODE(st, op, mlir::memref::TensorLoadOp);

    ENCODE(st, op, mlir::linalg::ConvInputNHWCFilterHWCFOp);
    ENCODE(st, op, mlir::linalg::DotOp);
    ENCODE(st, op, mlir::linalg::FillOp);
    ENCODE(st, op, mlir::linalg::GenericOp);
    ENCODE(st, op, mlir::linalg::IndexOp);
    ENCODE(st, op, mlir::linalg::InitTensorOp);
    ENCODE(st, op, mlir::linalg::MatmulOp);
    ENCODE(st, op, mlir::linalg::TensorCollapseShapeOp);
    ENCODE(st, op, mlir::linalg::TensorExpandShapeOp);
    
    ENCODE(st, op, mlir::shape::ShapeOfOp);
    ENCODE(st, op, mlir::shape::ToExtentTensorOp);

    RET_STR("Unknown op (" << op.getName() << "): " << op);
  }
  if (printOps)
    llvm::outs() << "\n";
  return {};
}

static optional<string> encode(State &st, mlir::FuncOp &fn, bool printOps) {
  return encodeRegion(st, fn.getRegion(), printOps);
}


static void printCounterEx(
    z3::solver &solver, const vector<z3::expr> &params, mlir::FuncOp src,
    mlir::FuncOp tgt, const State &st_src, const State &st_tgt,
    VerificationStep step) {
  auto m = solver.get_model();
  auto or_omit_z3 = [&](const z3::expr &e) -> string {
    string s;
    llvm::raw_string_ostream rso(s);
    rso << e;
    rso.flush();

    if (s.size() > 500)
      return "(omitted)";
    return s;
  };

  llvm::outs() << "<Inputs>\n";

  unsigned n = src.getNumArguments();
  llvm::DenseSet<mlir::Value> args_src, args_tgt;
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    args_src.insert(argsrc);
    args_tgt.insert(tgt.getArgument(i));
    llvm::outs() << "\targ" << argsrc.getArgNumber() << ": "
                 << st_src.regs.findOrCrash(argsrc) << "\n";
  }

  llvm::outs() << "\n<Source's instructions>\n";
  for (auto &[v, e]: st_src.regs) {
    if (args_src.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << e << "\n";
  }

  llvm::outs() << "\n<Target's instructions>\n";
  for (auto &[v, e]: st_tgt.regs) {
    if (args_tgt.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << e << "\n";
  }

  if (st_src.retValue && step == VerificationStep::RetValue) {
    if (src.getNumResults() == 1 &&
        src.getType().getResult(0).isa<mlir::TensorType>()) {
      llvm::outs() << "\n<Returned tensor>\n";

      auto model = solver.get_model();
      auto t_src = get<Tensor>(*st_src.retValue).eval(model);
      auto t_tgt = get<Tensor>(*st_tgt.retValue).eval(model);

      llvm::outs() << "Dimensions (src): " << t_src.getDims() << '\n';
      llvm::outs() << "Dimensions (tgt): " << t_tgt.getDims() << '\n';

      if (params.size() > 0) {
        // More than size mismatch
        assert(params.size() == 1);
        auto param = model.eval(params[0]);
        auto indices = simplifyList(from1DIdx(param, t_src.getDims()));
        llvm::outs() << "Index: " << indices << '\n';
        llvm::outs() << "Element (src): "
                    << or_omit_z3(t_src.get(indices).simplify())
                    << '\n';
        llvm::outs() << "Element (tgt): "
                    << or_omit_z3(t_tgt.get(indices).simplify())
                    << '\n';
      }

    } else {
      llvm::outs() << "\n<Returned value>\n";

      auto model = solver.get_model();
      for (auto &param: params)
        llvm::outs() << "\tIndex: " << model.eval(param) << "\n";
      visit([&](auto &&ret) { llvm::outs() << "\tSrc: " << ret.eval(model) << "\n"; }, *st_src.retValue);
      visit([&](auto &&ret) { llvm::outs() << "\tTgt: " << ret.eval(model) << "\n"; }, *st_tgt.retValue);
    }
  }

  if (step == VerificationStep::Memory) {
    // Print Memory counter example
    auto bid = params[0];
    auto offset = params[1];
    auto model = solver.get_model();
    auto [srcValue, srcSuccess] = st_src.m->load(bid, offset);
    auto [tgtValue, tgtSuccess] = st_tgt.m->load(bid, offset);
    auto srcWritable = st_src.m->getWritable(bid);
    auto tgtWritable = st_tgt.m->getWritable(bid);
    srcValue = model.eval(srcValue, true);
    srcSuccess = model.eval(srcSuccess);
    tgtValue = model.eval(tgtValue, true);
    tgtSuccess = model.eval(tgtSuccess);
    srcWritable = model.eval(srcWritable);
    tgtWritable = model.eval(tgtWritable);

    llvm::outs() << "\n<Source memory state>\n";
    llvm::outs() << "\tMemory[bid: " << model.eval(bid)
      << ", offset: " << model.eval(offset) << "] : "
      << srcValue << ", " << srcWritable <<  "\n";
    llvm::outs() << "\n<Target memory state>\n";
    llvm::outs() << "\tMemory[bid: " << model.eval(bid)
      << ", offset: " << model.eval(offset) << "] : "
      << tgtValue << ", " << tgtWritable <<  "\n\n";
  }

#if FALSE
  llvm::outs() << solver.get_model().to_string() << "\n";
#endif
}


static pair<z3::check_result, int64_t> solve(
    z3::solver &solver, const z3::expr &refinement_negated,
    const string &dumpSMTPath, const string &dump_string_to_suffix) {
  solver.reset();
  solver.add(refinement_negated);

  if (!dumpSMTPath.empty()) {
    ofstream fout(dumpSMTPath + "." + dump_string_to_suffix);
    fout << refinement_negated;
    fout.close();
  }

  auto startTime = chrono::system_clock::now();
  z3::check_result result = solver.check();
  auto elapsedMillisec =
      chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now() - startTime).count();

  return {result, elapsedMillisec};
}

static Results checkRefinement(
    const ValidationInput &vinput,
    const State &st_src, const State &st_tgt, int64_t &elapsedMillisec) {
  mlir::FuncOp src = vinput.src;
  mlir::FuncOp tgt = vinput.tgt;
  auto fnname = src.getName().str();

  auto printErrorMsg = [&](z3::solver &s, z3::check_result res, const char *msg,
                           vector<z3::expr> &&params, VerificationStep step){
    if (res == z3::unknown) {
      llvm::outs() << "== Result: timeout ==\n";
    } else if (res == z3::sat) {
      llvm::outs() << "== Result: " << msg << "\n";
      printCounterEx(s, params, src, tgt, st_src, st_tgt, step);
    } else {
      llvm_unreachable("unexpected result");
    }
  };

  { // 1. Check UB
    auto s = z3::solver(ctx, "QF_UFBV");
    auto not_refines =
        (st_src.isWellDefined && !st_tgt.isWellDefined).simplify();
    auto res = solve(s, not_refines, vinput.dumpSMTPath, fnname + ".1.ub");
    elapsedMillisec += res.second;
    if (res.first != z3::unsat) {
      printErrorMsg(s, res.first, "Source is more defined than target", {}, VerificationStep::UB);
      return res.first == z3::sat ? Results::UB : Results::TIMEOUT;
    }
  }

  { // 2. Check whether src is always UB
    auto s = z3::solver(ctx, "QF_UFBV");
    auto not_ub = st_src.isWellDefined.simplify();
    auto res = solve(s, not_ub, vinput.dumpSMTPath, fnname + ".2.notub");
    elapsedMillisec += res.second;
    if (res.first == z3::unsat) {
      llvm::outs() << "== Result: correct (source is always undefined) ==\n";
      return Results::SUCCESS;
    }
  }

  if (st_src.retValue) { // 3. Check the return values
    auto s = z3::solver(ctx, "QF_UFBV");

    z3::expr refines(ctx);
    vector<z3::expr> params;
    visit([&](auto &&src, auto &&tgt) {
      auto typedTarget = (decltype(src)) tgt;
      tie(refines, params) = src.refines(typedTarget);
    }, *st_src.retValue, *st_tgt.retValue);

    auto not_refines =
      (st_src.isWellDefined && st_tgt.isWellDefined && !refines).simplify();
    auto res = solve(s, not_refines, vinput.dumpSMTPath, fnname + ".3.retval");
    elapsedMillisec += res.second;
    if (res.first != z3::unsat) {
      printErrorMsg(s, res.first, "Return value mismatch", move(params), VerificationStep::RetValue);
      return res.first == z3::sat ? Results::RETVALUE : Results::TIMEOUT;
    }
  }

  if (st_src.m->getNumBlocks() > 0 ||
      st_tgt.m->getNumBlocks() > 0) { // 4. Check memory refinement
    auto s = z3::solver(ctx, "QF_UFBV");
    auto [refines, params] = st_src.m->refines(*st_tgt.m);
    auto not_refines =
      (st_src.isWellDefined && st_tgt.isWellDefined && !refines).simplify();
    auto res = solve(s, not_refines, vinput.dumpSMTPath, fnname + ".4.memory");
    elapsedMillisec += res.second;
    if (res.first != z3::unsat) {
      printErrorMsg(s, res.first, "Memory mismatch", move(params), VerificationStep::Memory);
      return res.first == z3::sat ? Results::RETVALUE : Results::TIMEOUT;
    }
  }

  llvm::outs() << "== Result: correct ==\n";
  return Results::SUCCESS;
}

static Results tryValidation(
    const ValidationInput &vinput, bool printOps, int64_t &elapsedMillisec) {
  auto src = vinput.src, tgt = vinput.tgt;
  auto raiseUnsupported = [](const string &msg) {
    llvm::errs() << msg << "\n";
    exit(1);
  };

  if (auto errmsg = checkFunctionSignatures(src, tgt))
    raiseUnsupported(*errmsg);

  ArgInfo args;

  auto st_src_or_err = createInputState(src, vinput.numBlocks, vinput.encoding, args);
  if (holds_alternative<string>(st_src_or_err))
    raiseUnsupported(get<string>(st_src_or_err));
  auto st_src = get<State>(st_src_or_err);

  auto st_tgt_or_err = createInputState(tgt, vinput.numBlocks, vinput.encoding, args);
  if (holds_alternative<string>(st_tgt_or_err))
    raiseUnsupported(get<string>(st_tgt_or_err));
  auto st_tgt = get<State>(st_tgt_or_err);

  if (printOps)
    llvm::outs() << "<src>\n";
  if (auto msg = encode(st_src, src, printOps))
    raiseUnsupported(*msg);

  if (printOps)
    llvm::outs() << "<tgt>\n";
  if (auto msg = encode(st_tgt, tgt, printOps))
    raiseUnsupported(*msg);

  auto res = checkRefinement(vinput, st_src, st_tgt, elapsedMillisec);
  return res;
}

static Results validate(ValidationInput vinput) {
  llvm::outs() << "Function " << vinput.src.getName() << "\n\n";
  assert(vinput.src.getNumArguments() == vinput.tgt.getNumArguments());

  int64_t elapsedMillisec = 0;
  Defer timePrinter([&]() {
    llvm::outs() << "solver's running time: " << elapsedMillisec << " msec.\n";
  });

  aop::setAbstractionLevel(aop::FULLY_ABS);
  auto res = tryValidation(vinput, true, elapsedMillisec);
  if (res.code == Results::SUCCESS || res.code == Results::TIMEOUT)
    return res;

  auto usedOps = aop::getUsedAbstractOps();
  if (usedOps.dot && usedOps.sum)
    // dot = mul + sum
    aop::setAbstractionLevel(aop::SUM_MUL);
  else
    return res;

  // Try more precise encoding
  llvm::outs()
      << "\n===============================================================\n"
      << "  Giving more precise semantics to abstractly defined ops...\n"
      << "===============================================================\n\n";
  return tryValidation(vinput, false, elapsedMillisec);
}


Results validate(
    mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
    const string &dumpSMTPath, unsigned int numBlocks, MemEncoding encoding) {
  map<llvm::StringRef, mlir::FuncOp> srcfns, tgtfns;
  auto fillFns = [](map<llvm::StringRef, mlir::FuncOp> &m, mlir::Operation &op) {
    auto fnop = mlir::dyn_cast<mlir::FuncOp>(op);
    if (fnop.isDeclaration())
      return;
    m[fnop.getName()] = fnop;
  };
  llvm::for_each(*src, [&](auto &op) { fillFns(srcfns, op); });
  llvm::for_each(*tgt, [&](auto &op) { fillFns(tgtfns, op); });

  Results verificationResult = Results::SUCCESS;
  for (auto [name, srcfn]: srcfns) {
    auto itr = tgtfns.find(name);
    if (itr == tgtfns.end()) {
      // The function does not exist in tgt! Let's skip this.
      // TODO: we should notify users that the functions are not checked.
      continue;
    }
    // TODO: check fn signature

    ValidationInput vinput;
    vinput.src = srcfn;
    vinput.tgt = itr->second;
    vinput.dumpSMTPath = dumpSMTPath;
    vinput.numBlocks = numBlocks;
    vinput.encoding = encoding;

    verificationResult.merge(validate(vinput));
  }

  return verificationResult;
}
