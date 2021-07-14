#include "value.h"
#include "smt.h"
#include "state.h"
#include "vcgen.h"

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
};

static variant<string, State>
createInputState(mlir::FuncOp fn) {
  State s;
  s.isWellDefined = ctx.bool_val(true);

  unsigned n = fn.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    auto argty = arg.getType();

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
      s.regs.add(arg, MemRef("arg0", dimsAndElemTy->first, dimsAndElemTy->second));

    } else if (auto ty = argty.dyn_cast<mlir::IndexType>()) {
      s.regs.add(arg, Index("arg" + to_string(arg.getArgNumber())));

    } else if (auto ty = argty.dyn_cast<mlir::FloatType>()) {
      s.regs.add(arg, Float("arg" + to_string(arg.getArgNumber())));

    } else {
      RET_STR("Unsupported type: " << arg.getType());
    }
  }

  return s;
}


template<class T>
optional<z3::expr> encodeAffineExpr(
    mlir::AffineExpr ae, const vector<T> &dimvars
) {
  switch (ae.getKind()) {
  case mlir::AffineExprKind::Add: {
    auto aboe = ae.dyn_cast<mlir::AffineBinaryOpExpr>();
    auto lhs = encodeAffineExpr(aboe.getLHS(), dimvars);
    auto rhs = encodeAffineExpr(aboe.getRHS(), dimvars);
    if (!lhs || !rhs)
      return {};
    return *lhs + *rhs;
  }
  case mlir::AffineExprKind::DimId: {
    auto ade = ae.dyn_cast<mlir::AffineDimExpr>();
    auto id = ade.getPosition();
    assert(id < dimvars.size());
    return dimvars[id];
  }
  case mlir::AffineExprKind::Constant: {
    auto ac = ae.dyn_cast<mlir::AffineConstantExpr>();
    return Index(ac.getValue());
  }
  default:
    // Unsupported
    return {};
  }
}



#define ENCODE(st, op, ty) { \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    auto errmsg = encodeOp(st, op2); \
    if (errmsg) { \
      RET_STR("Cannot encode " << op << "\n\t" << *errmsg << "\n"); \
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

  auto dimsAndElemTy = Tensor::getDimsAndElemTy(ty);
  if (!dimsAndElemTy)
    return "Unsupported tensor type";

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  auto name = string("init_tensor_") + to_string(new_var_idx++);
  st.regs.add(res, Tensor(name, dimsAndElemTy->first, dimsAndElemTy->second));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorCollapseShapeOp op) {
  // TODO: is tensor_collapse_shape with permutated indices valid?
  //   ex: %2 = linalg.tensor_collapse_shape %1 [[0, 2], [1, 3]]
  // Then, it isn't simply reinterpretation of the operand; it needs permutation
  // of elements.

  Tensor t = st.regs.get<Tensor>(op.getOperand());
  st.regs.add(op.getResult(), t.reshape(getDims(op.getResultType())));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorExpandShapeOp op) {
  // TODO: is tensor_expand_shape with permutated indices valid?
  //   ex: %2 = linalg.tensor_expand_shape %1 [[0], [2], [1, 3]]
  // Then, it isn't simply reinterpretation of the operand; it needs permutation
  // of elements.

  Tensor t = st.regs.get<Tensor>(op.getOperand());
  st.regs.add(op.getResult(), t.reshape(getDims(op.getResultType())));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::MatmulOp op) {
  if (!op.hasTensorSemantics())
    return "operation with tensor semantics is supported only";

  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    return "unsupported form";

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

  z3::expr wb = ctx.bool_val(true);
  for (unsigned i = 0; i < indices.size(); ++i)
    // TODO: revisit this; may not be axis-wise
    wb = wb && z3::ult(indices[i], t.getDim(i));

  st.isWellDefined = st.isWellDefined && wb;

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::LoadOp op) {
  // TODO: The MLIR doc isn't explicit about what happens if indices are
  // out-of-bounds. It is currently encoded as UB.

  const Memory &memory = st.m;
  auto m = st.regs.get<MemRef>(op.getOperand(0));
  vector<z3::expr> indices;
  for (auto idx0: op.indices())
    indices.emplace_back(st.regs.get<Index>(idx0));

  if (op.getType().isa<mlir::Float32Type>()) {
    auto [expr, success] = m.get(memory, indices);
    st.regs.add(op, Float(expr));
    st.isWellDefined = st.isWellDefined && success;
  }
  else
    // TODO: how to do this well?
    return "unsupported type";

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::TensorLoadOp op) {
  auto m = st.regs.get<MemRef>(op.getOperand());

  // step1. MemBlock which contains source memref marks as not writable.
  auto memBlock = st.m.getMemBlock(m.getBID());
  memBlock.writable = ctx.bool_val(false);

  st.m.getMemBlock(m.getBID()).writable = ctx.bool_val(false);

  // step2. create new Tensor that alias origin memref using Tensor::mkLambda
  auto dims = m.getDims();
  auto memrefSize = get1DSize(dims);
  vector<z3::expr> idxs;
  for (int i = 0; i < dims.size(); i ++) {
    idxs.push_back(Index("Index_" + std::to_string(i)));
  }
  auto [expr, success] = m.get(st.m, idxs);
  Tensor t_res = Tensor::mkLambda(move(dims), move(idxs), expr);

  // step3. add result tensor to register
  st.regs.add(op.getResult(), t_res);
  st.isWellDefined = st.isWellDefined &&
    z3::uge(memBlock.numelem, memrefSize) &&
    z3::ult(m.getOffset(), memBlock.numelem - memrefSize);

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::IndexOp op) {
  uint64_t i = op.dim();
  assert(i < st.linalgGenericScopes.top().size());
  z3::expr idxvar = st.linalgGenericScopes.top()[i];
  st.regs.add(op, Index(idxvar));
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

template<>
optional<string> encodeOp(State &st, mlir::AddIOp op) {
  auto a = st.regs.get<Integer>(op.getOperand(0));
  auto b = st.regs.get<Integer>(op.getOperand(1));
  st.regs.add(op, Integer((z3::expr)a + (z3::expr)b));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::SubIOp op) {
  auto a = st.regs.get<Integer>(op.getOperand(0));
  auto b = st.regs.get<Integer>(op.getOperand(1));
  st.regs.add(op, Integer((z3::expr)a - (z3::expr)b));
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

    auto dims = getDims(op.getType().cast<mlir::TensorType>());
    st.regs.add(op, Tensor(Float(splatfval.getValueAsDouble()), move(dims)));
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


static optional<string>
encodeUBForTensorShapeMatch(State &st, mlir::linalg::GenericOp op) {
  // In high-level:
  // (1) The size of the loop is calculated (analogous to what
  //     LinalgOp::createLoopRanges does).
  // (2) UB is encoded so that the elements of the tensors are accessed
  //     in-bounds.

  // Note that the process of getting the size of the loop is unclear;
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
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();

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

  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto ae = encodeAffineExpr(map.getResult(idx), res);
    if (!ae)
      return "unsupported affine expr";

    z3::expr inbounds = z3::ult(*ae, (z3::expr)viewSizes[idx]);
    st.isWellDefined = st.isWellDefined && inbounds;
  }

  return {};
}

static optional<string> initInputStateForLoopBody(
    State &st, mlir::linalg::GenericOp op) {
  auto indexingMaps = op.indexing_maps().getValue();
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  auto &block = *op.region().begin();

  vector<z3::expr> output_dimvars;
  for (unsigned i = 0; i < outputMap.getNumInputs(); ++i)
    output_dimvars.emplace_back(Index("i" + to_string(i)));

  // Fill in args
  assert(op.getInputOperands().size() + op.getNumOutputs() ==
         indexingMaps.size());
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

      if (!elemty.isa<mlir::IntegerType>() &&
          !elemty.isa<mlir::Float32Type>()) {
        return "unsupported element type";
      }

      auto toRegValue = [&elemty](const z3::expr &e) -> ValueTy {
        if (elemty.isa<mlir::Float32Type>())
          return Float(e);
        else if (elemty.isa<mlir::IntegerType>())
          return Integer(e);
        llvm_unreachable("unexpected elemty");
      };

      Tensor t_input = st.regs.get<Tensor>(op_i);

      if (inputMap.getNumResults() == 0) {
        // A tensor with a single element; e.g. tensor<f32>.
        st.regs.add(block.getArgument(arg_i),
                       toRegValue(t_input.get({Index::zero()})));
      } else {
        vector<z3::expr> affine_exprs;
        for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
          auto ae_res = encodeAffineExpr(inputMap.getResult(i), output_dimvars);
          if (!ae_res)
            RET_STR_WITH_PREFIX("unsupported affine expr ",
                                inputMap.getResult(i));

          affine_exprs.emplace_back(move(*ae_res));
        }

        auto t_elem = t_input.get(affine_exprs);
        st.regs.add(block.getArgument(arg_i), toRegValue(t_elem));
      }
    } else {
      return "unsupported block argument type";
    }
  }

  st.linalgGenericScopes.push(move(output_dimvars));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::GenericOp op) {
  if (!op.hasTensorSemantics())
    return "operation with tensor semantics is supported only";

  if (op.getNumOutputs() != 1)
    return "operation with single output is supported only";

  auto &region = op.region();
  if (!llvm::hasSingleElement(region))
    return "operation with one block is supported only";

  auto &block = region.front();
  if (!std::all_of(block.args_begin(), block.args_end(),
      [](auto &arg) { return arg.getType().isSignlessIntOrFloat(); }))
    return "unsupported block arguments";

  encodeUBForTensorShapeMatch(st, op);

  // Start from newst
  State newst = st;
  if (auto msg = initInputStateForLoopBody(newst, op))
    return msg;

  auto indexingMaps = op.indexing_maps().getValue();
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  if (!outputMap.isPermutation())
    return "permutation output map is supported only";

  // Encode the loop body
  // TODO: deal with merging UBs and memorys
  auto &ops = block.getOperations();
  mlir::Value yieldedValue;
  for (auto &op: ops) {
    ENCODE(newst, op, mlir::AddFOp);
    ENCODE(newst, op, mlir::MulFOp);
    ENCODE(newst, op, mlir::AddIOp);
    ENCODE(newst, op, mlir::SubIOp);
    ENCODE(newst, op, mlir::IndexCastOp);
    ENCODE(newst, op, mlir::linalg::IndexOp);
    if (auto op2 = mlir::dyn_cast<mlir::linalg::YieldOp>(op)) {
      yieldedValue = op2.getOperand(0);
      break;
    }
    RET_STR("has an unsupported operation: " << op);
  }

  // NOTE: op's output tensor (op.getOutputOperand()[0]->get()) isn't updated;
  // aqjune talked with mlir people and it is confirmed by them

  auto output_dimvars = move(newst.linalgGenericScopes.top());
  newst.linalgGenericScopes.pop();

  if (!outputMap.isIdentity()) {
    vector<z3::expr> newvars;
    for (unsigned i = 0; i < outputMap.getNumResults(); ++i) {
      auto ade = outputMap.getResult(i).dyn_cast<mlir::AffineDimExpr>();
      newvars.emplace_back(output_dimvars[ade.getPosition()]);
    }
    output_dimvars = move(newvars);
  }

  auto tensor_sz = getDims(
      op.getOutputOperand(0)->get().getType().cast<mlir::TensorType>());
  Tensor t_res = Tensor::mkLambda(move(tensor_sz), move(output_dimvars),
      newst.regs.getZ3Expr(yieldedValue));
  st.regs.add(op.getResult(0), move(t_res));
  return {};
}


static optional<string> encodeRegion(State &st, mlir::Region &region) {
  if (!llvm::hasSingleElement(region))
    return "Only a region with one block is supported";

  auto &block = region.front();
  for (auto &op: block) {
    llvm::outs() << "  " << op << "\n";
    ENCODE(st, op, mlir::ConstantIndexOp);
    ENCODE(st, op, mlir::ConstantFloatOp);
    ENCODE(st, op, mlir::ConstantOp);

    ENCODE(st, op, mlir::AddFOp);
    ENCODE(st, op, mlir::AddIOp);
    ENCODE(st, op, mlir::IndexCastOp);
    ENCODE(st, op, mlir::MulFOp);
    ENCODE(st, op, mlir::ReturnOp);
    ENCODE(st, op, mlir::SubIOp);

    ENCODE(st, op, mlir::tensor::DimOp);
    ENCODE(st, op, mlir::tensor::ExtractOp);

    ENCODE(st, op, mlir::memref::LoadOp);
    ENCODE(st, op, mlir::memref::TensorLoadOp);

    ENCODE(st, op, mlir::linalg::IndexOp);
    ENCODE(st, op, mlir::linalg::ConvInputNHWCFilterHWCFOp);
    ENCODE(st, op, mlir::linalg::GenericOp);
    ENCODE(st, op, mlir::linalg::InitTensorOp);
    ENCODE(st, op, mlir::linalg::MatmulOp);
    ENCODE(st, op, mlir::linalg::TensorCollapseShapeOp);
    ENCODE(st, op, mlir::linalg::TensorExpandShapeOp);
    
    ENCODE(st, op, mlir::shape::ShapeOfOp);
    ENCODE(st, op, mlir::shape::ToExtentTensorOp);

    RET_STR("Unknown op (" << op.getName() << "): " << op);
  }
  llvm::outs() << "\n";
  return {};
}

static optional<string> encode(State &st, mlir::FuncOp &fn) {
  return encodeRegion(st, fn.getRegion());
}


static void printCounterEx(
    z3::solver &solver, const vector<z3::expr> &params, mlir::FuncOp src,
    State &st_src, State &st_src_in, State &st_tgt, State &st_tgt_in) {
  auto m = solver.get_model();
  auto or_omit = [&](const ValueTy &val) -> string {
    ValueTy evaluatedVal;
    visit([&](auto &&v) { evaluatedVal = v.eval(m); }, val);

    string s;
    llvm::raw_string_ostream rso(s);
    visit([&](auto &&v) { rso << v; }, evaluatedVal);
    rso.flush();

    if (s.size() > 500)
      return "(omitted)";
    return s;
  };

  llvm::outs() << "<Inputs>\n";

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    llvm::outs() << "\targ" << argsrc.getArgNumber() << ": "
                 << or_omit(st_src_in.regs.findOrCrash(argsrc)) << "\n";
  }

  llvm::outs() << "\n<Source's variables>\n";
  for (auto &[v, e]: st_src.regs) {
    if (st_src_in.regs.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << or_omit(e) << "\n";
  }

  llvm::outs() << "\n<Target's variables>\n";
  for (auto &[v, e]: st_tgt.regs) {
    if (st_tgt_in.regs.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << or_omit(e) << "\n";
  }

  if (st_src.retValue) {
    llvm::outs() << "\n<Return values>\n";
    for (auto &param: params)
      llvm::outs() << "\tIndex: " << solver.get_model().eval(param) << "\n";
    llvm::outs() << "\tSrc: " << or_omit(*st_src.retValue)
        << "\n"
        << "\tTgt: " << or_omit(*st_tgt.retValue)
        << "\n";
  }

#if FALSE
  llvm::outs() << solver.get_model().to_string() << "\n";
#endif
}


static pair<z3::check_result, int64_t> solve(
    z3::solver &solver, const z3::expr &refinement_negated,
    const string &dump_smt_to, const string &dump_string_to_suffix) {
  solver.reset();
  solver.add(refinement_negated);

  if (!dump_smt_to.empty()) {
    ofstream fout(dump_smt_to + "." + dump_string_to_suffix);
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

static Results verifyFunction(
    mlir::FuncOp src, mlir::FuncOp tgt, const string &dump_smt_to) {
  llvm::outs() << "Function " << src.getName() << "\n\n";
  assert(src.getNumArguments() == tgt.getNumArguments());

  auto raiseUnsupported = [](const string &msg) {
    llvm::errs() << msg << "\n";
    exit(1);
  };

  auto st_src_or_err = createInputState(src);
  if (holds_alternative<string>(st_src_or_err))
    raiseUnsupported(get<string>(st_src_or_err));
  auto st_src = get<State>(st_src_or_err);
  auto st_src_in = st_src; // for printing counter ex.

  auto st_tgt_or_err = createInputState(tgt);
  if (holds_alternative<string>(st_tgt_or_err))
    raiseUnsupported(get<string>(st_tgt_or_err));
  auto st_tgt = get<State>(st_tgt_or_err);
  auto st_tgt_in = st_tgt; // for printing counter ex.

  llvm::outs() << "<src>\n";
  if (auto msg = encode(st_src, src))
    raiseUnsupported(*msg);

  llvm::outs() << "<tgt>\n";
  if (auto msg = encode(st_tgt, tgt))
    raiseUnsupported(*msg);


  auto fnname = src.getName().str();
  int64_t elapsedMillisec = 0;

  Defer timePrinter([&]() {
    llvm::outs() << "solver's running time: " << elapsedMillisec << " msec.\n";
  });
  auto printErrorMsg = [&](z3::solver &s, z3::check_result res, const char *msg,
                           vector<z3::expr> &&params){
    if (res == z3::unknown) {
      llvm::outs() << "== Result: timeout ==\n";
    } else if (res == z3::sat) {
      llvm::outs() << "== Result: " << msg << "\n";
      printCounterEx(s, params, src, st_src, st_src_in, st_tgt, st_tgt_in);
    } else {
      llvm_unreachable("unexpected result");
    }
  };

  { // 1. Check UB
    auto s = z3::solver(ctx, "QF_UFBV");
    auto not_refines =
        (st_src.isWellDefined && !st_tgt.isWellDefined).simplify();
    auto res = solve(s, not_refines, dump_smt_to, fnname + ".ub");
    elapsedMillisec += res.second;
    if (res.first != z3::unsat) {
      // Well... let's use Alive2's wording.
      printErrorMsg(s, res.first, "Source is more defined than target", {});
      return res.first == z3::sat ? Results::UB : Results::TIMEOUT;
    }
  }

  if (st_src.retValue) { // 2. Check the return values
    auto s = z3::solver(ctx, "QF_UFBV");

    z3::expr refines(ctx);
    vector<z3::expr> params;
    visit([&](auto &&src, auto &&tgt) {
      auto typedTarget = (decltype(src)) tgt;
      tie(refines, params) = src.refines(typedTarget);
    }, *st_src.retValue, *st_tgt.retValue);

    auto not_refines =
      (st_src.isWellDefined && st_tgt.isWellDefined && !refines).simplify();
    auto res = solve(s, not_refines, dump_smt_to, fnname + ".retval");
    elapsedMillisec += res.second;
    if (res.first != z3::unsat) {
      printErrorMsg(s, res.first, "Return value mismatch", move(params));
      return res.first == z3::sat ? Results::RETVALUE : Results::TIMEOUT;
    }
  }

  llvm::outs() << "== Result: correct ==\n";
  return Results::SUCCESS;
}

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const string &dump_smt_to) {
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
    verificationResult.merge(verifyFunction(srcfn, itr->second, dump_smt_to));
  }

  return verificationResult;
}
