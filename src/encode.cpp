#include "encode.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/TensorOps.h.inc"
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


static optional<Expr> getZero(mlir::Type eltType) {
  if (eltType.isa<mlir::FloatType>())
    return Float(0.0);
  else if (eltType.isa<mlir::IntegerType>())
    return Integer(0, eltType.getIntOrFloatBitWidth());
  else if (eltType.isa<mlir::IndexType>())
    return Index(0);
  return {};
}

static optional<ValueTy> attrToValueTy(mlir::Attribute a) {
  auto ty = a.getType();
  if (ty.isa<mlir::FloatType>()) {
    return Float(a.dyn_cast<mlir::FloatAttr>().getValueAsDouble());
  } else if (ty.isa<mlir::IntegerType>()) {
    if (64 < ty.getIntOrFloatBitWidth())
      // size is too large
      return {};

    return Integer(a.dyn_cast<mlir::IntegerAttr>().getValue());
  } else if (ty.isa<mlir::IndexType>()) {
    llvm::APInt i = a.dyn_cast<mlir::IntegerAttr>().getValue();
    assert(i.getBitWidth() == 64);
    // TODO: The result may not fit in Index::BITS
    return Index(i.getSExtValue());
  }
  return {};
}

static optional<ValueTy> fromExpr(Expr &&e, mlir::Type ty) {
  if (ty.isa<mlir::IndexType>())
    return Index(e);
  else if (ty.isa<mlir::Float32Type>())
    return Float(e);
  else if (ty.isa<mlir::IntegerType>()) {
    assert(e.sort().bitwidth() == ty.getIntOrFloatBitWidth());
    return Integer(e);
  }
  return {};
}

static vector<Expr> createBoundIndexVars(unsigned n) {
  vector<Expr> idxs;
  for (unsigned i = 0; i < n; i ++) {
    idxs.push_back(
      Index::var("i" + std::to_string(i), VarType::BOUND));
  }
  return idxs;
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


#define ENCODE(st, op, ty) { \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    auto errmsg = encodeOp(st, op2); \
    if (errmsg) { \
      RET_STR("Unknown op (" << op.getName() << "): " << op << "\n\t" << *errmsg << "\n") \
    } \
    continue; \
  } \
}

template<class T>
static optional<string> encodeOp(State &st, T op);

template<>
optional<string>
encodeOp(State &st, mlir::linalg::Conv2DNhwcHwcfOp op) {
  vector<Expr> strides, dilations;
  // TODO: The result may not fit in Index::BITS
  for (auto s: op.strides())
    strides.push_back(Index(s.getSExtValue()));
  for (auto d: op.dilations())
    dilations.push_back(Index(d.getSExtValue()));

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

  auto t_res = t_input.conv(t_filter, strides, dilations);
  st.regs.add(op.getResult(0), move(t_res));

  return {};
}

template<>
optional<string>
encodeOp(State &st, mlir::linalg::ConvOp op) {
  vector<Expr> strides, dilations;
  for (unsigned i = 0; i < op.getNumSpatialDimensions(); i ++) {
    strides.push_back(Index(op.getStride(i)));
    dilations.push_back(Index(op.getDilation(i)));
  }
  auto input = st.regs.get<MemRef>(op.input());
  auto filter = st.regs.get<MemRef>(op.filter());
  auto output = st.regs.get<MemRef>(op.output());

  if (!output.isIdentityMap())
    return "Currently output MemRef should have identity layout..";

  auto success = output.conv(input, filter, strides, dilations);
  // add well defined
  st.wellDefined(op, move(success));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::InitTensorOp op) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::TensorType>();
  assert(ty);

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

  auto elemTy = Tensor::getElemTy(ty);
  if (!elemTy)
    return "Unsupported tensor type";

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  st.regs.add(res,
      Tensor(string("init_tensor_") + to_string(new_var_idx++), sizes,
             *elemTy));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorCollapseShapeOp op) {
  Tensor t = st.regs.get<Tensor>(op.getOperand());
  mlir::RankedTensorType resTy = op.getResultType();

  auto reassocExprs = op.getReassociationIndices();
  assert(reassocExprs.size() == resTy.getRank());

  // If the collapsed size does not match op.getResultType(), it is UB.
  vector<Expr> newDims;
  for (unsigned i = 0; i < reassocExprs.size(); ++i) {
    Expr size = Index::one();
    for (auto &idx: reassocExprs[i])
      size = size * t.getDim(idx);

    if (resTy.getDimSize(i) != mlir::TensorType::kDynamicSize)
      st.wellDefined(op.getOperation(), size == resTy.getDimSize(i));
    newDims.push_back(move(size));
  }

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

    if (Index::BITS < 64 && const_size >= (1ull << Index::BITS))
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

  auto res = Tensor(elems);
  st.regs.add(op.getResult(), move(res));
  return {};
}

static variant<string, MemRef> createNewLocalBlk(
    Memory *m, vector<Expr> &&dims, mlir::MemRefType memrefTy, bool writable) {
  auto elemtyOrNull = MemRef::getElemTy(memrefTy);
  auto layoutOrNull = MemRef::getLayout(memrefTy, dims);
  if (!elemtyOrNull)
    return "unsupported element type";
  else if (!layoutOrNull)
    return "unsupported layout";

  // Add a new local block
  auto bid = m->addLocalBlock(smt::get1DSize(dims), Expr::mkBool(writable));
  // Create MemRef which points to the newly created block
  auto memref =
      MemRef(m, bid, Index::zero(), dims, move(*layoutOrNull),
             move(*elemtyOrNull));

  return {move(memref)};
}

template<>
optional<string> encodeOp(State &st, mlir::memref::AllocOp op) {
  auto memrefTy = op.getType().cast<mlir::MemRefType>();
  if (!memrefTy.getAffineMaps().empty())
    return "unsupported memref type for alloc: it has an affine map";

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
  vector<smt::Expr> sizes, offsets, strides;

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
  if (memrefTy.getAffineMaps().empty()) {
    // memref with identity map
    auto success = memref.storeArray(tensor.asArray(), Index::zero(),
        tensor.get1DSize(), false);
    st.wellDefined(op, move(success));

  } else {
    // TODO: can we further optimize this if we know that memref is a
    // freshly created block?
    // We may not need to preserve the 'previous' bytes.

    vector<Expr> idxs = createBoundIndexVars(memrefTy.getRank());
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
  vector<Expr> idxs = createBoundIndexVars(dims.size());
  auto [Expr, success] = m.get(idxs);
  Tensor t_res = Tensor::mkLambda(move(dims), move(idxs), Expr);

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
optional<string> encodeOp(State &st, mlir::linalg::IndexOp op) {
  uint64_t i = op.dim();
  assert(i < st.linalgGenericScopes.top().indVars.size());
  Expr idxvar = st.linalgGenericScopes.top().indVars[i];
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
  auto res = Tensor(st.regs.getExpr(op.getOperand(0)), t.getDims());
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
  st.regs.add(op.getResult(0), Tensor(move(res), move(outputDim)));
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
    State &st, mlir::Value res, const Expr &e, bool isIndex) {
  if (isIndex)
    st.regs.add(res, Index(e));
  else
    st.regs.add(res, Integer(e));
}

template<>
optional<string> encodeOp(State &st, mlir::AddIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a + b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::SubIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a - b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::MulIOp op) {
  auto a = st.regs.getExpr(op.getOperand(0));
  auto b = st.regs.getExpr(op.getOperand(1));
  addIntOrIndex(st, op, a * b, op.getType().isIndex());
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::IndexCastOp op) {
  auto src = st.regs.getExpr(op.getOperand());
  assert(src.sort().isBV());
  unsigned srcWidth = src.sort().bitwidth();

  unsigned destWidth = 0;
  if (auto dstty = op.getType().dyn_cast<mlir::IntegerType>())
    destWidth = dstty.getWidth();
  else {
    assert(op.getType().isa<mlir::IndexType>());
    destWidth = Index::BITS;
  }

  Expr casted = src;
  if (srcWidth > destWidth)
    casted = src.extract(destWidth - 1, 0);
  else if (srcWidth < destWidth)
    casted = Expr::mkBV(0, destWidth - srcWidth).concat(casted);
  st.regs.add(op, Integer(casted));
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

    // A constant tensor's type cannot have unknown dimensions
    auto dims = ShapedValue::getDims(
        op.getType().cast<mlir::TensorType>(), false);
    auto v = attrToValueTy(denseAttr.getSplatValue());
    if (!v)
      return "unsupported constant";

    st.regs.add(op, Tensor(getExpr(*v), move(dims)));
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
    auto rank = sparseType.getRank();
    vector<uint64_t> dims;
    for (unsigned i = 0; i < rank; ++i)
      dims.push_back(sparseType.getDimSize(i));

    // Unspecified locations are filled with zero.
    auto zero = getZero(sparseType.getElementType());
    if (!zero)
      return "unsupported element type";

    vector<vector<uint64_t>> sparseIndices;
    vector<smt::Expr> sparseValues;

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
    st.regs.add(op, Tensor(sparseIndices, sparseValues, dims, *zero));
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
  auto outputMap = indexingMaps.back().cast<mlir::AffineMapAttr>().getValue();
  auto &block = *op.region().begin();

  const vector<Expr> &inductionVars = st.linalgGenericScopes.top().indVars;

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

        auto [t_elem, inbounds] = t_input.get(affine_Exprs);
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
      auto [m_elem, success] = m_input.get(affine_Exprs);
      st.regs.add(block.getArgument(arg_i), Float(m_elem));
    } else {
      return "unsupported block argument type";
    }
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

static vector<Expr> addOne(vector<Expr> &&vec) {
  for (unsigned i = 0; i < vec.size(); ++i) {
    uint64_t v;
    if (vec[i].sort().isBV() && vec[i].isUInt(v))
      vec[i] = Expr::mkBV(v + 1, vec[i].sort().bitwidth());
    else
      vec[i] = vec[i] + 1;
  }
  return vec;
}

static optional<string> encodeParallelLoopBodyAndOutput(
    State &newst, mlir::Block &block, const mlir::AffineMap &outputMap,
    const mlir::ShapedType &outputType, optional<Tensor> &t_res) {
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
      newst.regs.getExpr(yieldedValue));

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
  //   %sum = addf %v, %arg_out or  %sum = addf %arg_out, %v
  //   yield %sum
  auto lastarg = block.getArgument(block.getNumArguments() - 1);
  assert(!newst.regs.contains(lastarg));

  auto p1 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::AddFOp>(m_Val(lastarg), m_Any()));
  auto p2 = m_Op<mlir::linalg::YieldOp>(
      m_Op<mlir::AddFOp>(m_Any(), m_Val(lastarg)));

  mlir::Value sumvar;
  if (p1.match(&ops.back()))
    sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(1);
  else if (p2.match(&ops.back()))
    sumvar = ops.back().getOperand(0).getDefiningOp()->getOperand(0);
  else
    return errmsg;

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
    t_res = Tensor(t_v.sum(), makeCube(Index(1), outputType.getRank()));
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
          addOne(move(boundsForRes)),
          move(indVarsForRes),
          t_v.get(linalgInfo.indVars).first)
        .sum();

    auto outputIndVars = doMap(linalgInfo.indVars, outputMap);
    t_res = Tensor::mkLambda(move(tensorSz), move(outputIndVars), t_sum);
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
    ENCODE(st, op, mlir::tensor::InsertOp);
    ENCODE(st, op, mlir::tensor::ExtractOp);
    ENCODE(st, op, mlir::tensor::FromElementsOp);

    ENCODE(st, op, mlir::memref::AllocOp);
    ENCODE(st, op, mlir::memref::LoadOp);
    ENCODE(st, op, mlir::memref::StoreOp);
    ENCODE(st, op, mlir::memref::SubViewOp);
    ENCODE(st, op, mlir::memref::BufferCastOp);
    ENCODE(st, op, mlir::memref::TensorLoadOp);
    ENCODE(st, op, mlir::memref::TensorStoreOp);

    ENCODE(st, op, mlir::linalg::Conv2DNhwcHwcfOp);
    ENCODE(st, op, mlir::linalg::ConvOp);
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
