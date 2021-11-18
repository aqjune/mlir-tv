#include "analysis.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <set>
#include <type_traits>

using namespace std;

// Contains absolute values of constants.
static set<llvm::APFloat> constF32Set;
static set<llvm::APFloat> constF64Set;

static void analyzeAttr(const mlir::Attribute &a) {
  assert(!a.isa<mlir::ElementsAttr>());

  auto ty = a.getType();
  if (!ty.isa<mlir::FloatType>())
    return;

  const auto val = a.dyn_cast<mlir::FloatAttr>().getValue();
  if (val.isNaN() || val.isInfinity())
    // Already specially treated in vcgen.cpp.
    return;

  auto val_f32 = val;
  auto val_f64 = val;
  bool is_rounded; // dummy

  if (ty.isF32()) {
    val_f64.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloatBase::rmNearestTiesToEven, &is_rounded);
  } else if (ty.isF64()) {
    val_f32.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloatBase::rmNearestTiesToEven, &is_rounded);
  } else {
      throw UnsupportedException(ty, "Unsupported type");
  }

  if (val_f32.isNegative())
    val_f32.clearSign();
  if (val_f64.isNegative())
    val_f64.clearSign();

  constF32Set.insert(val_f32);
  constF64Set.insert(val_f64);
}

static void analyzeElemAttr(const mlir::ElementsAttr &attr) {
  if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    if (denseAttr.isSplat()) {
      analyzeAttr(denseAttr.getSplatValue<mlir::Attribute>());
    } else {
      for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
        analyzeAttr(attr);
      }
    }
  } else if (auto sparseAttr = attr.dyn_cast<mlir::SparseElementsAttr>()) {
    auto denseAttr = sparseAttr.getValues();
    for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
      analyzeAttr(attr);
    }
  }
}

template<class FT>
static size_t analyzeVariable(const mlir::Value &value) {
  static_assert(is_base_of<mlir::FloatType, FT>::value, "FT must be mlir::FloatType");
  auto ty = value.getType();
  if (ty.isa<FT>()) {
    return 1;

  } else if (ty.isa<mlir::TensorType>()) {
    auto tensorty = ty.cast<mlir::TensorType>();
    if (!tensorty.getElementType().isa<FT>())
      return 0;

    if (tensorty.hasStaticShape()) 
      return tensorty.getNumElements();
    else 
      return Tensor::MAX_TENSOR_SIZE;

  } else if (ty.isa<mlir::MemRefType>()) {
    auto memrefty = ty.cast<mlir::MemRefType>();
    if (!memrefty.getElementType().isa<FT>())
      return 0;

    if (memrefty.hasStaticShape()) 
      return memrefty.getNumElements();
    else 
      return MemRef::MAX_MEMREF_SIZE;
  } else {
    return 0;
  }
}

template<class T>
static void analyzeOp(T op, bool isFullyAbstract);

template<class FT>
static size_t analyzeBlock(mlir::Block &block, bool isFullyAbstract);

template<>
void analyzeOp(mlir::arith::ConstantFloatOp op, bool isFullyAbstract) {
  auto ty = op.getType();
  const auto val = op.value();
  auto val_f32 = val, val_f64 = val;
  bool is_rounded; // dummy

  if (ty.isF32()) {
    val_f64.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloatBase::rmNearestTiesToEven, &is_rounded);
  } else if (ty.isF64()) {
    val_f32.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloatBase::rmNearestTiesToEven, &is_rounded);
  } else {
      throw UnsupportedException(ty, "Unsupported type");
  }

  constF32Set.insert(val_f32);
  constF64Set.insert(val_f64);
}

template<>
void analyzeOp(mlir::arith::ConstantOp op, bool isFullyAbstract) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analyzeElemAttr(eattr);
}

template<>
void analyzeOp(mlir::tosa::ConstOp op, bool isFullyAbstract) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analyzeElemAttr(eattr);
}

template<>
void analyzeOp(mlir::linalg::GenericOp op, bool isFullyAbstract) {
  auto &region = op.region();
  auto &block = region.front();
  analyzeBlock<mlir::Float32Type>(block, isFullyAbstract);
  analyzeBlock<mlir::Float64Type>(block, isFullyAbstract);
}

#define ANALYZE(op, ty, isFullyAbstract) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analyzeOp(op2, isFullyAbstract); \
    continue; \
  }

template<class FT>
static size_t analyzeBlock(mlir::Block &block, bool isFullyAbstract) {
  static_assert(is_base_of<mlir::FloatType, FT>::value,
      "FT must be mlir::FloatType");
  
  size_t fpVarCount = 0;
  for (auto &op: block) {
    // Analyze constant fp operations
    // This operations do not increase fpVarCount
    ANALYZE(op, mlir::arith::ConstantFloatOp, isFullyAbstract);
    ANALYZE(op, mlir::arith::ConstantOp, isFullyAbstract);
    ANALYZE(op, mlir::tosa::ConstOp, isFullyAbstract);

    for (const auto &result: op.getResults())
      fpVarCount += isFullyAbstract ? 1 : analyzeVariable<FT>(result);

    // This operations increase fpVarCount. So it should be executed after for-loop
    ANALYZE(op, mlir::linalg::GenericOp, isFullyAbstract);
  }

  return fpVarCount;
}

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract) {
  SolePrecisionAnalysisResult F32, F64;
  constF32Set.clear();
  constF64Set.clear();

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments()){
    F32.fpArgCount += isFullyAbstract ? 1 : analyzeVariable<mlir::Float32Type>(arg);
    F64.fpArgCount += isFullyAbstract ? 1 : analyzeVariable<mlir::Float64Type>(arg);
  }
    
  // Step2. analyze the block
  auto &block = region.front();
  F32.fpVarCount = analyzeBlock<mlir::Float32Type>(block, isFullyAbstract);
  F64.fpVarCount = analyzeBlock<mlir::Float64Type>(block, isFullyAbstract);

  F32.fpConstSet = move(constF32Set);
  F64.fpConstSet = move(constF64Set);

  return {
    .F32 = F32,
    .F64 = F64
  };
}
