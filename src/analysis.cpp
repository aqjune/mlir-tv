#include "analysis.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <type_traits>

using namespace std;

// Contains absolute values of constants.
static set<llvm::APFloat> constF32Set;
static set<llvm::APFloat> constF64Set;

static void analyzeAPFloat(const mlir::Type ty, const llvm::APFloat val) {
  if (val.isNaN() || val.isInfinity())
    // They cannot be inserted into set<APFloat>.
    // They will be specially treated in setAbstraction() (abstractops.cpp)
    return;

  auto val_f32 = val;
  auto val_f64 = val;
  bool lost_info; // dummy

  llvm::APFloat::opStatus op_status;
  if (ty.isF32()) {
    op_status = val_f64.convert(llvm::APFloat::IEEEdouble(),
                    // doesn't really matter in extension
                    llvm::APFloat::rmTowardZero, &lost_info);
  } else if (ty.isF64()) {
    op_status = val_f32.convert(llvm::APFloat::IEEEsingle(),
                    // floor in case of truncation (ordering issue)
                    llvm::APFloat::rmTowardZero, &lost_info);
  } else {
      throw UnsupportedException(ty, "Unsupported type");
  }

  if (val_f32.isNegative())
    val_f32.clearSign();
  if (val_f64.isNegative())
    val_f64.clearSign();

  // Values beyond the float range are mapped to Inf
  if (!(op_status & llvm::APFloat::opOverflow)) {
    constF32Set.insert(val_f32);
  }
  constF64Set.insert(val_f64);
}

static void analyzeAttr(const mlir::Attribute &a) {
  assert(!a.isa<mlir::ElementsAttr>());

  auto ty = a.getType();
  if (!ty.isa<mlir::FloatType>())
    return;

  const auto val = a.dyn_cast<mlir::FloatAttr>().getValue();
  analyzeAPFloat(ty, val);
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

template<class ValueType>
static size_t analyzeVariable(const mlir::Value &value) {
  auto ty = value.getType();
  if (ty.isa<ValueType>()) {
    return 1;

  } else if (ty.isa<mlir::TensorType>()) {
    auto tensorty = ty.cast<mlir::TensorType>();
    if (!tensorty.getElementType().isa<ValueType>())
      return 0;

    if (tensorty.hasStaticShape()) 
      return tensorty.getNumElements();
    else 
      return Tensor::MAX_TENSOR_SIZE;

  } else if (ty.isa<mlir::MemRefType>()) {
    auto memrefty = ty.cast<mlir::MemRefType>();
    if (!memrefty.getElementType().isa<ValueType>())
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

template<class ValueType>
static size_t analyzeBlock(mlir::Block &block, bool isFullyAbstract);

template<>
void analyzeOp(mlir::arith::ConstantFloatOp op, bool isFullyAbstract) {
  auto ty = op.getType();
  const auto val = op.value();
  analyzeAPFloat(ty, val);
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

template<class ValueType>
size_t analyzeRegion(mlir::Region &region, bool isFullyAbstract) {
  if (!region.hasOneBlock())
    throw UnsupportedException("Region with a single block is supported only");

  auto &block = region.front();
  return analyzeBlock<ValueType>(block, isFullyAbstract);
}

#define ANALYZE(op, ty, isFullyAbstract) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analyzeOp(op2, isFullyAbstract); \
    continue; \
  }

#define ANALYZE_REGION(op, ty, region_fn, isFullyAbstract) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    fpVarCount += analyzeRegion<ValueType>(op2.region_fn(), isFullyAbstract); \
    continue; \
  }

template<class ValueType>
static size_t analyzeBlock(mlir::Block &block, bool isFullyAbstract) {
  size_t fpVarCount = 0;
  for (auto &op: block) {
    // Analyze constant fp operations
    // These operations do not increase fpVarCount
    ANALYZE(op, mlir::arith::ConstantFloatOp, isFullyAbstract);
    ANALYZE(op, mlir::arith::ConstantOp, isFullyAbstract);
    ANALYZE(op, mlir::tosa::ConstOp, isFullyAbstract);

    // Non-constant operations; increase fpVarCount if returning fps
    for (const auto &result: op.getResults()) {
      auto numFps = analyzeVariable<ValueType>(result);
      if (isFullyAbstract && is_base_of<mlir::FloatType, ValueType>::value)
        fpVarCount += numFps ? 1 : 0;
      else
        fpVarCount += numFps;
    }

    // Analyze operations having subregions.
    ANALYZE_REGION(op, mlir::linalg::GenericOp, region, isFullyAbstract);
    ANALYZE_REGION(op, mlir::linalg::PadTensorOp, region, isFullyAbstract);
    ANALYZE_REGION(op, mlir::tensor::GenerateOp, body, isFullyAbstract);
  }

  return fpVarCount;
}

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract) {
  FPAnalysisResult F32, F64;
  MemRefAnalysisResult memref;
  constF32Set.clear();
  constF64Set.clear();

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments()){
    auto numF32 = analyzeVariable<mlir::Float32Type>(arg);
    auto numF64 = analyzeVariable<mlir::Float64Type>(arg);
    if (isFullyAbstract) {
      F32.fpArgCount += numF32 ? 1 : 0;
      F64.fpArgCount += numF64 ? 1 : 0;
    } else {
      F32.fpArgCount += numF32;
      F64.fpArgCount += numF64;
    }
    memref.argCount += analyzeVariable<mlir::MemRefType>(arg);
  }
    
  // Step2. analyze the block
  auto &block = region.front();
  F32.fpVarCount = analyzeBlock<mlir::Float32Type>(block, isFullyAbstract);
  F64.fpVarCount = analyzeBlock<mlir::Float64Type>(block, isFullyAbstract);
  memref.varCount = analyzeBlock<mlir::MemRefType>(block, isFullyAbstract);

  F32.fpConstSet = move(constF32Set);
  F64.fpConstSet = move(constF64Set);

  return {
    .F32 = F32,
    .F64 = F64,
    .memref = memref
  };
}
