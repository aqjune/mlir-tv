#include "analysis.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <set>
#include <type_traits>

using namespace std;

static set<llvm::APFloat> constF32Set;
static int varF32Count = 0;
static int argF32Count = 0;

static set<llvm::APFloat> constF64Set;
static int varF64Count = 0;
static int argF64Count = 0;

static void analyzeAttr(const mlir::Attribute &a) {
  assert(!a.isa<mlir::ElementsAttr>());

  auto ty = a.getType();
  if (ty.isa<mlir::FloatType>()) {
    const auto val = a.dyn_cast<mlir::FloatAttr>().getValue();
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

    constF32Set.insert(val_f32);
    constF64Set.insert(val_f64);
  }
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
static int analyzeVariable(const mlir::Value &value) {
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

#define ANALYSIS(op, ty, isFullyAbstract) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analyzeOp(op2, isFullyAbstract); \
    continue; \
  }

void analyzeBlock(mlir::Block &block, bool isFullyAbstract) {
  for (auto &op: block) {
    // Analyze constant fp operations
    ANALYSIS(op, mlir::arith::ConstantFloatOp, isFullyAbstract);
    ANALYSIS(op, mlir::arith::ConstantOp, isFullyAbstract);
    ANALYSIS(op, mlir::tosa::ConstOp, isFullyAbstract);

    for (const auto &result: op.getResults()) {
      varF32Count += isFullyAbstract ? 1 : analyzeVariable<mlir::Float32Type>(result);
      varF64Count += isFullyAbstract ? 1 : analyzeVariable<mlir::Float64Type>(result);
    }
  }
}

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract) {
  argF32Count = 0, argF64Count = 0;
  varF32Count = 0, varF64Count = 0;
  constF32Set.clear();
  constF64Set.clear();

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments()){
    argF32Count += isFullyAbstract ? 1 : analyzeVariable<mlir::Float32Type>(arg);
    argF64Count += isFullyAbstract ? 1 : analyzeVariable<mlir::Float64Type>(arg);
  }
    

  // Step2. analyze the block
  auto &block = region.front();
  analyzeBlock(block, isFullyAbstract);

  return {
    .constF32Count = static_cast<int>(constF32Set.size()),
    .varF32Count = varF32Count,
    .argF32Count = argF32Count,
    .constF64Count = static_cast<int>(constF64Set.size()),
    .varF64Count = varF64Count,
    .argF64Count = argF64Count
  };
}
