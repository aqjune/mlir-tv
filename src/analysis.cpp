#include "analysis.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <set>

using namespace std;

static set<llvm::APFloat> constF32Set;
static set<llvm::APFloat> constF64Set;
static int fpCount = 0;
static int fpArg = 0;

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
      analyzeAttr(denseAttr.getSplatValue());
    } else {
      for (unsigned i = 0; i < denseAttr.getNumElements(); i++) {
        analyzeAttr(denseAttr.getFlatValue<mlir::Attribute>(i));
      }
    }
  } else if (auto sparseAttr = attr.dyn_cast<mlir::SparseElementsAttr>()) {
    auto denseAttr = sparseAttr.getValues();
    for (unsigned i = 0; i < denseAttr.getNumElements(); i++) {
      analyzeAttr(denseAttr.getFlatValue<mlir::Attribute>(i));
    }
  }
}

static int analyzeVariable(const mlir::Value &value) {
  auto ty = value.getType();
  if (ty.isa<mlir::FloatType>()) {
    return 1;

  } else if (ty.isa<mlir::TensorType>()) {
    auto tensorty = ty.cast<mlir::TensorType>();
    if (!tensorty.getElementType().isa<mlir::FloatType>())
      return 0;

    if (tensorty.hasStaticShape()) 
      return tensorty.getNumElements();
    else 
      return Tensor::MAX_TENSOR_SIZE;

  } else if (ty.isa<mlir::MemRefType>()) {
    auto memrefty = ty.cast<mlir::MemRefType>();
    if (!memrefty.getElementType().isa<mlir::FloatType>())
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

    for (const auto &result: op.getResults())
      fpCount += isFullyAbstract ? 1 : analyzeVariable(result);
  }
}

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract) {
  fpArg = 0;
  fpCount = 0;
  constF32Set.clear();
  constF64Set.clear();

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments())
    fpArg += isFullyAbstract ? 1 : analyzeVariable(arg);

  // Step2. analyze the block
  auto &block = region.front();
  analyzeBlock(block, isFullyAbstract);

  return {
    .argFpCount = fpArg,
    .varFpCount = fpCount,
    .constF32Count = static_cast<int>(constF32Set.size()),
    .constF64Count = static_cast<int>(constF64Set.size())
  };
}
