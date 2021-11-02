#include "analysis.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <set>

using namespace std;

static set<llvm::APFloat> fpconstSet;
static int fpCount = 0;
static int fpArg = 0;

static void analysisAttr(const mlir::Attribute &a) {
  auto ty = a.getType();
  if (ty.isa<mlir::FloatType>()) 
    fpconstSet.insert(a.dyn_cast<mlir::FloatAttr>().getValue());
}

static void analysisElemAttr(const mlir::ElementsAttr &attr) {
  if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    if (denseAttr.isSplat()) {
      analysisAttr(denseAttr.getSplatValue());
    } else {
      for (unsigned i = 0; i < denseAttr.getNumElements(); i++) {
        analysisAttr(denseAttr.getFlatValue<mlir::Attribute>(i));
      }
    }
  } else if (auto sparseAttr = attr.dyn_cast<mlir::SparseElementsAttr>()) {
    auto denseAttr = sparseAttr.getValues();
    for (unsigned i = 0; i < denseAttr.getNumElements(); i++) {
      analysisAttr(denseAttr.getFlatValue<mlir::Attribute>(i));
    }
  }
}

static int analysisVariableCount(const mlir::Value &value) {
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
static void analysisOp(T op, bool isFullyAbstract);

template<>
void analysisOp(mlir::arith::ConstantFloatOp op, bool isFullyAbstract) {
  fpconstSet.insert(op.value());
}

template<>
void analysisOp(mlir::arith::ConstantOp op, bool isFullyAbstract) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analysisElemAttr(eattr);
}

template<>
void analysisOp(mlir::tosa::ConstOp op, bool isFullyAbstract) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analysisElemAttr(eattr);
}

#define ANALYSIS(op, ty, isFullyAbstract) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analysisOp(op2, isFullyAbstract); \
    continue; \
  }

void analysisBlock(mlir::Block &block, bool isFullyAbstract) {
  for (auto &op: block) {
    // Analysis constant fp operations
    ANALYSIS(op, mlir::arith::ConstantFloatOp, isFullyAbstract);
    ANALYSIS(op, mlir::arith::ConstantOp, isFullyAbstract);
    ANALYSIS(op, mlir::tosa::ConstOp, isFullyAbstract);

    for (const auto &result: op.getResults())
      fpCount += isFullyAbstract ? 1 : analysisVariableCount(result);
  }
}

AnalysisResult analysis(mlir::FuncOp &fn, bool isFullyAbstract) {
  fpArg = 0;
  fpCount = 0;
  fpconstSet.clear();

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analysis arguments
  for (const auto& arg: fn.getArguments())
    fpArg += isFullyAbstract ? 1 : analysisVariableCount(arg);

  // Step2. analysis block
  auto &block = region.front();
  analysisBlock(block, isFullyAbstract);

  return {
    .argFpCount = fpArg,
    .varFpCount = fpCount,
    .constFpCount= static_cast<int>(fpconstSet.size())
  };
}
