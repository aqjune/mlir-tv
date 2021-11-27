#include "analysis.h"
#include "debug.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <type_traits>

using namespace std;

static void analyzeAPFloat(
    const mlir::Type ty, const llvm::APFloat val, AnalysisResult &res) {
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
    res.F32.constSet.insert(val_f32);
  }
  res.F64.constSet.insert(val_f64);
}

static void analyzeAttr(const mlir::Attribute &a, AnalysisResult &res) {
  assert(!a.isa<mlir::ElementsAttr>());

  auto ty = a.getType();
  if (!ty.isa<mlir::FloatType>())
    return;

  const auto val = a.dyn_cast<mlir::FloatAttr>().getValue();
  analyzeAPFloat(ty, val, res);
}

static void analyzeElemAttr(
    const mlir::ElementsAttr &attr, AnalysisResult &res) {
  if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    if (denseAttr.isSplat()) {
      analyzeAttr(denseAttr.getSplatValue<mlir::Attribute>(), res);
    } else {
      for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
        analyzeAttr(attr, res);
      }
    }
  } else if (auto sparseAttr = attr.dyn_cast<mlir::SparseElementsAttr>()) {
    auto denseAttr = sparseAttr.getValues();
    for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
      analyzeAttr(attr, res);
    }
  }
}

static void analyzeVariable(
    const mlir::Value &var, AnalysisResult &res, bool numElemsIgnored,
    bool isArg = false) {
  auto ty = var.getType();
  size_t &f32Count = isArg ? res.F32.argCount : res.F32.varCount;
  size_t &f64Count = isArg ? res.F64.argCount : res.F64.varCount;
  decltype(res.memref.argCount) &memrefCnt =
      isArg ? res.memref.argCount : res.memref.varCount;

  if (ty.isF32()) {
    f32Count++;

  } else if (ty.isF64()) {
    f64Count++;

  } else if (ty.isa<mlir::TensorType>() || ty.isa<mlir::MemRefType>()) {
    auto tensorty = ty.cast<mlir::ShapedType>();
    auto elemty = tensorty.getElementType();
    int64_t cnt;

    if (tensorty.hasStaticShape()) 
      cnt = tensorty.getNumElements();
    else 
      cnt = Tensor::MAX_TENSOR_SIZE;

    if (numElemsIgnored)
      cnt = cnt ? 1 : 0;

    if (elemty.isF32())
      f32Count += cnt;
    else if (elemty.isF64())
      f64Count += cnt;

    if (ty.isa<mlir::MemRefType>()) {
      memrefCnt[elemty]++;
    }
  }
}

template<class T>
static void analyzeOp(T op, AnalysisResult &res);

static void analyzeBlock(
    mlir::Block &block, AnalysisResult &res, bool isFullyAbstract);

template<>
void analyzeOp(mlir::arith::ConstantFloatOp op, AnalysisResult &res) {
  auto ty = op.getType();
  const auto val = op.value();
  analyzeAPFloat(ty, val, res);
}

template<>
void analyzeOp(mlir::arith::ConstantOp op, AnalysisResult &res) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analyzeElemAttr(eattr, res);
}

template<>
void analyzeOp(mlir::tosa::ConstOp op, AnalysisResult &res) {
  auto tensorty = op.getType().dyn_cast<mlir::RankedTensorType>();
  auto eattr = op.value().dyn_cast<mlir::ElementsAttr>();
  if (!tensorty || !eattr) return;

  analyzeElemAttr(eattr, res);
}

void analyzeRegion(
    mlir::Region &region, AnalysisResult &res, bool isFullyAbstract) {
  if (!region.hasOneBlock())
    throw UnsupportedException("Region with a single block is supported only");

  auto &block = region.front();
  return analyzeBlock(block, res, isFullyAbstract);
}

#define ANALYZE(op, ty, res) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analyzeOp(op2, res); \
    continue; \
  }

#define ANALYZE_REGION(op, ty, region_fn, res, numElemsIgnored) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    analyzeRegion(op2.region_fn(), res, numElemsIgnored); \
    continue; \
  }

static void analyzeBlock(
    mlir::Block &block, AnalysisResult &res, bool numElemsIgnored) {
  for (auto &op: block) {
    // Analyze constant operations
    // These operations do not increase varCount
    ANALYZE(op, mlir::arith::ConstantFloatOp, res);
    ANALYZE(op, mlir::arith::ConstantOp, res);
    ANALYZE(op, mlir::tosa::ConstOp, res);

    // Non-constant operations; increase varCount if return type matches
    for (const auto &result: op.getResults()) {
      analyzeVariable(result, res, numElemsIgnored);
    }

    // Analyze operations having subregions.
    ANALYZE_REGION(op, mlir::linalg::GenericOp, region, res, numElemsIgnored);
    ANALYZE_REGION(op, mlir::linalg::PadTensorOp, region, res, numElemsIgnored);
    ANALYZE_REGION(op, mlir::tensor::GenerateOp, body, res, numElemsIgnored);
  }
}

AnalysisResult analyze(mlir::FuncOp &fn, bool isFullyAbstract) {
  AnalysisResult res;

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments()){
    analyzeVariable(arg, res, isFullyAbstract, /*isArg*/true);
  }

  // Step2. analyze the block
  auto &block = region.front();
  analyzeBlock(block, res, isFullyAbstract);

  verbose("analysis") << "<" << fn.getName().str() << ">\n";
  verbose("analysis") << "  f32 arg count: " << res.F32.argCount << "\n";
  verbose("analysis") << "  f32 var count: " << res.F32.varCount << "\n";
  verbose("analysis") << "  f64 arg count: " << res.F64.argCount << "\n";
  verbose("analysis") << "  f64 var count: " << res.F64.varCount << "\n";
  for (auto &[ty, cnt]: res.memref.argCount)
    verbose("analysis") << "  memref arg count (" << ty << "): " << cnt << "\n";
  for (auto &[ty, cnt]: res.memref.varCount)
    verbose("analysis") << "  memref var count (" << ty << "): " << cnt << "\n";

  return res;
}
