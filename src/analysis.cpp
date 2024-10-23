#include "analysis.h"
#include "debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "value.h"
#include "utils.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <type_traits>

#define ANALYZE(op, ty, res) \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    if (analyzeOp(op2, res)) \
      continue; \
  }

using namespace std;

namespace {

void analyzeBlock(mlir::Block &block, AnalysisResult &res);

// Return false if op is not processed
template<class T> bool analyzeOp(T op, AnalysisResult &res);

void analyzeAPFloat(
    const mlir::Type ty, const llvm::APFloat val, AnalysisResult &res) {
  if (val.isNaN() || val.isInfinity()) {
    // They cannot be inserted into set<APFloat>.
    // They will be specially treated in setAbstraction() (abstractops.cpp)
    if (ty.isF32())
      res.F32.hasInfOrNaN = true;
    else if (ty.isF64())
      res.F64.hasInfOrNaN = true;
    else
      throw UnsupportedException(ty, "Unsupported type");
    return;
  }

  auto val_f32_round = val, val_f32_floor = val, val_f32_ceil = val;
  auto val_f64 = val, val_f64_floor = val, val_f64_ceil = val;
  bool lost_info; // dummy

  llvm::APFloat::opStatus op_status;
  if (ty.isF32()) {
    op_status = val_f64.convert(llvm::APFloat::IEEEdouble(),
                    // doesn't matter in extension
                    llvm::APFloat::rmTowardZero, &lost_info);
  } else if (ty.isF64()) {
    op_status = val_f32_round.convert(llvm::APFloat::IEEEsingle(),
                    // round for correct analysis
                    llvm::APFloat::rmNearestTiesToEven, &lost_info);
    val_f32_floor.convert(llvm::APFloat::IEEEsingle(),
                          // floor for correct BV mapping (ordering issue)
                          llvm::APFloat::rmTowardZero, &lost_info);
    // ceiled value should also be added
    // as unknown variable(s) may have to be rounded upward
    if (val.isNegative()) {
      val_f32_ceil.convert(llvm::APFloat::IEEEsingle(),
                          llvm::APFloat::rmTowardNegative, &lost_info);
    } else {
      val_f32_ceil.convert(llvm::APFloat::IEEEsingle(),
                          llvm::APFloat::rmTowardPositive, &lost_info);
    }
  } else {
    throw UnsupportedException(ty, "Unsupported type");
  }

  // clear signs, as constSet only includes positive FPs
  if (val_f64.isNegative()) {
    val_f32_floor.clearSign();
    val_f32_ceil.clearSign();
    val_f64.clearSign();
  }

  val_f64_ceil = val_f32_ceil;
  val_f64_ceil.convert(llvm::APFloat::IEEEdouble(),
                        // doesn't really matter in extension
                        llvm::APFloat::rmTowardZero, &lost_info);
  val_f64_floor = val_f32_floor;
  val_f64_floor.convert(llvm::APFloat::IEEEdouble(),
                        // doesn't really matter in extension
                        llvm::APFloat::rmTowardZero, &lost_info);  

  // Values beyond the float range are mapped to Inf
  if (!(op_status & llvm::APFloat::opOverflow)) {
    res.F32.constSet.insert(val_f32_ceil);
    res.F32.constSet.insert(val_f32_floor);
    // Ceiled and floored values should be added to F64 as well
    // to map values correctly between different precisions
    res.F64.constSet.insert(val_f64_ceil);
    res.F64.constSet.insert(val_f64_floor);
  } else {
    // f64 -> f32 downcasting yields inf
    res.F32.hasInfOrNaN = true;
  }
  res.F64.constSet.insert(val_f64);
}

void analyzeAttr(const mlir::Attribute &a, AnalysisResult &res) {
  assert(!a.isa<mlir::ElementsAttr>());

  auto fa = mlir::dyn_cast<mlir::FloatAttr>(a);
  if (!fa)
    return;

  const auto val = fa.getValue();
  analyzeAPFloat(fa.getType(), val, res);
}

bool analyzeElemAttr(
    const mlir::ElementsAttr &attr, AnalysisResult &res) {
  if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (denseAttr.isSplat()) {
      analyzeAttr(denseAttr.getSplatValue<mlir::Attribute>(), res);
    } else {
      if (Tensor::MAX_CONST_SIZE >= 0 &&
          denseAttr.getNumElements() > Tensor::MAX_CONST_SIZE)
        return false;

      for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
        analyzeAttr(attr, res);
      }
    }
  } else if (auto sparseAttr = mlir::dyn_cast<mlir::SparseElementsAttr>(attr)) {
    if (Tensor::MAX_CONST_SIZE >= 0 &&
        sparseAttr.getNumElements() > Tensor::MAX_CONST_SIZE)
      return false;

    auto denseAttr = sparseAttr.getValues();
    for (const auto& attr: denseAttr.getValues<mlir::Attribute>()) {
      analyzeAttr(attr, res);
    }
  }
  return true;
}

struct VarAnalysisConfig {
  bool isArg;
  bool isOperand;
  bool createsNewFpVal;

public:
  static VarAnalysisConfig arg() {
    return { .isArg = true, .isOperand = false, .createsNewFpVal = false};
  }
  static VarAnalysisConfig operand() {
    return { .isArg = false, .isOperand = true, .createsNewFpVal = false};
  }
  static VarAnalysisConfig op(bool createsNewFpVal) {
    return {
      .isArg = false, .isOperand = false, .createsNewFpVal = createsNewFpVal
    };
  }
};

void analyzeVariable(
    const mlir::Value &var, AnalysisResult &res, VarAnalysisConfig config) {
  auto ty = var.getType();
  size_t &f32Count = config.isArg ? res.F32.argCount : res.F32.varCount;
  size_t &f64Count = config.isArg ? res.F64.argCount : res.F64.varCount;
  size_t &f32ElemsCount = res.F32.elemsCount;
  size_t &f64ElemsCount = res.F64.elemsCount;
  decltype(res.memref.argCount) &memrefCnt =
      config.isArg ? res.memref.argCount : res.memref.varCount;
  bool doCount = config.isArg || config.isOperand || config.createsNewFpVal;

  if (ty.isF32()) {
    if (doCount)
      f32Count++;

    return;
  } else if (ty.isF64()) {
    if (doCount)
      f64Count++;

    return;
  } else if (!mlir::isa<mlir::TensorType>(ty) && !mlir::isa<mlir::MemRefType>(ty))
    return;

  auto tensorty = mlir::cast<mlir::ShapedType>(ty);
  auto elemty = tensorty.getElementType();

  if (doCount) {
    int64_t cnt;

    if (tensorty.hasStaticShape()) 
      cnt = tensorty.getNumElements();
    else 
      cnt = Tensor::MAX_TENSOR_SIZE;

    if (elemty.isF32()) {
      // Regardless of cnt, increment f32Count.
      // This is necessary to enable preparation of abstract fp encoding
      // even if tensor<0xf32> happens.
      f32Count ++;
      if (cnt > 0)
        f32ElemsCount += cnt - 1;
    } else if (elemty.isF64()) {
      f64Count ++;
      if (cnt > 0)
        f64ElemsCount += cnt - 1;
    } else if (mlir::isa<mlir::FloatType>(elemty)) {
      throw UnsupportedException(ty, "Unsupported type");
    }
  }

  if (mlir::isa<mlir::MemRefType>(ty) && !config.isOperand)
    memrefCnt[elemty]++;
}

void analyzeRegion(mlir::Region &region, AnalysisResult &res) {
  if (!region.hasOneBlock())
    throw UnsupportedException("Region with a single block is supported only");

  auto &block = region.front();
  return analyzeBlock(block, res);
}

template<>
bool analyzeOp(mlir::memref::GetGlobalOp op, AnalysisResult &res) {
  llvm::StringRef glbName = op.getName();
  auto mop = op.getOperation()->getParentOfType<mlir::ModuleOp>();
  auto glb = mlir::cast<mlir::memref::GlobalOp>(mop.lookupSymbol(glbName));
  res.memref.usedGlobals[glbName.str()] = glb;

  if (glb.getConstant() && glb.getInitialValue()) {
    analyzeElemAttr(mlir::cast<mlir::ElementsAttr>(glb.getInitialValue().value()), res);
  }
  return true;
}

template<>
bool analyzeOp(mlir::arith::ConstantFloatOp op, AnalysisResult &res) {
  auto ty = op.getType();
  const auto val = op.value();
  analyzeAPFloat(ty, val, res);
  return true;
}

template<>
bool analyzeOp(mlir::arith::ConstantOp op, AnalysisResult &res) {
  auto tensorty = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  auto eattr = mlir::dyn_cast<mlir::ElementsAttr>(op.getValue());
  if (!tensorty || !eattr) return true;

  bool processed = analyzeElemAttr(eattr, res);
  if (!processed) {
    auto &ros = verbose("analyzeOp(mlir::arith::ConstantOp)");
    ros << "skipped: ";
    op.print(ros, mlir::OpPrintingFlags().elideLargeElementsAttrs());
    ros << "\n";
  }
  return processed;
}

template<>
bool analyzeOp(mlir::tosa::ConstOp op, AnalysisResult &res) {
  auto tensorty = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  auto eattr = mlir::dyn_cast<mlir::ElementsAttr>(op.getValue());
  if (!tensorty || !eattr) return true;

  bool processed = analyzeElemAttr(eattr, res);
  if (!processed) {
    auto &ros = verbose("analyzeOp(mlir::arith::ConstantOp)");
    ros << "skipped: ";
    op.print(ros, mlir::OpPrintingFlags().elideLargeElementsAttrs());
    ros << "\n";
  }
  return processed;
}

template<>
bool analyzeOp(mlir::tosa::ClampOp op, AnalysisResult &res) {
  auto ty = mlir::Float32Type::get(op.getContext());
  analyzeAPFloat(ty, op.getMinFp(), res);
  analyzeAPFloat(ty, op.getMaxFp(), res);
  return true;
}

template<>
bool analyzeOp(mlir::linalg::GenericOp op, AnalysisResult &res) {
  // If generic loop has reduction loops, then result is not elementwise
  auto indexingMaps = op.getIndexingMaps().getValue();
  auto outputMap = mlir::cast<mlir::AffineMapAttr>(indexingMaps.back()).getValue();
  bool isReudctionLoop = !outputMap.isPermutation();
  if (isReudctionLoop)
    res.isElementwiseFPOps = false;

  analyzeRegion(op.getRegion(), res);
  return true;
}

template<>
bool analyzeOp(mlir::tensor::PadOp op, AnalysisResult &res) {
  analyzeRegion(op.getRegion(), res);
  return true;
}

template<>
bool analyzeOp(mlir::tensor::GenerateOp op, AnalysisResult &res) {
  analyzeRegion(op.getBody(), res);
  return true;
}

void analyzeBlock(
    mlir::Block &block, AnalysisResult &res) {
  for (auto &op: block) {
    // Analyze constant operations
    // These operations do not increase varCount
    // If it is a constant tensor that is too large (> Tensor::MAX_CONST_SIZE),
    // ANALYZE falls through and increases varCount.
    ANALYZE(op, mlir::arith::ConstantFloatOp, res);
    ANALYZE(op, mlir::arith::ConstantOp, res);
    ANALYZE(op, mlir::tosa::ConstOp, res);

    // Non-constant operations; increase varCount if return type matches
    // For constant globals: conservatively assume that they increase varCount
    for (const auto &result: op.getResults()) {
      bool canCreateNewFp =
          // Operations create new fp except these.
          !mlir::isa<mlir::tosa::ConcatOp>(op) &&
          !mlir::isa<mlir::tosa::GatherOp>(op) &&
          !mlir::isa<mlir::tosa::ReshapeOp>(op) &&
          !mlir::isa<mlir::tosa::ReverseOp>(op) &&
          !mlir::isa<mlir::tosa::TransposeOp>(op) &&
          !mlir::isa<mlir::tensor::CollapseShapeOp>(op) &&
          !mlir::isa<mlir::tensor::ExpandShapeOp>(op) &&
          !mlir::isa<mlir::tensor::ExtractOp>(op) &&
          !mlir::isa<mlir::tensor::ExtractSliceOp>(op) &&
          !mlir::isa<mlir::tensor::InsertOp>(op) &&
          !mlir::isa<mlir::tensor::InsertSliceOp>(op);
      analyzeVariable(result, res, VarAnalysisConfig::op(canCreateNewFp));
    }

    // GenericOp accepts memref argument as input & output
    // and newly created FPs can be stored to output memref.
    if (auto op2 = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
      if (op2.hasPureBufferSemantics()) {
        for (const auto &operand: op2.getOutputs()) {
          analyzeVariable(operand, res, VarAnalysisConfig::operand());
        }
      }
    }

    // Check whether op has reductions such as summation, etc
    if (mlir::isa<mlir::linalg::DotOp>(op) ||
        mlir::isa<mlir::linalg::MatmulOp>(op) ||
        mlir::isa<mlir::linalg::Conv2DNchwFchwOp>(op) ||
        mlir::isa<mlir::linalg::Conv2DNhwcHwcfOp>(op) ||
        mlir::isa<mlir::tosa::Conv2DOp>(op) ||
        mlir::isa<mlir::tosa::DepthwiseConv2DOp>(op) ||
        mlir::isa<mlir::tosa::FullyConnectedOp>(op) ||
        mlir::isa<mlir::tosa::ReduceSumOp>(op)) {
      res.isElementwiseFPOps = false;

      // Reduction op can create intermediate fp values.
      // We also count them in a conservative manner.
      for (const auto &operand: op.getOperands()) {
        analyzeVariable(operand, res, VarAnalysisConfig::operand());
      }
    }

    ANALYZE(op, mlir::tosa::ClampOp, res);

    // Detect global vars.
    // # fps & blocks are already increased by the loop above.
    ANALYZE(op, mlir::memref::GetGlobalOp, res);

    // Analyze operations having subregions.
    ANALYZE(op, mlir::linalg::GenericOp, res);
    ANALYZE(op, mlir::tensor::PadOp, res);
    ANALYZE(op, mlir::tensor::GenerateOp, res);
  }
}
}

AnalysisResult analyze(mlir::func::FuncOp &fn) {
  AnalysisResult res;

  auto &region = fn.getRegion();
  if (!llvm::hasSingleElement(region))
    throw UnsupportedException(
        region.getParentOp(), "Only a region with one block is supported");

  // Step1. analyze arguments
  for (const auto& arg: fn.getArguments()){
    analyzeVariable(arg, res, VarAnalysisConfig::arg());
  }

  // Step2. analyze the block
  auto &block = region.front();
  analyzeBlock(block, res);

  verbose("analysis") << "<" << fn.getName().str() << ">\n";
  verbose("analysis") << "  fn has only elementwise op?: "
      << (res.isElementwiseFPOps ? "YES\n" : "NO\n");
  verbose("analysis") << "  f32 arg count: " << res.F32.argCount << "\n";
  verbose("analysis") << "  f32 var count: " << res.F32.varCount << "\n";
  verbose("analysis") << "  f32 elements count: " << res.F32.elemsCount << "\n";
  verbose("analysis") << "  f32 consts count: " << res.F32.constSet.size()
      << "\n";
  verbose("analysis") << "  f64 arg count: " << res.F64.argCount << "\n";
  verbose("analysis") << "  f64 var count: " << res.F64.varCount << "\n";
  verbose("analysis") << "  f64 elements count: " << res.F64.elemsCount << "\n";
  verbose("analysis") << "  f64 consts count: " << res.F64.constSet.size()
      << "\n";
  for (auto &[ty, cnt]: res.memref.argCount)
    verbose("analysis") << "  memref arg count (" << ty << "): " << cnt << "\n";
  for (auto &[ty, cnt]: res.memref.varCount)
    verbose("analysis") << "  memref var count (" << ty << "): " << cnt << "\n";
  return res;
}
