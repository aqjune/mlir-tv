#include "abstractops.h"
#include "debug.h"
#include "memory.h"
#include "opts.h"
#include "smt.h"
#include "vcgen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include <string>

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
  llvm::cl::desc("first-mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<string> filename_tgt(llvm::cl::Positional,
  llvm::cl::desc("second-mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned> arg_smt_to("smt-to",
  llvm::cl::desc("Timeout for SMT queries (default=30000)"),
  llvm::cl::init(30000), llvm::cl::value_desc("ms"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<smt::SolverType> arg_solver("solver",
  llvm::cl::desc("Type of SMT solvers used when verifying"
                 " (default=Z3)"),
  llvm::cl::init(smt::SolverType::Z3),
  llvm::cl::values(
    clEnumValN(smt::SolverType::Z3, "Z3", "Z3 Solver"),
    clEnumValN(smt::SolverType::CVC5, "CVC5", "CVC5 Solver")
  ),
  llvm::cl::cat(MlirTvCategory)
);

llvm::cl::opt<bool> arg_verbose("verbose",
  llvm::cl::desc("Be verbose about what's going on"), llvm::cl::Hidden,
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));


// These functions are excerpted from ToolUtilities.cpp in mlir
static unsigned validateBuffer(unique_ptr<llvm::MemoryBuffer> srcBuffer,
    unique_ptr<llvm::MemoryBuffer> tgtBuffer,
    MLIRContext *context) {
  llvm::SourceMgr src_sourceMgr,  tgt_sourceMgr;
  src_sourceMgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());
  tgt_sourceMgr.AddNewSourceBuffer(move(tgtBuffer), llvm::SMLoc());

  auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, context);
  if (!ir_before) {
    llvm::errs() << "Cannot parse source file\n";
    return 81;
  }

  auto ir_after = parseSourceFile<ModuleOp>(tgt_sourceMgr, context);
  if (!ir_after) {
    llvm::errs() << "Cannot parse target file\n";
    return 82;
  }

  return validate(ir_before, ir_after).code;
}

int main(int argc, char* argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);
  setVerbose(arg_verbose.getValue());

  smt::setTimeout(arg_smt_to.getValue());
  if (arg_solver.getValue() == smt::Z3)
    smt::useZ3();
  if (arg_solver.getValue() == smt::CVC5) {
#ifdef SOLVER_CVC5
    smt::useCVC5();
#else
    if (arg_solver.getValue() == smt::CVC5) {
      llvm::errs() << "CVC5_DIR was not set while building this project! aborting..\n";
      return 1;
    }
#endif
  }

  MLIRContext context;
  DialectRegistry registry;
  // NOTE: we cannot use mlir::registerAllDialects because IREE does not have
  // dependency on some of those dialects
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<sparse_tensor::SparseTensorDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<tosa::TosaDialect>();
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);
  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 66;
  }

  auto tgt_file = openInputFile(filename_tgt, &errorMessage);
  if (!tgt_file) {
    llvm::errs() << errorMessage << "\n";
    return 66;
  }

  unsigned verificationResult = validateBuffer(
      move(src_file), move(tgt_file), &context);

  return verificationResult;
}
