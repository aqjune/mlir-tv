#include "abstractops.h"
#include "memory.h"
#include "smt.h"
#include "vcgen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include <string>

using namespace std;
using namespace mlir;

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
  llvm::cl::desc("first-mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"));

llvm::cl::opt<string> filename_tgt(llvm::cl::Positional,
  llvm::cl::desc("second-mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"));

llvm::cl::opt<unsigned> arg_smt_to("smt-to",
  llvm::cl::desc("Timeout for SMT queries (default=10000)"),
  llvm::cl::init(10000), llvm::cl::value_desc("ms"));

llvm::cl::opt<string> arg_dump_smt_to("dump-smt-to",
  llvm::cl::desc("Dump SMT queries to"), llvm::cl::value_desc("path"));

llvm::cl::opt<smt::SolverType> arg_solver("solver",
  llvm::cl::desc("Type of SMT solvers used when verifying"
                 " (default=Z3)"),
  llvm::cl::init(smt::SolverType::Z3),
  llvm::cl::values(
    clEnumValN(smt::SolverType::Z3, "Z3", "Z3 Solver"),
    clEnumValN(smt::SolverType::CVC5, "CVC5", "CVC5 Solver"),
    clEnumValN(smt::SolverType::ALL, "ALL", "Z3, CVC5 Solvers")
  )
);

llvm::cl::opt<int> fp_bits("fp-bits",
  llvm::cl::desc("The number of bits for the abstract representation of"
                 "float and double types (default=their bitwidths)"),
  llvm::cl::init(-1), llvm::cl::value_desc("number"));

llvm::cl::opt<unsigned int> num_memblocks("num-memory-blocks",
  llvm::cl::desc("Number of memory blocks required to validate translation"
                 " (default=8)"),
  llvm::cl::init(8), llvm::cl::value_desc("number"));

llvm::cl::opt<bool> arg_associative_sum("associative",
  llvm::cl::desc("Assume that floating point add is associative "
                 "(experimental)"),
  llvm::cl::init(false));

llvm::cl::opt<MemEncoding> memory_encoding("memory-encoding",
  llvm::cl::desc("Type of memref memory model (default=MULTIPLE)"),
  llvm::cl::init(MemEncoding::MULTIPLE_ARRAY), llvm::cl::Hidden,
  llvm::cl::values(
    clEnumValN(MemEncoding::SINGLE_ARRAY, "SINGLE", "Using single array memory encoding"),
    clEnumValN(MemEncoding::MULTIPLE_ARRAY, "MULTIPLE", "Using multiple arrays memory encoding")
  ));

llvm::cl::opt<bool> arg_multiset("multiset",
  llvm::cl::desc("Use multiset when encoding the associativity of the floating"
                 " point addition"),  llvm::cl::Hidden,
  llvm::cl::init(false));

// These functions are excerpted from ToolUtilities.cpp in mlir
static unsigned validateBuffer(unique_ptr<llvm::MemoryBuffer> srcBuffer,
    unique_ptr<llvm::MemoryBuffer> tgtBuffer,
    MLIRContext *context) {
  llvm::SourceMgr src_sourceMgr,  tgt_sourceMgr;
  src_sourceMgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());
  tgt_sourceMgr.AddNewSourceBuffer(move(tgtBuffer), llvm::SMLoc());

  auto ir_before = parseSourceFile(src_sourceMgr, context);
  if (!ir_before) {
    llvm::errs() << "Cannot parse source file\n";
    return 81;
  }

  auto ir_after = parseSourceFile(tgt_sourceMgr, context);
  if (!ir_after) {
    llvm::errs() << "Cannot parse target file\n";
    return 82;
  }

  int fp_bits_arg = fp_bits.getValue();
  pair<unsigned, unsigned> fp_bits;
  if (fp_bits_arg == -1)
    // TODO: Double is set to 63 instead of 64 (which is the correct bitwidth
    // of double) because compilers do not support int128 in general.
    fp_bits = {32, 63};
  else
    fp_bits = {fp_bits_arg, fp_bits_arg};

  return validate(ir_before, ir_after,
      arg_dump_smt_to.getValue(),
      num_memblocks.getValue(),
      memory_encoding.getValue(),
      fp_bits,
      arg_associative_sum.getValue(),
      arg_multiset.getValue()
    ).code;
}

int main(int argc, char* argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);

  smt::setTimeout(arg_smt_to.getValue());
  if (arg_solver.getValue() == smt::ALL || arg_solver.getValue() == smt::Z3)
    smt::useZ3();
  if (arg_solver.getValue() == smt::ALL || arg_solver.getValue() == smt::CVC5) {
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
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<arith::ArithmeticDialect>();
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
