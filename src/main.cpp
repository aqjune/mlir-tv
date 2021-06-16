#include "smt.h"
#include "vcgen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include <string>

using namespace std;
using namespace mlir;

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
  llvm::cl::desc("first-mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"));

llvm::cl::opt<string> filename_tgt(llvm::cl::Positional,
  llvm::cl::desc("second-mlir-file"),
  llvm::cl::Optional, llvm::cl::value_desc("filename"));

llvm::cl::opt<unsigned> arg_smt_to("smt-to",
  llvm::cl::desc("Timeout for SMT queries (default=10000)"),
  llvm::cl::init(10000), llvm::cl::value_desc("ms"));

llvm::cl::opt<string> arg_dump_smt_to("dump-smt-to",
  llvm::cl::desc("Dump SMT queries to"), llvm::cl::value_desc("path"));

int main(int argc, char* argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);

  z3::set_param("timeout", (int)arg_smt_to.getValue());

  MLIRContext context;
  DialectRegistry registry;
  // NOTE: we cannot use mlir::registerAllDialects because IREE does not have
  // dependency on some of those dialects
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<memref::MemRefDialect>();
  context.appendDialectRegistry(registry);

  auto ir_before = parseSourceFile(filename_src, &context);
  if (!ir_before) {
    llvm::errs() << "Cannot read " << filename_src << "\n";
    return 1;
  }

  auto ir_after = parseSourceFile(filename_tgt, &context);
  if (!ir_after) {
    llvm::errs() << "Cannot read " << filename_tgt << "\n";
    return 1;
  }

  verify(ir_before, ir_after, arg_dump_smt_to.getValue());

  return 0;
}
