#include "smt.h"
#include "vcgen.h"
#include "utilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  llvm::cl::Optional, llvm::cl::value_desc("filename"));

llvm::cl::opt<unsigned> arg_smt_to("smt-to",
  llvm::cl::desc("Timeout for SMT queries (default=10000)"),
  llvm::cl::init(10000), llvm::cl::value_desc("ms"));

llvm::cl::opt<string> arg_dump_smt_to("dump-smt-to",
  llvm::cl::desc("Dump SMT queries to"), llvm::cl::value_desc("path"));

llvm::cl::opt<bool> split_input_file("split-input-file",
  llvm::cl::desc("Split the input file into pieces and process each chunk independently"),
  llvm::cl::init(false));

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

  std::string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);
  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto tgt_file = openInputFile(filename_tgt, &errorMessage);
  if (!tgt_file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> srcBuffer, std::unique_ptr<llvm::MemoryBuffer> tgtBuffer) {
    llvm::SourceMgr src_sourceMgr,  tgt_sourceMgr;
    src_sourceMgr.AddNewSourceBuffer(std::move(srcBuffer), llvm::SMLoc());
    tgt_sourceMgr.AddNewSourceBuffer(std::move(tgtBuffer), llvm::SMLoc());

    auto ir_before = parseSourceFile(src_sourceMgr, &context);
    if (!ir_before) {
      llvm::errs() << "Cannot read " << filename_src << "\n";
      return 1;
    }

    auto ir_after = parseSourceFile(tgt_sourceMgr, &context);
    if (!ir_after) {
      llvm::errs() << "Cannot read " << filename_tgt << "\n";
      return 1;
    }

    return verify(ir_before, ir_after, arg_dump_smt_to.getValue());
  };

  if (split_input_file) {
    return splitAndProcessBuffer(std::move(src_file), std::move(tgt_file), processBuffer);
  } else {
    return processBuffer(std::move(src_file), std::move(tgt_file));
  }
}
