#include "smt.h"
#include "vcgen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// These functions are excerpted from ToolUtilities.cpp in mlir
static unsigned verifyBuffer(unique_ptr<llvm::MemoryBuffer> srcBuffer,
    unique_ptr<llvm::MemoryBuffer> tgtBuffer,
    MLIRContext *context) {
  llvm::SourceMgr src_sourceMgr,  tgt_sourceMgr;
  src_sourceMgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());
  tgt_sourceMgr.AddNewSourceBuffer(move(tgtBuffer), llvm::SMLoc());

  auto ir_before = parseSourceFile(src_sourceMgr, context);
  if (!ir_before) {
    llvm::errs() << "Cannot read source file\n";
    return 65;
  }

  auto ir_after = parseSourceFile(tgt_sourceMgr, context);
  if (!ir_after) {
    llvm::errs() << "Cannot read target file\n";
    return 66;
  }

  return verify(ir_before, ir_after, arg_dump_smt_to.getValue()).code;
}

static unsigned splitAndVerifyBuffer(unique_ptr<llvm::MemoryBuffer> srcBuffer,
    unique_ptr<llvm::MemoryBuffer> tgtBuffer,
    MLIRContext *context) {
  const char splitMarker[] = "// -----";

  SmallVector<llvm::StringRef, 8> sourceBuffers, targetBuffers;
  auto *srcMemBuffer = srcBuffer.get();
  auto *tgtMemBuffer = tgtBuffer.get();
  srcMemBuffer->getBuffer().split(sourceBuffers, splitMarker);
  tgtMemBuffer->getBuffer().split(targetBuffers, splitMarker);

  if (sourceBuffers.size() != targetBuffers.size()) {
    return 64;
  }

  unsigned retcode = 0;
  for (int i = 0; i < sourceBuffers.size(); i ++) {
    auto sourceSubMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(sourceBuffers[i]);
    auto targetSubMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(targetBuffers[i]);

    retcode = max(retcode,
        verifyBuffer(move(sourceSubMemBuffer), move(targetSubMemBuffer),
                     context));
  }

  // If any fails, then return a failure of the tool.
  return retcode;
}

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
  registry.insert<tensor::TensorDialect>();
  registry.insert<memref::MemRefDialect>();
  context.appendDialectRegistry(registry);

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);
  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 67;
  }

  auto tgt_file = openInputFile(filename_tgt, &errorMessage);
  if (!tgt_file) {
    llvm::errs() << errorMessage << "\n";
    return 68;
  }

  unsigned verificationResult;
  if (split_input_file) {
    verificationResult = splitAndVerifyBuffer(move(src_file), move(tgt_file), &context);
  } else {
    verificationResult = verifyBuffer(move(src_file), move(tgt_file), &context);
  }

  return verificationResult;
}
