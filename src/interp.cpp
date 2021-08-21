#include "encode.h"
#include "memory.h"
#include "print.h"
#include "smt.h"
#include "vcgen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include <string>

using namespace smt;
using namespace std;
using namespace mlir;

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
  llvm::cl::desc("mlir-file"),
  llvm::cl::Required, llvm::cl::value_desc("filename"));

llvm::cl::opt<unsigned int> num_memblocks("num-memory-blocks",
  llvm::cl::desc("Number of memory blocks required to validate translation"
                 " (default=8)"),
  llvm::cl::init(8), llvm::cl::value_desc("number"));

llvm::cl::opt<MemEncoding> memory_encoding("memory-encoding",
  llvm::cl::desc("Type of memref memory model (default=MULTIPLE)"),
  llvm::cl::init(MemEncoding::MULTIPLE_ARRAY), llvm::cl::Hidden,
  llvm::cl::values(
    clEnumValN(MemEncoding::SINGLE_ARRAY, "SINGLE", "Using single array memory encoding"),
    clEnumValN(MemEncoding::MULTIPLE_ARRAY, "MULTIPLE", "Using multiple arrays memory encoding")
  ));


static void runFunction(mlir::FuncOp fn) {
  if (fn.getNumArguments() != 0) {
    llvm::outs() << fn.getName()
                 << ": a function with arguments is unsupported.\n\n";
    return;
  }

  llvm::outs() << "Function " << fn.getName() << "\n\n";

  // FIXME: max. # local blocks does not need to be num_memblocks
  State s(unique_ptr<Memory>{
      Memory::create(num_memblocks, num_memblocks, memory_encoding)});
  encode(s, fn, false);
  printOperations(smt::Model::empty(), fn, s);
}

static unsigned runBuffer(unique_ptr<llvm::MemoryBuffer> srcBuffer,
    MLIRContext *context) {

  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());
  auto the_module = parseSourceFile(mgr, context);

  if (!the_module) {
    llvm::errs() << "Cannot parse source file\n";
    return 81; // Sync this with mlir-tv's main.cpp
  }

  llvm::for_each(*the_module, [&](auto &op) {
    auto fnop = mlir::dyn_cast<mlir::FuncOp>(op);
    if (fnop.isDeclaration())
      return;

    runFunction(fnop);
  });

  return 0;
}

int main(int argc, char* argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);

  MLIRContext context;
  DialectRegistry registry;
  // NOTE: we cannot use mlir::registerAllDialects because IREE does not have
  // dependency on some of those dialects
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<tensor::TensorDialect>();
  context.appendDialectRegistry(registry);

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);
  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 66; // Sync this with mlir-tv's main.cpp
  }

  return runBuffer(move(src_file), &context);
}
