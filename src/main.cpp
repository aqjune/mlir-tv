#include "vcgen.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include <string>

using namespace std;
using namespace mlir;


int main(int argc, char* argv[]) {
  if (argc != 3) {
    llvm::errs() << "iree-tv\n";
    llvm::errs() << "USAGE: iree-tv <.mlir before opt> <.mlir after opt>\n";
    return 1;
  }

  string filename_src = argv[1];
  string filename_tgt = argv[2];

  MLIRContext context;
  DialectRegistry registry;
  // NOTE: we cannot use mlir::registerAllDialects because IREE does not have
  // dependency on some of those dialects
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

  verify(ir_before, ir_after);

  return 0;
}
