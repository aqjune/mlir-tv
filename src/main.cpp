#include "mlir/Parser.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        llvm::errs() << "iree-tv\n";
        llvm::errs() << "USAGE: iree-tv <.mlir before opt> <.mlir after opt>\n";
        return 1;
    }

    std::string filename_before = argv[1];
    std::string filename_after = argv[2];

    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<mlir::linalg::LinalgDialect>();
    context.appendDialectRegistry(registry);

    auto ir_before = mlir::parseSourceFile(filename_before, &context);
    if (!ir_before) {
        return 1;
    }

    auto ir_after = mlir::parseSourceFile(filename_after, &context);
    if (!ir_after) {
        return 1;
    }

    llvm::outs() << "MLIR-before\n\n";
    ir_before->print(llvm::outs());
    llvm::outs() << "\nMLIR-after\n\n";
    ir_after->print(llvm::outs());

    return 0;
}