#include "mlir/Parser.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include <string>

bool compare_opt(mlir::OwningModuleRef &ir_before, mlir::OwningModuleRef &ir_after) {
    llvm::outs() << "--- IR before opt ---\n";
    for (auto &func: *ir_before) {
        func.walk([&](mlir::Operation *op) {
            auto nwhc_filter_hwcf_op = mlir::dyn_cast<mlir::linalg::ConvInputNHWCFilterHWCFOp>(op);
            if (nwhc_filter_hwcf_op) {
                llvm::outs() << nwhc_filter_hwcf_op.getOperationName() << "\n";
                for (auto operand : nwhc_filter_hwcf_op->getOperands()) {
                    llvm::outs() << operand;
                }
                llvm::outs() << "\n";
            }
        });
    }

    llvm::outs() << "--- IR after opt ---\n";
    for (auto &func: *ir_after) {
        func.walk([&](mlir::Operation *op) {
            if (mlir::dyn_cast<mlir::linalg::InitTensorOp>(op) || mlir::dyn_cast<mlir::linalg::GenericOp>(op) || mlir::dyn_cast<mlir::linalg::TensorReshapeOp>(op) || mlir::dyn_cast<mlir::linalg::MatmulOp>(op) || mlir::dyn_cast<mlir::linalg::YieldOp>(op)) {
                llvm::outs() << op->getName() << "\n";
                for (const auto &operand : op->getOperands()) {
                    llvm::outs() << operand.getType();
                    // if (mlir::isa<mlir::TensorType>(&operand)) {

                    // }
                    // if (mlir::isa<mlir::TensorType>(operand)) {
                    //     llvm::outs() << operand.getType();
                    // } else {
                    //     llvm::outs() << "integral type ignored";
                    // }
                    
                }
                llvm::outs() << "\n";
            }
        });
    }
    return true;
}

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

    // llvm::outs() << "MLIR-before\n\n";
    // ir_before->print(llvm::outs());
    // llvm::outs() << "\nMLIR-after\n\n";
    // ir_after->print(llvm::outs());

    compare_opt(ir_before, ir_after);

    // if (compare_opt(ir_before, ir_after)) {
    //     llvm::outs() << "valid optimization\n";
    // } else {
    //     llvm::outs() << "INVALID optimization\n";
    // }

    return 0;
}