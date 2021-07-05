#pragma once

#include <string>
#include "mlir/IR/BuiltinOps.h"

int verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const std::string &dump_smt_to);
