#pragma once

#include <string>
#include "mlir/IR/BuiltinOps.h"

using namespace std;

int verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const string &dump_smt_to);
