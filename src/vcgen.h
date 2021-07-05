#pragma once

#include "results.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>

using namespace std;

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const string &dump_smt_to);
