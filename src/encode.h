#pragma once

#include "state.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <optional>
#include <string>

// encode can throw UnsupportedException.
void encode(State &st, mlir::func::FuncOp &fn, bool printOps);
