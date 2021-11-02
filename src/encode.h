#pragma once

#include "state.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include <optional>
#include <string>

// encode can throw UnsupportedException.
void encode(State &st, mlir::FuncOp &fn, bool printOps);
