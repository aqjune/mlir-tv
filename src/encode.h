#pragma once

#include "state.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <optional>
#include <string>

// encode can throw UnsupportedException.
void encode(State &st, mlir::FuncOp &fn, bool printOps);
