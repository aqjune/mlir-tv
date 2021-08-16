#pragma once

#include "state.h"

#include <optional>
#include <string>

std::optional<std::string> encode(State &st, mlir::FuncOp &fn, bool printOps);
