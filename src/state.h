#pragma once

#include "tensor.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <vector>

struct RegFile {
  std::vector<std::pair<mlir::Value, Tensor>> m;

  void add(mlir::Value v, Tensor &&t) {
    for (auto &itm: m) {
      if (itm.first == v) {
        itm.second = std::move(t);
        return;
      }
    }
    m.emplace_back(v, std::move(t));
  }

  Tensor &get(mlir::Value v) {
    for (auto &[a, b]: m)
      if (a == v)
        return b;

    llvm::errs() << "Cannot find key: " << v << "\n";
    assert(false && "Unknown key");
  }

  bool contains(mlir::Value v) {
    for (auto &[a, b]: m) {
      if (a == v)
        return true;
    }
    return false;
  }
};

struct State {
  RegFile regs;
  Tensor retValue;
  // TODO: add memory

  // Returns (this(tgt) state refines src,
  //          variables used for encoding refinement)
  std::pair<z3::expr, std::vector<z3::expr>> refines(const State &src);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
