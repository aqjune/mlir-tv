#pragma once

#include "tensor.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <variant>
#include <vector>

using ValueTy = std::variant<Tensor, Index>;

class ValuePrinter {
  llvm::raw_ostream *os;
public:
  ValuePrinter(llvm::raw_ostream *os): os(os) {}
  void operator() (const Tensor &t) { *os << t; }
  void operator() (const Index &i) { *os << i; }
};


struct RegFile {
private:
  std::vector<std::pair<mlir::Value, ValueTy>> m;

public:
  void add(mlir::Value v, ValueTy &&t);
  template<class T> T get(mlir::Value v) const {
    for (auto &[a, b]: m)
      if (a == v)
        return std::get<T>(b);

    llvm::errs() << "Cannot find key: " << v << "\n";
    assert(false && "Unknown key");
  }
  bool contains(mlir::Value v) const;

  auto begin() const { return m.begin(); }
  auto end() const { return m.end(); }
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
