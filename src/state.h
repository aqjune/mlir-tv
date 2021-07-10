#pragma once

#include "tensor.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <stack>
#include <variant>
#include "mlir/Support/LLVM.h"

using ValueTy = std::variant<Tensor, Index, Float, Integer>;

struct RegFile {
private:
  llvm::DenseMap<mlir::Value, ValueTy> m;
  ValueTy findOrCrash(mlir::Value v) const;

public:
  void add(mlir::Value v, ValueTy &&t);
  template<class T> T get(mlir::Value v) const {
    return std::get<T>(findOrCrash(v));
  }
  bool contains(mlir::Value v) const;
  z3::expr getZ3Expr(mlir::Value v) const;

  auto begin() const { return m.begin(); }
  auto end() const { return m.end(); }
};

struct State {
  RegFile regs;
  std::stack<std::vector<z3::expr>> linalgGenericScopes;
  Tensor retValue;
  // TODO: add memory

  // Returns (this(tgt) state refines src,
  //          variables used for encoding refinement)
  std::pair<z3::expr, std::vector<z3::expr>> refines(const State &src);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
