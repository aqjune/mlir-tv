#pragma once

#include "tensor.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <variant>
#include <unordered_map>

using ValueTy = std::variant<Tensor, Index, Float, Integer>;

struct RegFile {
private:
  struct HashV {
    std::size_t operator()(mlir::Value v) const {
      return std::hash<void*>()(v.getAsOpaquePointer());
    }
  };

  std::unordered_map<mlir::Value, ValueTy, HashV> m;
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
  Tensor retValue;
  // TODO: add memory

  // Returns (this(tgt) state refines src,
  //          variables used for encoding refinement)
  std::pair<z3::expr, std::vector<z3::expr>> refines(const State &src);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
