#pragma once

#include "memory.h"
#include "smt.h"
#include "value.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <stack>
#include <optional>
#include <variant>
#include "mlir/Support/LLVM.h"

using ValueTy = std::variant<Tensor, MemRef, Index, Float, Integer>;

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const ValueTy &);
smt::Expr getExpr(const ValueTy &vty);
ValueTy eval(const ValueTy &vty, smt::Model m);

class ArgInfo {
private:
  llvm::DenseMap<int, ValueTy> m;
public:
  void add(int v, ValueTy &&t) { m.insert({v, std::move(t)}); }
  std::optional<ValueTy> get(int v) const {
    auto itr = m.find(v);
    if (itr != m.end())
      return {itr->second};
    return {};
  }
};

class RegFile {
private:
  llvm::DenseMap<mlir::Value, ValueTy> m;

public:
  void add(mlir::Value v, ValueTy &&t);

  // For non-aggregate types only
  void add(mlir::Value v, const smt::Expr &e, mlir::Type ty);

  ValueTy findOrCrash(mlir::Value v) const;
  template<class T> T get(mlir::Value v) const {
    return std::get<T>(findOrCrash(v));
  }
  bool contains(mlir::Value v) const;
  smt::Expr getExpr(mlir::Value v) const;

  auto begin() const { return m.begin(); }
  auto end() const { return m.end(); }
};

class State {
public:
  smt::Expr precond;
  // welldef[i]: is instruction i well-defined?
  // The negated form of UB is tracked because the neg. of value refinement is:
  // 'src.no-ub /\ tgt.no-ub /\ src.retvalue != tgt.retvalue'.
  // We'll need to implement our own version of peephole optimizations on Z3
  // expr some day (or simply use Alive2's one), and this form will be helpful
  // then.
  llvm::DenseMap<mlir::Operation *, smt::Expr> welldef;

public:
  class LinalgGenericScope {
  public:
    // Bound induction variables.
    std::vector<smt::Expr> indVars;
    // indVars[i] <= indVarUpperBounds[i]
    std::vector<smt::Expr> indVarUpperBounds;

    LinalgGenericScope(const std::vector<Index> &upperbounds);
  };

  RegFile regs;
  std::stack<LinalgGenericScope> linalgGenericScopes;
  // Return value tuples
  std::vector<ValueTy> retValues;

  bool hasQuantifier;
  std::shared_ptr<Memory> m;

  State(std::unique_ptr<Memory> &&initMem);

  void addPrecondition(smt::Expr &&e);
  void wellDefined(mlir::Operation *op, smt::Expr &&e);
  smt::Expr precondition() const;
  smt::Expr isWellDefined() const;
  smt::Expr isOpWellDefined(mlir::Operation *op) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
