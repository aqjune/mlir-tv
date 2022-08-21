#pragma once

#include "memory.h"
#include "smt.h"
#include "value.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include <stack>
#include <optional>
#include "mlir/Support/LLVM.h"


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
  // op -> (desc -> cond)
  llvm::DenseMap<mlir::Operation *,
      std::map<std::string, smt::Expr>> welldef;

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
  bool hasConstArray;
  std::shared_ptr<Memory> m;

  State(std::unique_ptr<Memory> &&initMem);

  void addPrecondition(smt::Expr &&e);
  void wellDefined(mlir::Operation *op, smt::Expr &&e, std::string &&desc = "");
  smt::Expr precondition() const;
  smt::Expr isWellDefined() const;
  smt::Expr isOpWellDefined(mlir::Operation *op) const;
  std::map<std::string, smt::Expr> getOpWellDefinedness(mlir::Operation *op)
      const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
