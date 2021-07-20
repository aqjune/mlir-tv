#pragma once

#include "memory.h"
#include "value.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <stack>
#include <variant>
#include "mlir/Support/LLVM.h"

using ValueTy = std::variant<Tensor, MemRef, Index, Float, Integer>;

class RegFile {
private:
  llvm::DenseMap<mlir::Value, ValueTy> m;

public:
  void add(mlir::Value v, ValueTy &&t);

  // For non-aggregate types only
  void add(mlir::Value v, const z3::expr &e, mlir::Type ty);

  ValueTy findOrCrash(mlir::Value v) const;
  template<class T> T get(mlir::Value v) const {
    return std::get<T>(findOrCrash(v));
  }
  bool contains(mlir::Value v) const;
  z3::expr getZ3Expr(mlir::Value v) const;

  auto begin() const { return m.begin(); }
  auto end() const { return m.end(); }
};

class State {
public:

  class LinalgGenericScope {
  public:
    std::vector<z3::expr> indVars;
    // indVars[i] <= indVarUpperBounds[i]
    std::vector<z3::expr> indVarUpperBounds;

    LinalgGenericScope(std::vector<Index> &&upperbounds);
  };

  RegFile regs;
  std::stack<LinalgGenericScope> linalgGenericScopes;
  // If returns void, it is nullopt
  std::optional<ValueTy> retValue;

  // The negated form of UB is tracked because the neg. of value refinement is:
  // 'src.no-ub /\ tgt.no-ub /\ src.retvalue != tgt.retvalue'.
  // We'll need to implement our own version of peephole optimizations on Z3
  // expr some day (or simply use Alive2's one), and this form will be helpful
  // then.
  z3::expr isWellDefined;
  Memory *m;

  State(unsigned int numBlocks);

  void wellDefined(const z3::expr &e) { isWellDefined = isWellDefined && e; };

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
