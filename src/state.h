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
  void add(mlir::Value v, const smt::expr &e, mlir::Type ty);

  ValueTy findOrCrash(mlir::Value v) const;
  template<class T> T get(mlir::Value v) const {
    return std::get<T>(findOrCrash(v));
  }
  bool contains(mlir::Value v) const;
  smt::expr getZ3Expr(mlir::Value v) const;

  auto begin() const { return m.begin(); }
  auto end() const { return m.end(); }
};

class State {
public:

  class LinalgGenericScope {
  public:
    std::vector<smt::expr> indVars;
    // indVars[i] <= indVarUpperBounds[i]
    std::vector<smt::expr> indVarUpperBounds;

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
  bool hasQuantifier;
  smt::expr isWellDefined;
  std::shared_ptr<Memory> m;

  State(unsigned int numBlocks, MemEncoding encoding);

  void wellDefined(const smt::expr &e) {
    isWellDefined = isWellDefined && e;
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};
