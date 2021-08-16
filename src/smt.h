#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>
#include <functional>
#include <optional>

namespace smt {
using expr = z3::expr;
using model = z3::model;
using sort = z3::sort;
using func_decl = z3::func_decl;

class Expr;
class ExprVec;
class Context;

extern z3::context ctx;

expr get1DSize(const std::vector<expr> &dims);
std::vector<expr> from1DIdx(
    expr idx1d, const std::vector<expr> &dims);
std::vector<expr> simplifyList(const std::vector<expr> &exprs);

expr to1DIdx(const std::vector<expr> &idxs,
                const std::vector<expr> &dims);
expr to1DIdxWithLayout(const std::vector<expr> &idxs, expr layout);
expr fitsInDims(const std::vector<expr> &idxs,
                const std::vector<expr> &sizes);

std::string or_omit(const expr &e);
std::string or_omit(const std::vector<expr> &evec);

// TODO: these functions must be member functions of Expr
expr substitute(expr e, const std::vector<expr> &vars,
                const std::vector<expr> &values);
expr forall(const std::vector<expr> &vars, const expr &e);
expr mkFreshVar(const sort &s, std::string &&prefix);
expr mkVar(const sort &s, std::string &&name);
expr mkBV(uint64_t i, unsigned bw);
expr mkBool(bool b);
func_decl mkUF(const sort &domain, const sort &range, std::string &&name);
func_decl mkUF(const std::vector<sort> &domain, const sort &range,
               std::string &&name);
bool structurallyEq(const expr &e1, const expr &e2);

// TODO: these functions must be member functions of Sort
sort bvSort(unsigned bw);
sort boolSort();
sort arraySort(const sort &domain, const sort &range);

class ContextBuilder {
  private:
    bool use_z3;

  public:
    ContextBuilder();
    ContextBuilder& useZ3();
    std::optional<Context> build() const;
};

class Context {
  friend ContextBuilder;

private:
    z3::context* z3_ctx;
    Context();
    Context(bool use_z3);

public:
    Expr bvVal(const uint32_t val, const size_t sz);
    Expr bvConst(char* const name, const size_t sz);
    Expr boolVal(const bool val);
};

class Expr {
  friend Context;
private:
  Context* ctx;
  std::optional<z3::expr> z3_expr;
  
  Expr(Context* const ctx) : ctx(ctx) {};
  Expr(Context* const ctx, std::optional<z3::expr> &&z3_expr);

public:
  Expr(Expr&& from) = default;
  Expr& operator=(Expr &&from) = default;
  Expr clone() const;

  Expr simplify() const;
  ExprVec toNDIndices(const ExprVec &dims) const;

  Expr urem(const Expr &rhs) const;
  Expr udiv(const Expr &rhs) const;
  Expr add(const Expr &rhs) const;
  Expr sub(const Expr &rhs) const;
  Expr mul(const Expr &rhs) const;
  Expr ult(const Expr &rhs) const;
  Expr ugt(const Expr &rhs) const;
  Expr boolAnd(const Expr &rhs) const;
  Expr boolOr(const Expr &rhs) const;
};

class ExprVec {
  friend Context;
  friend Expr;

private:
  Context* ctx;
  std::vector<Expr> exprs;

  ExprVec(Context* const ctx): ctx(ctx) {};
  ExprVec(Context* const ctx, std::vector<Expr>&& exprs);

  static ExprVec withCapacity(Context* ctx, size_t size);

public:
  ExprVec(ExprVec&& from) = default;
  ExprVec& operator=(ExprVec &&from) = default;
  ExprVec clone() const;

  ExprVec simplify() const;
  Expr to1DIdx(const ExprVec &dims) const;
  Expr fitsInDims(const ExprVec &sizes) const;
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
