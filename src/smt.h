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

class Expr {
private:
  std::optional<z3::expr> z3_expr;

  Expr(std::optional<z3::expr> &&z3_expr);

public:
  Expr simplify() const;
  std::vector<Expr> toNDIndices(const std::vector<Expr> &dims) const;

  Expr urem(const Expr &rhs) const;
  Expr udiv(const Expr &rhs) const;
  Expr ult(const Expr &rhs) const;
  Expr ugt(const Expr &rhs) const;

  friend Expr operator+(const Expr &lhs, const Expr &rhs);
  friend Expr operator-(const Expr &lhs, const Expr &rhs);
  friend Expr operator*(const Expr &lhs, const Expr &rhs);
  friend Expr operator&(const Expr &lhs, const Expr &rhs);
  friend Expr operator|(const Expr &lhs, const Expr &rhs);

  Expr mkBV(const uint32_t val, const size_t sz);
  Expr mkVar(char* const name, const size_t sz);
  Expr mkBool(const bool val);
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
