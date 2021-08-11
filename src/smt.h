#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>
#include <functional>
#include <optional>

namespace smt {
using expr = z3::expr;

class Expr;

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
z3::expr_vector toExprVector(const std::vector<expr> &vec);
std::string or_omit(const expr &e);

class Expr {
private:
  std::optional<z3::expr> z3_expr;

  Expr(std::optional<z3::expr> z3_expr): z3_expr(z3_expr) {}

public:
  Expr simplify() const;

  Expr urem(const Expr &rhs) const;
  Expr udiv(const Expr &rhs) const;
};
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
