#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>
#include <functional>
#include <optional>

namespace smt {
using expr = z3::expr;

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
    Expr(const Expr& from);

    void applyZ3Operation(std::function<z3::expr(z3::expr const&)>&& op, const Expr& arg0);
    void applyZ3Operation(std::function<z3::expr(z3::expr const&, z3::expr const&)>&& op, const Expr& arg0, const Expr& arg1);

public:
    // default constructor
    Expr();
    // move constructor
    Expr(Expr&& from);
    // explicit copy 
    Expr clone() const;
    // move assignment operator
    Expr& operator=(Expr&& from);
    // simplify expressions
    Expr simplify() const;
    // equivalent to from1DIdx
    std::vector<Expr> toElements(const std::vector<Expr>& dims) const;

    // update internal z3::expr and get previous z3::expr
    std::optional<z3::expr> replaceExpr(z3::expr&& z3_expr);

    friend Expr urem(const Expr& lhs, const Expr& rhs);
    friend Expr udiv(const Expr& lhs, const Expr& rhs);
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
