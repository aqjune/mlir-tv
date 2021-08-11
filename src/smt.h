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

    bool checkZ3(const Expr& arg0) {
        return arg0.z3_expr.has_value();
    }

    template<typename... Es>
    bool checkZ3(const Expr& arg0, const Es&... args) {
        return arg0.z3_expr.has_value() && checkZ3(args...);
    }

    template<typename F, typename... Ts>
    void applyZ3Op(F&& op, const Expr& arg0, const Ts... args) {
        if (checkZ3(arg0, args...)) {
            this->replaceExpr(op(arg0.z3_expr.value(), args.z3_expr.value()...));
        }
    }

public:
    Expr() = default;
    Expr(const Expr& from) = default;
    Expr(Expr&& from) = default;
    Expr& operator=(const Expr& from) = default;
    Expr& operator=(Expr&& from) = default;
    // simplify expressions
    Expr simplify() const;
    // equivalent to from1DIdx
    std::vector<Expr> toElements(const std::vector<Expr>& dims) const;

    // update internal z3::expr and get previous z3::expr
    std::optional<z3::expr> replaceExpr(z3::expr&& z3_expr);

    Expr urem(const Expr& rhs) const;
    Expr udiv(const Expr& rhs) const;
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
