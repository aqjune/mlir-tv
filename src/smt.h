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

class Context {
private:
    z3::context* z3_ctx;

    template<typename F, typename T, typename... Ts>
    std::optional<z3::expr> applyZ3Op(const F&& op, const T arg0, const Ts... args) {
        if (this->z3_ctx) {
            return std::optional(op(arg0, args...));
        } else {
            return {};
        }
    }

public:
    Context();
    void useZ3();

    Expr bvVal(const uint32_t val, const size_t sz);
    Expr bvConst(char* const name, const size_t sz);
};

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
    void applyZ3Op(const F&& op, const Expr& arg0, const Ts... args) {
        if (checkZ3(arg0, args...)) {
            this->replaceExpr(op(arg0.z3_expr.value(), args.z3_expr.value()...));
        }
    }

public:
    Expr() = default;
    Expr(const Expr& from) = default;
    Expr(Expr&& from) = default;
    Expr(std::optional<z3::expr>&& z3_expr);
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

class ExprVec {
private:
    std::vector<Expr> exprs;
    ExprVec(std::vector<Expr>&& exprs);
    ExprVec(ExprVec&& from);

public:
    size_t size() const;
    std::vector<Expr>::const_iterator cbegin() const;
    std::vector<Expr>::const_iterator cend() const;
    std::vector<Expr>::const_reverse_iterator crbegin() const;
    std::vector<Expr>::const_reverse_iterator crend() const;

    ExprVec simplify() const;
    Expr to1DSize() const;
    Expr to1DIdx(ExprVec dims) const;
    Expr to1DIdxWithLayout(Expr layout) const;
};

class Sort {
private:
    z3::sort z3_sort;

public:
};

class SortVec {
private:
    std::vector<Sort> sorts;

public:

};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
