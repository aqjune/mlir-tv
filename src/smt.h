#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>
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
expr fitsInDims(const std::vector<expr> &idxs,
                const std::vector<expr> &sizes);

std::string or_omit(const expr &e);
std::string or_omit(const std::vector<expr> &evec);

// TODO: these functions must be member functions of Expr
expr init();
expr substitute(expr e, const std::vector<expr> &vars,
                const std::vector<expr> &values);
expr implies(const expr &a, const expr &b);
expr forall(const std::vector<expr> &vars, const expr &e);
expr lambda(const expr &var, const expr &e);
expr lambda(const std::vector<expr> &vars, const expr &e);
expr select(const expr &arr, const expr &idx);
expr select(const expr &arr, const std::vector<expr> &idxs);
expr mkFreshVar(const sort &s, std::string &&prefix);
expr mkVar(const sort &s, std::string &&name);
expr mkBV(uint64_t i, unsigned bw);
expr mkBool(bool b);
func_decl mkUF(const sort &domain, const sort &range, std::string &&name);
func_decl mkUF(const std::vector<sort> &domain, const sort &range,
               std::string &&name);
expr fapply(const func_decl &func, const std::vector<expr> &vars);
bool structurallyEq(const expr &e1, const expr &e2);

// TODO: these functions must be member functions of Sort
sort bvSort(unsigned bw);
sort boolSort();
sort arraySort(const sort &domain, const sort &range);

class Sort;
class Solver;

class Expr {
  friend Solver;

private:
  std::optional<z3::expr> z3_expr;

  Expr(std::optional<z3::expr> &&z3_expr);

public:
  Expr simplify() const;
  Expr substitute(const std::vector<Expr> &vars, const std::vector<Expr> &values) const;
  Expr implies(const Expr &rhs) const;
  Expr select(const Expr &idx) const;
  Expr select(const std::vector<Expr> &indices) const;
  std::vector<Expr> toNDIndices(const std::vector<Expr> &dims) const;

  Expr urem(const Expr &rhs) const;
  Expr udiv(const Expr &rhs) const;
  Expr ult(const Expr &rhs) const;
  Expr ugt(const Expr &rhs) const;

  Expr operator+(const Expr &rhs);
  Expr operator-(const Expr &rhs);
  Expr operator*(const Expr &rhs);
  Expr operator&(const Expr &rhs);
  Expr operator|(const Expr &rhs);

  static Expr mkFreshVar(const Sort &s, std::string_view prefix);
  static Expr mkVar(const Sort &s, std::string_view name);
  static Expr mkBV(const uint64_t val, const size_t sz);
  static Expr mkBool(const bool val);

  friend z3::expr_vector toZ3ExprVector(const std::vector<Expr> &vec);
};

class Sort {
  friend Expr;

private:
  std::optional<z3::sort> z3_sort;

  Sort(std::optional<z3::sort> &&z3_sort);

public:
  static Sort bvSort(size_t bw);
  static Sort boolSort();
  static Sort arraySort(const Sort &domain, const Sort &range);
};

class Result {
public:
  enum Internal {
    SAT,
    UNSAT,
    UNKNOWN
  };

  Result(const std::optional<z3::check_result> &z3_result);
  const bool operator==(const Result &rhs);
  const bool operator!=(const Result &rhs) { return !(*this == rhs); }
  
private:
  Internal result;
};

class Solver {
private:
  std::optional<z3::solver> z3_solver;

public:
  Solver();

  void add(const Expr &e);
  void reset();
  Result check();
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es);
