#pragma once

#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <optional>

#ifdef SOLVER_Z3
  #include "z3++.h"
  #define IF_Z3_ENABLED(stmt) stmt
#else
  #define IF_Z3_ENABLED(stmt)
#endif

#ifdef SOLVER_CVC5
  #include "cvc5/cvc5.h"
  #define IF_CVC5_ENABLED(stmt) stmt
#else
  #define IF_CVC5_ENABLED(stmt)
#endif

namespace smt {

class Context {
public:
  std::optional<z3::context> z3;

  Context() {}
  Context(bool use_z3) {
    if (use_z3) {
      this->z3.emplace();
    }
  }
};

extern Context sctx;

class Expr;
class FnDecl;
class Model;
class Sort;

namespace matchers {
class Matcher;
}

Expr get1DSize(const std::vector<Expr> &dims);
std::vector<Expr> from1DIdx(
    Expr idx1d, const std::vector<Expr> &dims);
std::vector<Expr> simplifyList(const std::vector<Expr> &exprs);

Expr to1DIdx(const std::vector<Expr> &idxs,
                const std::vector<Expr> &dims);
Expr fitsInDims(const std::vector<Expr> &idxs,
                const std::vector<Expr> &sizes);

std::string or_omit(const Expr &e);
std::string or_omit(const std::vector<Expr> &evec);


class Solver;

class Expr {
private:
  std::optional<z3::expr> z3_expr;

  Expr(std::optional<z3::expr> &&z3_expr);

public:
  z3::expr getZ3Expr() const; // Crashes if z3_expr is nullopt

  Expr simplify() const;
  Sort sort() const;
  std::vector<Expr> toNDIndices(const std::vector<Expr> &dims) const;

  bool isUInt(uint64_t &v) const;
  bool isInt(int64_t &v) const;
  bool isNumeral() const;
  bool isFalse() const;

  Expr urem(const Expr &rhs) const;
  Expr urem(uint64_t rhs) const;
  Expr udiv(const Expr &rhs) const;
  Expr udiv(uint64_t rhs) const;
  Expr mod(const Expr &rhs) const;
  Expr mod(uint64_t rhs) const;

  Expr ult(const Expr &rhs) const;
  Expr ult(uint64_t rhs) const;
  Expr ule(const Expr &rhs) const;
  Expr ule(uint64_t rhs) const;
  Expr ugt(const Expr &rhs) const;
  Expr ugt(uint64_t rhs) const;
  Expr uge(const Expr &rhs) const;
  Expr uge(uint64_t rhs) const;

  Expr select(const Expr &idx) const;
  Expr select(const std::vector<Expr> &idxs) const;
  Expr store(const Expr &idx, const Expr &val) const;
  Expr store(uint64_t idx, const Expr &val) const;

  Expr extract(unsigned hbit, unsigned lbit) const;
  Expr concat(const Expr &lowbits) const;

  Expr operator+(const Expr &rhs) const;
  Expr operator+(uint64_t rhs) const;
  Expr operator-(const Expr &rhs) const;
  Expr operator-(uint64_t rhs) const;
  Expr operator*(const Expr &rhs) const;
  Expr operator*(uint64_t rhs) const;
  Expr operator&(const Expr &rhs) const;
  Expr operator|(const Expr &rhs) const;
  Expr operator==(const Expr &rhs) const;
  Expr operator==(uint64_t rhs) const;
  Expr operator!() const;

  Expr implies(const Expr &rhs) const;
  Expr isNonZero() const;

  Expr substitute(const std::vector<Expr> &vars,
                  const std::vector<Expr> &values) const;

  bool structurallyEq(const Expr &e2) const;

  static Expr mkFreshVar(const Sort &s, std::string_view prefix);
  static Expr mkVar(const Sort &s, std::string_view name);
  static Expr mkBV(const uint64_t val, const size_t sz);
  static Expr mkBool(const bool val);

  static Expr mkForall(const std::vector<Expr> &vars, const Expr &body);
  static Expr mkLambda(const Expr &var, const Expr &body);
  static Expr mkLambda(const std::vector<Expr> &vars, const Expr &body);
  static Expr mkConstArray(const Sort &domain, const Expr &splatElem);
  static Expr mkIte(const Expr &cond, const Expr &then, const Expr &els);

  static Expr mkAddNoOverflow(const Expr &a, const Expr &b, bool is_signed);


  friend Context;
  friend FnDecl;
  friend Model;
  friend Solver;
  friend matchers::Matcher;
};

class FnDecl {
private:
  std::optional<z3::func_decl> z3_fdecl;

  FnDecl(std::optional<z3::func_decl> &&z3_fdecl);

public:
  FnDecl(const Sort &domain, const Sort &range, std::string &&name);
  FnDecl(const std::vector<Sort> &domain, const Sort &range,
         std::string &&name);

  Expr apply(const std::vector<Expr> &args) const;
  Expr apply(const Expr &arg) const;
  Expr operator()(const Expr &arg) const { return apply(arg); }

  friend Expr;
};

class Sort {
private:
  std::optional<z3::sort> z3_sort;

  Sort(std::optional<z3::sort> &&z3_sort);

public:
  unsigned bitwidth() const;
  bool isArray() const;
  Sort getArrayDomain() const;
  bool isBV() const;

  static Sort bvSort(size_t bw);
  static Sort boolSort();
  static Sort arraySort(const Sort &domain, const Sort &range);

  friend Expr;
  friend FnDecl;
};

class CheckResult {
public:
  CheckResult(const std::optional<z3::check_result> &z3_result);
  const bool operator==(const CheckResult &rhs);
  const bool operator!=(const CheckResult &rhs) { return !(*this == rhs); }

  bool isUnknown() const { return result == UNKNOWN; }
  bool isSat() const { return result == SAT; }
  bool isUnsat() const { return result == UNSAT; }

private:
  enum Internal {
    SAT,
    UNSAT,
    UNKNOWN
  };

  Internal result;
};

class Model {
private:
  std::optional<z3::model> z3;

  Model(std::optional<z3::model> &&z3): z3(move(z3)) {}

public:
  Expr eval(const Expr &e, bool modelCompletion = false) const;

  static Model empty();

  friend Solver;
};

class Solver {
private:
  std::optional<z3::solver> z3;

public:
  Solver(const char *logic);

  void add(const Expr &e);
  void reset();
  CheckResult check();
  Model getModel() const;
};
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::Expr &e);
std::ostream& operator<<(std::ostream& os, const smt::Expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::Expr> &es);
