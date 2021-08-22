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

template<class T_Z3, class T_CVC5>
class Object {
public:
  Object() {}

#ifdef SOLVER_Z3
  std::optional<T_Z3> z3;
  void setZ3(std::optional<T_Z3> &&e) { z3 = std::move(e); }
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  std::optional<T_CVC5> cvc5;
  void setCVC5(std::optional<T_CVC5> &&e) { cvc5 = std::move(e); }
#endif
};

class Expr: private Object<z3::expr, cvc5::api::Term> {
private:
  Expr() {}

public:
#ifdef SOLVER_Z3
  z3::expr getZ3Expr() const;
  bool hasZ3Expr() const;
#endif
#ifdef SOLVER_CVC5
  cvc5::api::Term getCVC5Term() const;
  bool hasCVC5Term() const;
#endif

  Expr simplify() const;
  Sort sort() const;
  std::vector<Expr> toNDIndices(const std::vector<Expr> &dims) const;

  // Returns true if at least one expr in z3, cvc5, ... is uint.
  // If it has different uint values (e.g., 10 in z3 and 20 in cvc5), it raises
  // a fault.
  bool isUInt(uint64_t &v) const;
  bool isInt(int64_t &v) const;
  // Returns true if at least one expr in z3, cvc5, ... is numeric.
  bool isNumeral() const;
  // Returns true if at least one expr in z3, cvc5, ... is constant false.
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
  Expr getMSB() const;
  Expr concat(const Expr &lowbits) const;
  Expr zext(unsigned bits) const;
  Expr sext(unsigned bits) const;

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

  // Returns true if this and e2's expr are equal.
  // Note that
  //   z3: two Z3_mk_const calls with the same name returns an identical expr
  //   cvc5: two Solver::mkConst calls with the same name returns different
  //         terms!
  // The meaning of isIdentical relies on which solver you are using.
  // If is_or is true, this returns true if at least one solver's expr is equal.
  // Otherwise, it returns true if all of the solvers' exprs are equivalent.
  bool isIdentical(const Expr &e2, bool is_or = true) const;

  // Make a fresh, unbound variable.
  static Expr mkFreshVar(const Sort &s, const std::string &prefix);
  // Set boundVar to true if the variable is to be used in a binder (e.g.,
  // a quantified variable, lambda).
  static Expr mkVar(
      const Sort &s, const std::string &name, bool boundVar = false);
  static Expr mkBV(const uint64_t val, const size_t sz);
  static Expr mkBool(const bool val);

  static Expr mkForall(const std::vector<Expr> &vars, const Expr &body);
  static Expr mkLambda(const Expr &var, const Expr &body);
  static Expr mkLambda(const std::vector<Expr> &vars, const Expr &body);
  static Expr mkSplatArray(const Sort &domain, const Expr &splatElem);
  static Expr mkIte(const Expr &cond, const Expr &then, const Expr &els);

  static Expr mkAddNoOverflow(const Expr &a, const Expr &b, bool is_signed);


  friend FnDecl;
  friend Model;
  friend Solver;
  friend matchers::Matcher;
};

class FnDecl: private Object<z3::func_decl, cvc5::api::Term> {
private:
  FnDecl() {}

public:
  FnDecl(const Sort &domain, const Sort &range, std::string &&name);
  FnDecl(const std::vector<Sort> &domain, const Sort &range,
         std::string &&name);

  Expr apply(const std::vector<Expr> &args) const;
  Expr apply(const Expr &arg) const;
  Expr operator()(const Expr &arg) const { return apply(arg); }

  friend Expr;
};

class Sort: private Object<z3::sort, cvc5::api::Sort> {
private:
  Sort() {}

public:
  IF_Z3_ENABLED(z3::sort getZ3Sort() const);
  IF_Z3_ENABLED(cvc5::api::Sort getCVC5Sort() const);

  unsigned bitwidth() const;
  bool isArray() const;
  Sort getArrayDomain() const;
  bool isBV() const;

  // Convert to a function type if this is an array type.
  // This is necessary in CVC5 because it differentiates those two.
  Sort toFnSort() const;

  static Sort bvSort(size_t bw);
  static Sort boolSort();
  static Sort arraySort(const Sort &domain, const Sort &range);

  friend Expr;
  friend FnDecl;
};

class CheckResult : private Object<z3::check_result, cvc5::api::Result> {
private:
  CheckResult() {}

public:
  bool isUnknown() const;
  bool hasSat() const;
  bool hasUnsat() const;
  // Has both SAT and UNSAT?
  bool isInconsistent() const;

  friend Solver;
};

// TODO: Model and Solver are special because they simply reuse
// cvc5::api::Solver.
class Model {
private:
  Model() {}

#ifdef SOLVER_Z3
  std::optional<z3::model> z3;
  void setZ3(std::optional<z3::model> &&m) { z3 = std::move(m); }
#endif
  // No need for CVC5

public:
  Expr eval(const Expr &e, bool modelCompletion = false) const;

  static Model empty();

  friend Solver;
};

class Solver {
private:
#ifdef SOLVER_Z3
  std::optional<z3::solver> z3;
#endif
  // No need for CVC5

public:
  Solver(const char *logic);
  Solver(const Solver &) = delete;

  void add(const Expr &e);
  void reset();
  CheckResult check();
  Model getModel() const;
};

void useZ3();
void useCVC5();
void setTimeout(const uint64_t ms);
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::Expr &e);
std::ostream& operator<<(std::ostream& os, const smt::Expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::Expr> &es);
