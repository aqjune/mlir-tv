#pragma once

#include "smt.h"
#include <optional>

namespace smt {
namespace matchers {

class Matcher {
protected:
  Expr newExpr() const { return Expr(); }
#ifdef SOLVER_Z3
  void setZ3(Expr &e, std::optional<z3::expr> &&ze) const;
  bool matchBinaryOp(const Expr &expr, Z3_decl_kind z3Kind,
      std::function<bool(const Expr&)> lhsMatcher,
      std::function<bool(const Expr&)> rhsMatcher) const;
#endif
#ifdef SOLVER_CVC5
  void setCVC5(Expr &e, std::optional<cvc5::Term> &&opt) const;
  bool matchBinaryOp(const Expr &expr, cvc5::Kind opKind,
    std::function<bool(const Expr&)> lhsMatcher,
    std::function<bool(const Expr&)> rhsMatcher) const;
#endif
};

class Any: Matcher {
  std::optional<Expr> *e;

public:
  Any(std::optional<Expr> &e): e(&e) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &expr) const {
    e->emplace(expr);
    return true;
  }
};

class ConstBool: Matcher {
  bool val;

public:
  ConstBool(bool val): val(val) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class ConstSplatArray: Matcher {
  std::function<bool(const Expr &)> subMatcher;

public:
  template<class T>
  ConstSplatArray(T &&m): subMatcher(std::move(m)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class Lambda: Matcher {
  std::function<bool(const Expr &)> bodyMatcher, idxMatcher;

public:
  template<class T1>
  Lambda(T1 &&m): bodyMatcher(std::move(m)) {}

  // NOTE: this constructor must not be used when using Z3; idxMatcher is not used.
  template<class T1, class T2>
  Lambda(T1 &&m, T2 &&idx):
    bodyMatcher(std::move(m)), idxMatcher(std::move(idx)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class Store: Matcher {
  std::function<bool(const Expr &)> arrMatcher, idxMatcher, valMatcher;

public:
  template<class T1, class T2, class T3>
  Store(T1 &&arr, T2 &&idx, T3 &&val):
      arrMatcher(std::move(arr)), idxMatcher(std::move(idx)),
      valMatcher(std::move(val)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class Concat: Matcher {
  std::function<bool(const Expr &)> lhsMatcher, rhsMatcher;

public:
  template<class T1, class T2>
  Concat(T1 &&lhs, T2 &&rhs):
      lhsMatcher(std::move(lhs)), rhsMatcher(std::move(rhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class ZeroExt: Matcher {
  std::function<bool(const Expr &)> matcher;

public:
  template<class T1>
  ZeroExt(T1 &&lhs): matcher(std::move(lhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class URem: Matcher {
  std::function<bool(const Expr &)> lhsMatcher, rhsMatcher;

public:
  template<class T1, class T2>
  URem(T1 &&lhs, T2 &&rhs):
      lhsMatcher(std::move(lhs)), rhsMatcher(std::move(rhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class Mul: Matcher {
  std::function<bool(const Expr &)> lhsMatcher, rhsMatcher;

public:
  template<class T1, class T2>
  Mul(T1 &&lhs, T2 &&rhs):
      lhsMatcher(std::move(lhs)), rhsMatcher(std::move(rhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class UDiv: Matcher {
  std::function<bool(const Expr &)> lhsMatcher, rhsMatcher;

public:
  template<class T1, class T2>
  UDiv(T1 &&lhs, T2 &&rhs):
      lhsMatcher(std::move(lhs)), rhsMatcher(std::move(rhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

class Equals: Matcher {
  std::function<bool(const Expr &)> lhsMatcher, rhsMatcher;

public:
  template<class T1, class T2>
  Equals(T1 &&lhs, T2 &&rhs):
      lhsMatcher(std::move(lhs)), rhsMatcher(std::move(rhs)) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &e) const;
};

}
}
