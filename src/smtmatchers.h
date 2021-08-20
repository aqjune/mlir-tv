#pragma once

#include "smt.h"
#include <optional>
#include "z3++.h"

namespace smt {
namespace matchers {

class Matcher {
protected:
  Expr createExpr(std::optional<z3::expr> &&ze) const;
};

class Any: Matcher {
  std::optional<Expr> *e;

public:
  Any(std::optional<Expr> &e): e(&e) {}

  bool match(const Expr &expr) const { return (*this)(expr); }
  bool operator()(const Expr &expr) const {
    e->emplace(std::move(expr));
    return true;
  }
};

class ConstSplatArray: Matcher {
  std::function<bool(const Expr &)> subMatcher;

public:
  template<class T>
  ConstSplatArray(T &&m): subMatcher(std::move(m)) {}

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

}
}