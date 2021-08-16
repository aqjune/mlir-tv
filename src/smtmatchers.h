#pragma once

#include "smt.h"
#include <optional>

namespace smt {
namespace matchers {

class Any {
  std::optional<expr> *e;

public:
  Any(std::optional<expr> &e): e(&e) {}

  bool match(const expr &expr) const {
    e->emplace(std::move(expr));
    return true;
  }
};

template<class T>
class ConstSplatArray {
  T subMatcher;

public:
  ConstSplatArray(T &&m): subMatcher(std::move(m)) {}

  bool match(const expr &e) const {
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(ctx, a);
    if (Z3_get_decl_kind(ctx, decl) != Z3_OP_CONST_ARRAY)
      return false;

    return subMatcher.match({ctx, Z3_get_app_arg(ctx, a, 0)});
  }
};

template<class T1, class T2, class T3>
class Store {
  T1 arrMatcher;
  T2 idxMatcher;
  T3 valMatcher;

public:
  Store(T1 &&arr, T2 &&idx, T3 &&val):
      arrMatcher(std::move(arr)), idxMatcher(std::move(idx)),
      valMatcher(std::move(val)) {}

  bool match(const expr &e) const {
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(ctx, a);
    if (Z3_get_decl_kind(ctx, decl) != Z3_OP_STORE)
      return false;

    return arrMatcher.match({ctx, Z3_get_app_arg(ctx, a, 0)}) &&
        idxMatcher.match({ctx, Z3_get_app_arg(ctx, a, 1)}) &&
        valMatcher.match({ctx, Z3_get_app_arg(ctx, a, 2)});
  }
};

}
}