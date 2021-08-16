#pragma once

#include <optional>

// optional::map from
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0798r0.html
// Fn is simply declared because std::function with template arguments works
// poorly. :(
template<class T, class Fn>
auto fmap(const std::optional<T> &x, Fn fn) {
  if (x)
    return std::optional(fn(*x));
  return std::optional<decltype(fn(*x))>();
}
