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

template<class T, class Fn>
auto fupdate(std::optional<T> &x, Fn fn) {
  if (x)
    return std::optional(fn(*x));
  return std::optional<decltype(fn(*x))>();
}

template<class T1, class T2, class Fn>
auto fupdate2(std::optional<T1> &x, const std::optional<T2> &x2, Fn fn) {
  if (x && x2)
    return std::optional(fn(*x, *x2));
  return std::optional<decltype(fn(*x, *x2))>();
}

class UnsupportedException : public std::exception {
  const char* reason;

public:
  UnsupportedException(const char* reason) : reason(reason) {}
  const char* what() {
    return reason;
  }
};
