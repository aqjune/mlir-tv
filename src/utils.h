#pragma once

#include <optional>
#include <string>
#include <variant>
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

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

inline unsigned log2_ceil(unsigned count) {
  unsigned bits = 0;
  while (count > (1 << bits)) {
    bits += 1;
  }
  return bits;
}

#define TO_STRING(msg, V) { \
  llvm::raw_string_ostream rso(msg); \
  rso << V; \
  rso.flush(); \
  msg = rso.str(); }

class UnsupportedException : public std::exception {
  // 0: the operation is unsupported
  // 1: the type is unsupported
  std::variant<mlir::Operation *, mlir::Type> obj;
  std::string reason;

  UnsupportedException(decltype(obj) obj, decltype(reason) reason)
      : obj(obj), reason(reason) {}

public:
  UnsupportedException(std::string &&reason)
      : obj((mlir::Operation *)nullptr), reason(move(reason)) {}
  UnsupportedException(mlir::Operation *obj): obj(obj) {}
  UnsupportedException(mlir::Operation *obj, std::string &&reason)
      : obj(obj), reason(move(reason)) {}
  UnsupportedException(mlir::Type ty): obj(ty) {}
  UnsupportedException(mlir::Type ty, std::string &&reason)
      : obj(ty), reason(move(reason)) {}

  std::string getReason() const { return reason; }
  auto getObject() const { return obj; }
};

template<class ValueTy>
using TypeMap = mlir::DenseMap<mlir::Type, ValueTy>;

std::string to_string(mlir::Type t);