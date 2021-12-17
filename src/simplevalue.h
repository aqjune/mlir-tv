#pragma once

// Values in MLIR that do not require special techniques to encode in SMT.

#include "smt.h"

#include "llvm/Support/raw_ostream.h"
#include <string>
#include <optional>
#include <vector>

enum class VarType {
  BOUND, // a bound variable; see Expr::mkVar
  FRESH, // a fresh, unbound variable
  UNBOUND
};

class Integer {
  smt::Expr e;

public:
  Integer(const smt::Expr &e): e(e) { this->e.unlockOps(); }
  Integer(int64_t i, unsigned bw);
  Integer(const llvm::APInt &api);

  operator smt::Expr() const { return e; }
  unsigned bitwidth() { return e.bitwidth(); }

  static smt::Sort sort(unsigned bw);
  static Integer var(std::string &&name, unsigned bw, VarType vty);
  static Integer boolTrue();
  static Integer boolFalse();

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Integer &);
  // (refinement, {})
  std::pair<smt::Expr, std::vector<smt::Expr>> refines(const Integer &other)
      const;
  Integer eval(smt::Model m) const;
};

class Index {
  smt::Expr e;

public:
  static const unsigned BITS = 32;

  Index(unsigned);
  Index(const smt::Expr &e): e(e) { this->e.unlockOps(); }
  Index(smt::Expr &&e): e(std::move(e)) { this->e.unlockOps(); }

  operator smt::Expr() const { return e; }
  Index operator*(const Index &b) const { return Index(e * b.e); }
  Index operator+(const Index &b) const { return Index(e + b.e); }
  Index operator-(const Index &b) const { return Index(e - b.e); }
  smt::Expr operator==(const Index &b) const { return e == b.e; }
  Index ofs(int i) const {
    uint64_t v;
    if (e.isUInt(v))
      return Index(v + i);
    return Index(e + i);
  }

  static smt::Sort sort();
  static Index one();
  static Index zero();
  static Index var(std::string &&name, enum VarType);
  // Creates N variables that must be bound to forall/lambda/exists.
  static std::vector<smt::Expr> boundIndexVars(unsigned N);
  static Index fromInteger(const Integer &i) { return Index((smt::Expr)i); }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Index &);
  // (refinement, unbound variables used in the refinement formula)
  std::pair<smt::Expr, std::vector<smt::Expr>> refines(
      const Index &other) const;
  Index eval(smt::Model m) const;
  Integer asInteger() const { return Integer(e); }
};
