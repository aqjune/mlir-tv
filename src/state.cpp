#include "smt.h"
#include "state.h"

using namespace smt;
using namespace std;



ValueTy RegFile::findOrCrash(mlir::Value v) const {
  auto itr = m.find(v);
  if (itr != m.end()) {
    return itr->second;
  } else {
    llvm::errs() << "Cannot find key: " << v << "\n";
    abort();
  }
}

void RegFile::add(mlir::Value v, ValueTy &&t) {
  assert(!contains(v));
  m.insert({v, std::move(t)});
}

void RegFile::add(mlir::Value v, const Expr &e, mlir::Type ty) {
  assert(!contains(v));
  if (ty.isa<mlir::FloatType>())
    m.insert({v, Float(e, ty)});
  else if (ty.isa<mlir::IntegerType>())
    m.insert({v, Integer(e)});
  else if (ty.isIndex())
    m.insert({v, Index(e)});
  else {
    llvm::errs() << "Unsupported type: " << ty << "\n";
    abort();
  }
}

bool RegFile::contains(mlir::Value v) const {
  return (bool)m.count(v);
}

Expr RegFile::getExpr(mlir::Value v) const {
  return ::getExpr(findOrCrash(v));
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, State &s) {
  for (auto itm: s.regs) {
    os << "Register: " << itm.first;
    os << "Value: ";
    visit([&](auto &&itm) {
      os << itm;
    }, itm.second);
    os << "\n";
  }
  return os;
}

State::LinalgGenericScope::LinalgGenericScope(
    const std::vector<Index> &upperbounds) {
  for (unsigned i = 0; i < upperbounds.size(); ++i) {
    indVarUpperBounds.push_back(upperbounds[i]);
    indVars.emplace_back(
        Index::var("i" + to_string(i), VarType::BOUND));
  }
}

State::State(unique_ptr<Memory> &&initMem):
  precond(Expr::mkBool(true)), hasQuantifier(false), hasConstArray(false),
  m(move(initMem)) {}

void State::addPrecondition(smt::Expr &&e) {
  precond = precond & e;
}

void State::wellDefined(mlir::Operation *val, Expr &&e) {
  auto itr = welldef.find(val);
  if (itr == welldef.end()) {
    welldef.insert({val, move(e)});
  } else {
    itr->second = itr->second & move(e);
  }
}

Expr State::precondition() const {
  return precond;
}

Expr State::isWellDefined() const {
  Expr e = Expr::mkBool(true);
  for (auto &itm: welldef) {
    e = e & itm.second;
  }
  return e;
}

Expr State::isOpWellDefined(mlir::Operation *op) const {
  if (!welldef.count(op))
    return Expr::mkBool(true);
  return welldef.find(op)->second;
}
