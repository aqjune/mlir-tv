#include "smt.h"
#include "state.h"

using namespace smt;
using namespace std;

llvm::raw_ostream& operator<<(llvm::raw_ostream &os, const ValueTy &v) {
  visit([&](auto &&itm) {
    os << itm;
  }, v);
  return os;
}

Expr getExpr(const ValueTy &v) {
  optional<Expr> e;
  visit([&](auto &&itm) {
    e = (Expr)itm;
  }, v);
  return move(*e);
}

ValueTy eval(const ValueTy &v, smt::Model m) {
  optional<ValueTy> e;
  visit([&](auto &&itm) {
    e = itm.eval(m);
  }, v);
  return move(*e);
}


ValueTy RegFile::findOrCrash(mlir::Value v) const {
  auto itr = m.find(v);
  if (itr != m.end()) {
    return itr->second;
  } else {
    llvm::errs() << "Cannot find key: " << v << "\n";
    llvm_unreachable("Unknown key");
  }
}

void RegFile::add(mlir::Value v, ValueTy &&t) {
  assert(!contains(v));
  m.insert({v, std::move(t)});
}

void RegFile::add(mlir::Value v, const Expr &e, mlir::Type ty) {
  assert(!contains(v));
  if (ty.isa<mlir::Float32Type>())
    m.insert({v, Float(e)});
  else if (ty.isa<mlir::IntegerType>())
    m.insert({v, Integer(e)});
  else if (ty.isa<mlir::IndexType>())
    m.insert({v, Index(e)});
  else
    // TODO: tensor?
    llvm_unreachable("Unsupported type");
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
    std::vector<Index> &&upperbounds) {
  for (unsigned i = 0; i < upperbounds.size(); ++i) {
    indVarUpperBounds.push_back(upperbounds[i]);
    indVars.emplace_back(
        Index::var("i" + to_string(i), VarType::BOUND));
  }
}

State::State(unique_ptr<Memory> &&initMem):
  hasQuantifier(false), precond(Expr::mkBool(true)),
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
