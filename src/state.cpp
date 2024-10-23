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
  if (mlir::isa<mlir::FloatType>(ty))
    m.insert({v, Float(e, ty)});
  else if (mlir::isa<mlir::IntegerType>(ty))
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
  m(std::move(initMem)) {}

void State::addPrecondition(smt::Expr &&e) {
  precond = precond & e;
}

void State::wellDefined(mlir::Operation *val, Expr &&e, string &&desc) {
  auto itr = welldef.find(val);
  if (itr == welldef.end()) {
    itr = welldef.insert({val, {}}).first;
  }

  auto &ubmap = itr->second;
  auto itr2 = ubmap.find(desc);
  if (itr2 == ubmap.end()) {
    ubmap.insert({std::move(desc), std::move(e)});
  } else {
    itr2->second = itr2->second & std::move(e);
  }
}

Expr State::precondition() const {
  return precond;
}

Expr State::isWellDefined() const {
  Expr e = Expr::mkBool(true);
  for (auto &ubmap: welldef) {
    for (auto &itm: ubmap.second)
      e = e & itm.second;
  }
  return e;
}

Expr State::isOpWellDefined(mlir::Operation *op) const {
  auto ubmap = welldef.find(op);
  if (ubmap == welldef.end())
    return Expr::mkBool(true);

  Expr e = Expr::mkBool(true);
  for (auto &itm: ubmap->second)
    e &= itm.second;
  return e;
}

map<string, Expr> State::getOpWellDefinedness(mlir::Operation *op) const {
  auto ubmap = welldef.find(op);
  if (ubmap == welldef.end())
    return {};
  return ubmap->second;
}
