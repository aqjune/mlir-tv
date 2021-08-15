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

expr getExpr(const ValueTy &v) {
  optional<expr> e;
  visit([&](auto &&itm) {
    e = (expr)itm;
  }, v);
  return move(*e);
}

ValueTy eval(const ValueTy &v, smt::model m) {
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

void RegFile::add(mlir::Value v, const expr &e, mlir::Type ty) {
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

expr RegFile::getExpr(mlir::Value v) const {
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
    indVars.emplace_back(Index("i" + to_string(i), true));
  }
}

State::State(unsigned int numBlocks, MemEncoding encoding):
  hasQuantifier(false),
  m(Memory::create(numBlocks, numBlocks, encoding)) {}

void State::wellDefined(mlir::Operation *val, expr &&e) {
  auto itr = welldef.find(val);
  if (itr == welldef.end()) {
    welldef.insert({val, move(e)});
  } else {
    itr->second = itr->second && move(e);
  }
}

expr State::isWellDefined() const {
  expr e = ctx.bool_val(true);
  for (auto &itm: welldef) {
    e = e && itm.second;
  }
  return e;
}