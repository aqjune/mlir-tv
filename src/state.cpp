#include "smt.h"
#include "state.h"

using namespace std;

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

bool RegFile::contains(mlir::Value v) const {
  return (bool)m.count(v);
}

z3::expr RegFile::getZ3Expr(mlir::Value v) const {
  auto var = findOrCrash(v);
  z3::expr e(ctx);
  visit([&](auto &&itm) {
    e = (z3::expr)itm;
  }, var);
  return e;
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

State::State(): isWellDefined(ctx) {}