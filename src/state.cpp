#include "smt.h"
#include "state.h"

using namespace std;

void RegFile::add(mlir::Value v, ValueTy &&t) {
  assert(!contains(v));
  m.emplace_back(v, std::move(t));
}

bool RegFile::contains(mlir::Value v) const {
  for (auto &[a, b]: m) {
    if (a == v)
      return true;
  }
  return false;
}

z3::expr RegFile::getZ3Expr(mlir::Value v) const {
  for (auto &[a, b]: m) {
    if (a != v) continue;
    z3::expr e(ctx);
    visit([&](auto &&itm) {
      e = (z3::expr)itm;
    }, b);
    return e;
  }

  llvm::errs() << "Cannot find key: " << v << "\n";
  assert(false && "Unknown key");
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

pair<z3::expr, vector<z3::expr>> State::refines(const State &src) {
  // TODO: encode the final memory
  auto [refines, idx] = retValue.refines(src.retValue);
  return {move(refines), {idx}};
}
