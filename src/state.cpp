#include "state.h"

using namespace std;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, State &s) {
  for (auto itm: s.regs.m) {
    os << "Register: " << itm.first;
    os << "Value: " << itm.second << "\n";
  }
  return os;
}

pair<z3::expr, vector<z3::expr>> State::refines(const State &src) {
  // TODO: encode the final memory
  auto [refines, idx] = retValue.refines(src.retValue);
  return {move(refines), {idx}};
}