#include "smt.h"

z3::context ctx;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e) {
  std::stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}