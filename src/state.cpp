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

void RegFile::add(mlir::Value v, const z3::expr &e, mlir::Type ty) {
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

State::State(): isWellDefined(ctx) {
  m = Memory::create(10, MemType::MULTIPLE);
}
