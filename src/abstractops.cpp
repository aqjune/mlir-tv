#include "abstractops.h"
#include "smt.h"
#include "value.h"
#include <map>

using namespace smt;
using namespace std;

namespace {
// Abstract representation of fp constants.
map<double, Expr> fpconst_absrepr;
unsigned fpconst_absrepr_num;

// TODO: this must be properly set
// What we need to do is to statically find how many 'different' fp values a
// program may observe.
const unsigned FP_BITS = 4;
}

namespace aop {

static AbsLevelDot alDot;
static UsedAbstractOps usedOps;

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstractionLevel(AbsLevelDot ad) {
  alDot = ad;
  memset(&usedOps, 0, sizeof(usedOps));

  fpconst_absrepr.clear();
  fpconst_absrepr_num = 0;
}


Sort fpSort() {
  return Sort::bvSort(FP_BITS);
}

Expr fpConst(double f) {
  // We don't explicitly encode f
  auto itr = fpconst_absrepr.find(f);
  if (itr != fpconst_absrepr.end())
    return itr->second;

  uint64_t absval;
  if (f == 0.0)
    absval = 0; // This is consistent with what mkZeroElemFromArr assumes
  else {
    assert(1 + fpconst_absrepr_num < (1ull << (uint64_t)FP_BITS));
    absval = 1 + fpconst_absrepr_num++;
  }
  Expr e = Expr::mkBV(absval, FP_BITS);
  fpconst_absrepr.emplace(f, e);
  return e;
}

vector<double> fpPossibleConsts(const Expr &e) {
  vector<double> vec;
  for (auto &[k, v]: fpconst_absrepr) {
    if (structurallyEq(v, e))
      vec.push_back(k);
  }
  return vec;
}

Expr mkZeroElemFromArr(const Expr &arr) {
  unsigned bvsz = arr.select(Index::zero()).sort().bitwidth();
  return Expr::mkBV(0, bvsz);
}

Expr fpAdd(const Expr &f1, const Expr &f2) {
  usedOps.add = true;
  auto fty = f1.sort();

  auto addfn = mkUF({fty, fty}, fty, "fp_add");
  return addfn(f1, f2);
}

Expr fpMul(const Expr &a, const Expr &b) {
  usedOps.mul = true;

  // TODO: check that a.get_Sort() == b.get_Sort()
  auto mulfn = mkUF({a.get_Sort(), b.get_Sort()}, Float::Sort(), "fp_mul");
  return mulfn(a, b);
}

Expr sum(const Expr &a, const Expr &n) {
  usedOps.sum = true;
  // TODO: check that a.Sort is Index::Sort() -> Float::Sort()

  auto sumfn = mkUF(a.get_Sort(), Float::Sort(), "smt_sum");
  auto i = Index("idx");
  Expr ai = z3::select(a, i);
  Expr zero = mkZeroElemFromArr(a);
  return sumfn(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
}

Expr dot(const Expr &a, const Expr &b, const Expr &n) {
  if (alDot == AbsLevelDot::FULLY_ABS) {
    usedOps.dot = true;
    // TODO: check that a.get_Sort() == b.get_Sort()
    auto i = Index("idx");
    auto dotfn = mkUF({a.get_Sort(), b.get_Sort()}, Float::Sort(), "smt_dot");

    Expr ai = z3::select(a, i), bi = z3::select(b, i);
    Expr zero = mkZeroElemFromArr(a);
    return dotfn(
        z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)),
        z3::lambda(i, z3::ite(z3::ult(i, n), bi, zero)));
  } else if (alDot == AbsLevelDot::SUM_MUL) {
    usedOps.mul = usedOps.sum = true;
    // TODO: check that a.get_Sort() == b.get_Sort()
    auto i = Index("idx");
    Expr ai = z3::select(a, i), bi = z3::select(b, i);
    return sum(z3::lambda(i, fpMul(ai, bi)), n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

}