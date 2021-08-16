#include "abstractops.h"
#include "smt.h"
#include "value.h"
#include <map>

using namespace smt;
using namespace std;

namespace {
map<double, expr> const_vars;
map<double, expr> const_vars_evaluated;

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
  const_vars.clear();
}


smt::sort fpSort() {
  return bvSort(FP_BITS);
}

expr fpConst(double f) {
  // We don't explicitly encode f
  auto itr = const_vars.find(f);
  if (itr == const_vars.end()) {
    optional<expr> e;
    if (f == 0.0)
      e = ctx.bv_val(0, FP_BITS);
    else
      e = mkFreshVar(fpSort(), "#float_const");

    const_vars.emplace(f, *e);
    return *e;
  }
  return itr->second;
}

void fpEvalConstVars(smt::model mdl) {
  for (auto &[k, v]: const_vars)
    const_vars_evaluated.emplace(k, mdl.eval(v));
}

vector<double> fpPossibleConsts(const expr &e) {
  vector<double> vec;
  for (auto &[k, v]: const_vars_evaluated) {
    if (structurallyEq(v, e))
      vec.push_back(k);
  }
  return vec;
}

expr mkZeroElemFromArr(const expr &arr) {
  unsigned bvsz = z3::select(arr, Index::zero()).get_sort().bv_size();
  return ctx.bv_val(0, bvsz);
}

expr fpAdd(const expr &f1, const expr &f2) {
  usedOps.add = true;
  auto fty = f1.get_sort();

  auto addfn = mkUF({fty, fty}, fty, "fp_add");
  return addfn(f1, f2);
}

expr fpMul(const expr &a, const expr &b) {
  usedOps.mul = true;

  // TODO: check that a.get_sort() == b.get_sort()
  auto mulfn = mkUF({a.get_sort(), b.get_sort()}, Float::sort(), "fp_mul");
  return mulfn(a, b);
}

expr sum(const expr &a, const expr &n) {
  usedOps.sum = true;
  // TODO: check that a.sort is Index::sort() -> Float::sort()

  auto sumfn = mkUF(a.get_sort(), Float::sort(), "smt_sum");
  auto i = Index("idx");
  expr ai = z3::select(a, i);
  expr zero = mkZeroElemFromArr(a);
  return sumfn(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
}

expr dot(const expr &a, const expr &b, const expr &n) {
  if (alDot == AbsLevelDot::FULLY_ABS) {
    usedOps.dot = true;
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");
    auto dotfn = mkUF({a.get_sort(), b.get_sort()}, Float::sort(), "smt_dot");

    expr ai = z3::select(a, i), bi = z3::select(b, i);
    expr zero = mkZeroElemFromArr(a);
    return dotfn(
        z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)),
        z3::lambda(i, z3::ite(z3::ult(i, n), bi, zero)));
  } else if (alDot == AbsLevelDot::SUM_MUL) {
    usedOps.mul = usedOps.sum = true;
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");
    expr ai = z3::select(a, i), bi = z3::select(b, i);
    return sum(z3::lambda(i, fpMul(ai, bi)), n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

}