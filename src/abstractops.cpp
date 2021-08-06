#include "abstractops.h"
#include "smt.h"
#include "value.h"

using namespace smt;

namespace aop {

static AbsLevelDot alDot;
static UsedAbstractOps usedOps;

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstractionLevel(AbsLevelDot ad) {
  alDot = ad;
  memset(&usedOps, 0, sizeof(usedOps));
}


expr mkZeroElemFromArr(const expr &arr) {
  unsigned bvsz = z3::select(arr, Index::zero()).get_sort().bv_size();
  return ctx.bv_val(0, bvsz);
}

expr fp_add(const expr &f1, const expr &f2) {
  usedOps.add = true;
  auto fty = f1.get_sort();

  z3::sort_vector domain(ctx);
  domain.push_back(fty);
  domain.push_back(fty);
  auto addfn = ctx.function("fp_add", domain, fty);
  return addfn(f1, f2);
}

expr fp_mul(const expr &a, const expr &b) {
  usedOps.mul = true;

  // TODO: check that a.get_sort() == b.get_sort()
  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  domain.push_back(b.get_sort());
  auto mulfn = ctx.function("fp_mul", domain, Float::sort());
  return mulfn(a, b);
}

expr sum(const expr &a, const expr &n) {
  usedOps.sum = true;
  // TODO: check that a.sort is Index::sort() -> Float::sort()

  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  auto sumfn = ctx.function("smt_sum", domain, Float::sort());

  auto i = Index("idx");
  expr ai = z3::select(a, i);
  expr zero = mkZeroElemFromArr(a);
  return sumfn(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
}

expr dot(const expr &a, const expr &b, const expr &n) {
  if (alDot == FULLY_ABS) {
    usedOps.dot = true;
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");

    z3::sort_vector domain(ctx);
    domain.push_back(a.get_sort());
    domain.push_back(b.get_sort());
    auto dotfn = ctx.function("smt_dot", domain, Float::sort());

    z3::expr_vector args(ctx);
    expr ai = z3::select(a, i), bi = z3::select(b, i);
    expr zero = mkZeroElemFromArr(a);
    args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
    args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), bi, zero)));
    return dotfn(args);
  } else if (alDot == SUM_MUL) {
    usedOps.mul = usedOps.sum = true;
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");
    expr ai = z3::select(a, i), bi = z3::select(b, i);
    return sum(z3::lambda(i, fp_mul(ai, bi)), n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

}