#include "abstractops.h"
#include "smt.h"
#include "value.h"

namespace aop {

static AbsLevelDot alDot;

void setAbstractionLevel(AbsLevelDot ad) {
  alDot = ad;
}


z3::expr mkZeroElemFromArr(const z3::expr &arr) {
  unsigned bvsz = z3::select(arr, Index::zero()).get_sort().bv_size();
  return ctx.bv_val(0, bvsz);
}

z3::expr fp_add(const z3::expr &f1, const z3::expr &f2) {
  auto fty = f1.get_sort();

  z3::sort_vector domain(ctx);
  domain.push_back(fty);
  domain.push_back(fty);
  auto addfn = ctx.function("fp_add", domain, fty);
  return addfn(f1, f2);
}

z3::expr fp_mul(const z3::expr &a, const z3::expr &b) {
  // TODO: check that a.get_sort() == b.get_sort()
  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  domain.push_back(b.get_sort());
  auto mulfn = ctx.function("fp_mul", domain, Float::sort());
  return mulfn(a, b);
}

z3::expr sum(const z3::expr &a, const z3::expr &n) {
  // TODO: check that a.sort is Index::sort() -> Float::sort()

  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  auto sumfn = ctx.function("smt_sum", domain, Float::sort());

  auto i = Index("idx");
  z3::expr ai = z3::select(a, i);
  z3::expr zero = mkZeroElemFromArr(a);
  return sumfn(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
}

z3::expr dot(const z3::expr &a, const z3::expr &b, const z3::expr &n) {
  if (alDot == FULLY_ABS) {
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");

    z3::sort_vector domain(ctx);
    domain.push_back(a.get_sort());
    domain.push_back(b.get_sort());
    auto dotfn = ctx.function("smt_dot", domain, Float::sort());

    z3::expr_vector args(ctx);
    z3::expr ai = z3::select(a, i), bi = z3::select(b, i);
    z3::expr zero = mkZeroElemFromArr(a);
    args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
    args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), bi, zero)));
    return dotfn(args);
  } else if (alDot == SUM_MUL) {
    // TODO: check that a.get_sort() == b.get_sort()
    auto i = Index("idx");
    z3::expr ai = z3::select(a, i), bi = z3::select(b, i);
    return sum(z3::lambda(i, fp_mul(ai, bi)), n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

}