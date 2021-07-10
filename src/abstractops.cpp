#include "abstractops.h"
#include "smt.h"
#include "value.h"

namespace aop {

z3::expr mkZeroElemFromArr(const z3::expr &arr) {
  unsigned bvsz = z3::select(arr, Index::zero()).get_sort().bv_size();
  return ctx.bv_val(0, bvsz);
}

z3::expr mul(const z3::expr &a, const z3::expr &b) {
  // TODO: check that a.get_sort() == b.get_sort()
  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  domain.push_back(b.get_sort());
  auto mulfn = ctx.function("mul", domain, Float::sort());
  return mulfn(a, b);
}

z3::expr dot(const z3::expr &a, const z3::expr &b, const z3::expr &n) {
  // TODO: check that a.get_sort() == b.get_sort()
  auto i = Index("idx");

  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  auto raddfn = ctx.function("reduce_add", domain, Float::sort());

  z3::expr_vector args(ctx);
  z3::expr ai = z3::select(a, i), bi = z3::select(b, i);
  z3::expr zero = mkZeroElemFromArr(a);
  return raddfn(z3::lambda(i, z3::ite(z3::ult(i, n), mul(ai, bi), zero)));
}

// dot2 is even more abstract than dot
z3::expr dot2(const z3::expr &a, const z3::expr &b, const z3::expr &n) {
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
}

}