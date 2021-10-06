#include "abstractops.h"
#include "smt.h"
#include "value.h"
#include <cmath>
#include <map>

using namespace smt;
using namespace std;

static string freshName(string prefix) {
  static int count = 0;
  return prefix + to_string(count ++);
}

namespace {
// Keep 'special' fp constants in separate variables
optional<Expr> fpconst_zero_pos;
optional<Expr> fpconst_zero_neg;
optional<Expr> fpconst_nan;
optional<Expr> fpconst_inf_pos;
optional<Expr> fpconst_inf_neg;
// Abstract representation of valid fp constants.
map<float, Expr> fpconst_absrepr;
unsigned fpconst_absrepr_num;

const unsigned SIGN_BITS = 1;
const unsigned TYPE_BITS = 1;
// TODO: this must be properly set
// What we need to do is to statically find how many 'different' fp values a
// program may observe.
// FP_BITS must be geq than 1 (otherwise it can't handle reserved values)
const unsigned VALUE_BITS = 31;
const unsigned FP_BITS = SIGN_BITS + TYPE_BITS + VALUE_BITS;
const uint64_t INF_VALUE = 1ull << (uint64_t)VALUE_BITS;
const uint64_t NAN_VALUE = INF_VALUE + 1;
const uint64_t SIGNED_VALUE = 1ull << (uint64_t)(TYPE_BITS + VALUE_BITS);
}

namespace aop {

static AbsLevelDot alDot;
static bool isAddAssociative;
static UsedAbstractOps usedOps;
static vector<tuple<Expr, Expr, Expr>> staticArrays;

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstractionLevel(AbsLevelDot ad, bool addAssoc) {
  alDot = ad;
  isAddAssociative = addAssoc;
  memset(&usedOps, 0, sizeof(usedOps));

  fpconst_absrepr.clear();
  fpconst_absrepr_num = 0;

  staticArrays.clear();
}

bool getAddAssociativity() { return isAddAssociative; }
AbsLevelDot getDotAbstractionLevel() { return alDot; }


Sort fpSort() {
  return Sort::bvSort(FP_BITS);
}

Expr fpConst(float f) {
  // NaNs, Infs, and +-0 are stored in separate variable
  // as they do not work well with map
  // due to comparison issue
  if (isnan(f)) {
    if (!fpconst_nan)
      fpconst_nan = Expr::mkBV(NAN_VALUE, FP_BITS);
    return *fpconst_nan;
  }

  if (isinf(f)) {
    if (!fpconst_inf_pos)
        fpconst_inf_pos = Expr::mkBV(INF_VALUE, FP_BITS);
    if (!fpconst_inf_neg)
        fpconst_inf_neg = Expr::mkBV(SIGNED_VALUE + INF_VALUE, FP_BITS);
    
    return signbit(f) ? *fpconst_inf_neg : *fpconst_inf_pos;
  }

  if (f == 0.0f) {
    if (!fpconst_zero_pos)
        fpconst_zero_pos = Expr::mkBV(0, FP_BITS);
    if (!fpconst_zero_neg)
        fpconst_zero_neg = Expr::mkBV(SIGNED_VALUE + 0, FP_BITS);

    return signbit(f) ? *fpconst_zero_neg : *fpconst_zero_pos;
  }

  // We don't explicitly encode f
  auto itr = fpconst_absrepr.find(f);
  if (itr != fpconst_absrepr.end())
    return itr->second;

  uint64_t absval;
  float abs_f = abs(f);
  if (abs_f == 1.0f) {
    absval = 1;
  } else {
    assert(static_cast<uint64_t>(2 + fpconst_absrepr_num) < INF_VALUE);
    absval = 2 + fpconst_absrepr_num++;
  }

  Expr e_pos = Expr::mkBV(absval, FP_BITS);
  fpconst_absrepr.emplace(abs_f, e_pos);
  Expr e_neg = Expr::mkBV(SIGNED_VALUE + absval, FP_BITS);
  fpconst_absrepr.emplace(-abs_f, e_neg);
  
  return signbit(f) ? e_neg : e_pos;
}

vector<float> fpPossibleConsts(const Expr &e) {
  vector<float> vec;
  for (auto &[k, v]: fpconst_absrepr) {
    if (v.isIdentical(e))
      vec.push_back(k);
  }

  // for 'reserved' values that do not belong to fpconst_absrepr
  if (fpconst_nan && fpconst_nan->isIdentical(e)) {
    vec.push_back(nanf("0"));
  } else if (fpconst_zero_pos && fpconst_zero_pos->isIdentical(e)) {
    vec.push_back(0.0f);
  } else if (fpconst_zero_neg && fpconst_zero_neg->isIdentical(e)) {
    vec.push_back(-0.0f);
  } else if (fpconst_inf_pos && fpconst_inf_pos->isIdentical(e)) {
    vec.push_back(INFINITY);
  } else if (fpconst_inf_neg && fpconst_inf_neg->isIdentical(e)) {
    vec.push_back(-INFINITY);
  }

  return vec;
}

Expr mkZeroElemFromArr(const Expr &arr) {
  unsigned bvsz = arr.select(Index::zero()).sort().bitwidth();
  return Expr::mkBV(0, bvsz);
}

optional<FnDecl> sumfn, assoc_sumfn, dotfn, fpaddfn, fpmulfn, fpaddufn;

Expr fpAdd(const Expr &f1, const Expr &f2) {
  usedOps.add = true;
  auto fty = f1.sort();

  if (!fpaddfn) {
    // Fully abstract fp_add(fty, fty) -> fty
    // may be interpreted into 'invalid' value.
    // So fp_add should yield BV[SIGN_BITS + VALUE_BITS].
    // Then, an appropriate value will be inserted to fill in TYPE_BITS 
    auto fp_value_ty = Sort::bvSort(SIGN_BITS + VALUE_BITS);
    fpaddfn.emplace({fty, fty}, fp_value_ty, "fp_add");
  }

  auto fp_zero = Float(0.0f);
  auto fp_id = Float(-0.0f);
  auto fp_inf_pos = Float(INFINITY);
  auto fp_inf_neg = Float(-INFINITY);
  auto fp_nan = Float(nanf("0"));
  auto bv_true = Expr::mkBV(1, 1);
  auto bv_false = Expr::mkBV(0, 1);

  auto fp_add_res = fpaddfn->apply({f1, f2}) + fpaddfn->apply({f2, f1});
  auto fp_add_sign = fp_add_res.getMSB();
  auto fp_add_value = fp_add_res.extract(VALUE_BITS - 1, 0);

  return Expr::mkIte(f1 == fp_id, f2,         // -0.0 + x -> x
    Expr::mkIte(f2 == fp_id, f1,              // x + -0.0 -> x
      Expr::mkIte(f1 == fp_nan, f1,           // NaN + x -> NaN
        Expr::mkIte(f2 == fp_nan, f2,         // x + NaN -> NaN
    // inf + -inf -> NaN, -inf + inf -> NaN
    // IEEE 754-2019 section 7.2 'Invalid operation'
    Expr::mkIte(((f1 == fp_inf_pos) & (f2 == fp_inf_neg)) |
                ((f1 == fp_inf_neg) & (f2 == fp_inf_pos)), fp_nan,
    // inf + x -> inf, -inf + x -> -inf (both commutative)
    // IEEE 754-2019 section 6.1 'Infinity arithmetic'
    Expr::mkIte((f1 == fp_inf_pos) | (f1 == fp_inf_neg), f1,
      Expr::mkIte((f2 == fp_inf_pos) | (f2 == fp_inf_neg), f2,
    // If both operands do not fall into any of the cases above,
    // use fp_add for abstract representation.
    // But fp_add only yields BV[SIGN_BITS + VALUE_BIT], so we must insert
    // type bit(s) into the fp_add result.
    // Fortunately we can just assume that type bit(s) is 0,
    // because the result of fp_add is always some finite value
    // as Infs and NaNs are already handled in the previous Ites.
    // 
    // fp_add will yield some arbitrary sign bit when called.
    // But then there are some cases where we must override this sign bit.
    // If signbit(f1) == 0 and signbit(f2) == 0, signbit(fpAdd(f1, f2)) must be 0.
    // And if signbit(f1) == 1 and signbit(f2) == 1, signbit(fpAdd(f1, f2)) must be 1.
    // But if f1 and f2 have different signs, we can just use an arbitrary sign
    // yielded from fp_add
    Expr::mkIte(((f1.getMSB() == bv_false) & (f2.getMSB() == bv_false)),
      // pos + pos -> pos
      bv_false.concat(fp_add_value.zext(TYPE_BITS)),
      Expr::mkIte(((f1.getMSB() == bv_true) & (f2.getMSB() == bv_true)),
        // neg + neg -> neg
        bv_true.concat(fp_add_value.zext(TYPE_BITS)),
        Expr::mkIte(f1.extract(VALUE_BITS - 1, 0) == f2.extract(VALUE_BITS - 1, 0),
          // x + -x -> 0.0
          fp_zero,
          fp_add_sign.concat(fp_add_value.zext(TYPE_BITS))
  ))))))))));
}

Expr fpMul(const Expr &f1, const Expr &f2) {
  usedOps.mul = true;
  // TODO: check that a.get_Sort() == b.get_Sort()
  auto exprSort = f1.sort();

  if (!fpmulfn)
    fpmulfn.emplace({exprSort, exprSort}, Float::sort(), "fp_mul");

  auto fp_id = Float(1.0);
  // if neither a nor b is 1.0, the result should be
  // an abstract and pairwise commutative value.
  // therefore we return fp_mul(f1, f2) + fp_mul(f2, f1)
  return Expr::mkIte(f1 == fp_id, f2,                     // if f1 == 1.0, then f2
    Expr::mkIte(f2 == fp_id, f1,                          // elif f2 == 1.0 , then f1
      fpmulfn->apply({f1, f2}) + fpmulfn->apply({f2, f1}) // else fp_mul(f1, f2) + fp_mul(f2, f1)
    )
  );
}

Expr sum(const Expr &a, const Expr &n) {
  usedOps.sum = true;
  // TODO: check that a.Sort is Index::Sort() -> Float::Sort()

  if (!sumfn)
    sumfn.emplace(a.sort(), Float::sort(), "smt_sum");
  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr zero = mkZeroElemFromArr(a);
  Expr result = (*sumfn)(Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n), ai, zero)));

  if (n.isNumeral())
    staticArrays.push_back({a, n, result});

  return result;
}

Expr dot(const Expr &a, const Expr &b, const Expr &n) {
  if (alDot == AbsLevelDot::FULLY_ABS) {
    usedOps.dot = true;
    // TODO: check that a.get_Sort() == b.get_Sort()
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    auto fnSort = a.sort().toFnSort();
    if (!dotfn)
      dotfn.emplace({fnSort, fnSort}, Float::sort(), "smt_dot");

    Expr ai = a.select(i), bi = b.select(i);
    Expr zero = mkZeroElemFromArr(a);
    Expr lhs = dotfn->apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, zero)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, zero))});
    Expr rhs = dotfn->apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, zero)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, zero))});
    return lhs + rhs;
  } else if (alDot == AbsLevelDot::SUM_MUL) {
    usedOps.mul = usedOps.sum = true;
    // TODO: check that a.get_Sort() == b.get_Sort()
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, fpMul(ai, bi));

    return sum(arr, n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

Expr getAssociativePrecondition() {
  Expr precond = Expr::mkBool(true);
  for (unsigned i = 0; i < staticArrays.size(); i ++) {
    for (unsigned j = i + 1; j < staticArrays.size(); j ++) {
      auto [a, an, asum] = staticArrays[i];
      auto [b, bn, bsum] = staticArrays[j];
      uint64_t alen, blen;
      if (!an.isUInt(alen) || !bn.isUInt(blen) || alen != blen) continue;
      FnDecl hashfn(Float::sort(), Index::sort(), freshName("hash"));

      auto aVal = hashfn.apply(a.select(Index(0)));
      for (unsigned i = 1; i < alen; i ++)
        aVal = aVal + hashfn.apply(a.select(Index(i)));
      auto bVal = hashfn.apply(b.select(Index(0)));
      for (unsigned i = 1; i < blen; i ++)
        bVal = bVal + hashfn.apply(b.select(Index(i)));

      // precond: sumfn(A) != sumfn(B) -> hashfn(A) != hashfn(B)
      // This means if two summations are different, we can find concrete hash function that hashes into different value.
      auto associativity = (!(asum == bsum)).implies(!(aVal == bVal));
      precond = precond & associativity;
    }
  }
  precond = precond.simplify();
  return precond;
}

}
