#include "abstractops.h"
#include "smt.h"
#include "value.h"
#include <map>

using namespace smt;
using namespace std;

static string freshName(string prefix) {
  static int count = 0;
  return prefix + to_string(count ++);
}

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

Expr fpConst(double f) {
  // We don't explicitly encode f
  auto itr = fpconst_absrepr.find(f);
  if (itr != fpconst_absrepr.end())
    return itr->second;

  uint64_t absval;
  if (f == 0.0) {
    absval = 0; // This is consistent with what mkZeroElemFromArr assumes
  } else if (f == 1.0) {
    absval = 1;
  } else {
    assert(2 + fpconst_absrepr_num < (1ull << (uint64_t)FP_BITS));
    absval = 2 + fpconst_absrepr_num++;
  }
  Expr e = Expr::mkBV(absval, FP_BITS);
  fpconst_absrepr.emplace(f, e);
  return e;
}

vector<double> fpPossibleConsts(const Expr &e) {
  vector<double> vec;
  for (auto &[k, v]: fpconst_absrepr) {
    if (v.isIdentical(e))
      vec.push_back(k);
  }
  return vec;
}

Expr mkZeroElemFromArr(const Expr &arr) {
  unsigned bvsz = arr.select(Index::zero()).sort().bitwidth();
  return Expr::mkBV(0, bvsz);
}

optional<FnDecl> sumfn, assoc_sumfn, dotfn, fpaddfn, fpmulfn;

Expr fpAdd(const Expr &f1, const Expr &f2) {
  usedOps.add = true;
  auto fty = f1.sort();

  if (!fpaddfn)
    fpaddfn.emplace({fty, fty}, fty, "fp_add");
  return fpaddfn->apply({f1, f2});
}

Expr fpMul(const Expr &a, const Expr &b) {
  usedOps.mul = true;
  // TODO: check that a.get_Sort() == b.get_Sort()
  auto exprSort = a.sort();

  if (!fpmulfn)
    fpmulfn.emplace({exprSort, exprSort}, Float::sort(), "fp_mul");

  auto fp_id = Float(1.0);
  // if a or b is 1.0, the return value need not to be
  // an abstract and pairwise commutative value.
  // therefore it returns b or a, not b+b or a+a.
  return Expr::mkIte(a == fp_id, b,                   // if a == 1.0, then
    Expr::mkIte(b == fp_id, a,                        // elif b == 1.0 , then
      fpmulfn->apply({a, b}) + fpmulfn->apply({b, a}) // else
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
