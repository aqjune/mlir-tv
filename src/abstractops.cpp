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
map<float, Expr> fpconst_absrepr;
unsigned fpconst_absrepr_num;

// TODO: this must be properly set
// What we need to do is to statically find how many 'different' fp values a
// program may observe.
const unsigned FP_BITS = 4;
}

namespace aop {

static AbsLevelDot alDot;
static bool isAddAssociative;
static bool useMultiset;
static UsedAbstractOps usedOps;
static vector<tuple<Expr, Expr, Expr>> staticArrays;

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstractionLevel(AbsLevelDot ad, bool addAssoc, bool multiset) {
  alDot = ad;
  isAddAssociative = addAssoc;
  useMultiset = multiset;
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

vector<float> fpPossibleConsts(const Expr &e) {
  vector<float> vec;
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

Expr associativeSum(const Expr &a, const Expr &n) {
  uint64_t length;
  if (!n.isUInt(length))
    assert("Only an array of constant length is supported.");
  auto bag = Expr::mkEmptyBag(Float::sort());
  for (unsigned i = 0; i < length; i ++) {
    bag = bag.insert(a.select(Index(i)));
    bag = bag.simplify();
  }

  if (!assoc_sumfn)
    assoc_sumfn.emplace(bag.sort(), Float::sort(), "smt_assoc_sum");
  Expr result = (*assoc_sumfn)(bag);

  if (n.isNumeral())
    staticArrays.push_back({bag, n, result});

  return result;
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

    if (isAddAssociative && useMultiset)
      return associativeSum(arr, n);
    else
      return sum(arr, n);
  }
  llvm_unreachable("Unknown abstraction level for dot");
}

Expr getAssociativePrecondition() {
  if (useMultiset) {
    // precondition between `bag equality <-> assoc_sumfn`
    Expr precond = Expr::mkBool(true);
    for (unsigned i = 0; i < staticArrays.size(); i ++) {
      for (unsigned j = i + 1; j < staticArrays.size(); j ++) {
        auto [abag, an, asum] = staticArrays[i];
        auto [bbag, bn, bsum] = staticArrays[j];
        uint64_t alen, blen;
        if (!an.isUInt(alen) || !bn.isUInt(blen) || alen != blen) continue;
        precond = precond & (abag == bbag).implies(asum == bsum);
      }
    }
    precond = precond.simplify();
    return precond;
  } else {
    // precondition between `hashfn <-> sumfn`
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

}
