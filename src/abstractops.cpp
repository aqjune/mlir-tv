#include "abstractops.h"
#include "simplevalue.h"
#include "smt.h"
#include <map>

using namespace smt;
using namespace std;

namespace {
string freshName(string prefix) {
  static int count = 0;
  return prefix + to_string(count ++);
}

bool useMultiset;
aop::UsedAbstractOps usedOps;

// ----- Constants and global vars for abstract floating point operations ------

aop::AbsLevelFpDot alFpDot;
bool isFpAddAssociative;
optional<aop::AbsFpEncoding> floatEnc;
optional<aop::AbsFpEncoding> doubleEnc;

// ----- Constants and global vars for abstract int operations ------

aop::AbsLevelIntDot alIntDot;
map<unsigned, FnDecl> int_sumfn;
map<unsigned, FnDecl> int_dotfn;

FnDecl getIntSumFn(unsigned bitwidth) {
  auto itr = int_sumfn.find(bitwidth);
  if (itr != int_sumfn.end())
    return itr->second;

  auto codomainty = Sort::bvSort(bitwidth);
  auto arrty = Sort::arraySort(Index::sort(), codomainty).toFnSort();
  FnDecl hashfn(arrty, Integer::sort(bitwidth),
                "int_sum" + to_string(bitwidth));
  int_sumfn.emplace(bitwidth, hashfn);
  return hashfn;
}

FnDecl getIntDotFn(unsigned bitwidth) {
  auto itr = int_dotfn.find(bitwidth);
  if (itr != int_dotfn.end())
    return itr->second;

  auto codomainty = Sort::bvSort(bitwidth);
  auto arrty = Sort::arraySort(Index::sort(), codomainty).toFnSort();
  FnDecl hashfn({arrty, arrty}, codomainty, "int_dot" + to_string(bitwidth));
  int_dotfn.emplace(bitwidth, hashfn);
  return hashfn;
}
}


namespace aop {

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstraction(
    AbsLevelFpDot afd, AbsLevelIntDot aid, bool addAssoc,
    unsigned floatBits, unsigned doubleBits) {
  alFpDot = afd;
  alIntDot = aid;
  isFpAddAssociative = addAssoc;
  memset(&usedOps, 0, sizeof(usedOps));

  if (floatBits > 1)
    floatBits--;
  if (doubleBits > 1)
    doubleBits--;

  floatEnc.emplace(llvm::APFloat::IEEEsingle(), floatBits, "float");
  doubleEnc.emplace(llvm::APFloat::IEEEdouble(), doubleBits, "double");
}

// A set of options that must not change the precision of validation.
void setEncodingOptions(bool use_multiset) {
  useMultiset = use_multiset;
}

bool getFpAddAssociativity() { return isFpAddAssociative; }

AbsFpEncoding &getFloatEncoding() { return *floatEnc; }
AbsFpEncoding &getDoubleEncoding() { return *doubleEnc; }
AbsFpEncoding &getFpEncoding(mlir::Type ty) {
  if (ty.isa<mlir::Float32Type>()) {
    return getFloatEncoding();
  } else if (ty.isa<mlir::Float64Type>()) {
     return getDoubleEncoding();
  }
  llvm_unreachable("Unknown type");
}


AbsFpEncoding::AbsFpEncoding(const llvm::fltSemantics &semantics,
                             unsigned valuebits, string &&fn_suffix)
     :semantics(semantics), fn_suffix(move(fn_suffix)) {
  assert(valuebits > 0);
  value_bv_bits = valuebits;
  fp_bv_bits = SIGN_BITS + TYPE_BITS + value_bv_bits;
  inf_value = 1ull << (uint64_t)value_bv_bits; // The type bit is set to 1
  nan_value = inf_value + 1; // Type bit = 1 && Value bits != 0
  signed_value = 1ull << (uint64_t)(TYPE_BITS + value_bv_bits);

  fpconst_nan = Expr::mkBV(nan_value, fp_bv_bits);
  fpconst_inf_pos = Expr::mkBV(inf_value, fp_bv_bits);
  fpconst_inf_neg = Expr::mkBV(signed_value + inf_value, fp_bv_bits);
  fpconst_zero_pos = Expr::mkBV(0, fp_bv_bits);
  fpconst_zero_neg = Expr::mkBV(signed_value + 0, fp_bv_bits);

  fp_sumfn.reset();
  fp_assoc_sumfn.reset();
  fp_dotfn.reset();
  fp_addfn.reset();
  fp_mulfn.reset();
  fp_sum_relations.clear();
}

FnDecl AbsFpEncoding::getAddFn() {
  if (!fp_addfn) {
    auto fty = sort();
    auto fty2 = Sort::bvSort(fp_bv_bits - SIGN_BITS);
    fp_addfn.emplace({fty, fty}, fty2, "fp_add_" + fn_suffix);
  }
  return *fp_addfn;
}

// TODO: update
FnDecl AbsFpEncoding::getMulFn() {
  if (!fp_mulfn) {
    auto fty = sort();
    fp_mulfn.emplace({fty, fty}, fty, "fp_mul_" + fn_suffix);
  }
  return *fp_mulfn;
}

FnDecl AbsFpEncoding::getAssocSumFn() {
  auto s = Expr::mkEmptyBag(sort()).sort();
  if (!fp_assoc_sumfn)
    fp_assoc_sumfn.emplace(s, sort(), "fp_assoc_sum_" + fn_suffix);
  return *fp_assoc_sumfn;
}

FnDecl AbsFpEncoding::getSumFn() {
  auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
  if (!fp_sumfn)
    fp_sumfn.emplace(arrs, sort(), "fp_sum_" + fn_suffix);
  return *fp_sumfn;
}

FnDecl AbsFpEncoding::getDotFn() {
  auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
  if (!fp_dotfn)
    fp_dotfn.emplace({arrs, arrs}, sort(), "fp_dot_" + fn_suffix);
  return *fp_dotfn;
}

Expr AbsFpEncoding::constant(const llvm::APFloat &f) {
  if (f.isNaN())
    return *fpconst_nan;
  else if (f.isInfinity())
    return f.isNegative() ? *fpconst_inf_neg : *fpconst_inf_pos;
  else if (f.isPosZero())
    return *fpconst_zero_pos;
  else if (f.isNegZero())
    return *fpconst_zero_neg;

  // We don't explicitly encode f
  auto itr = fpconst_absrepr.find(f);
  if (itr != fpconst_absrepr.end())
    return itr->second;

  uint64_t value_id;
  auto abs_f = f;
  abs_f.clearSign();
  if (abs_f.compare(llvm::APFloat(semantics, 1)) == llvm::APFloat::cmpEqual) {
    value_id = 1;
  } else {
    assert(static_cast<uint64_t>(2 + fpconst_absrepr_num) < inf_value);
    value_id = 2 + fpconst_absrepr_num++;
  }

  uint64_t bw = fp_bv_bits;
  Expr e_pos = Expr::mkBV(value_id, bw);
  fpconst_absrepr.emplace(abs_f, e_pos);
  Expr e_neg = Expr::mkBV(signed_value | value_id, bw);
  fpconst_absrepr.emplace(-abs_f, e_neg);

  return f.isNegative() ? e_neg : e_pos;
}

vector<llvm::APFloat> AbsFpEncoding::possibleConsts(const Expr &e) const {
  vector<llvm::APFloat> vec;

  for (auto &[k, v]: fpconst_absrepr) {
    if (v.isIdentical(e))
      vec.push_back(k);
  }

  // for 'reserved' values that do not belong to fpconst_absrepr
  if (fpconst_nan && fpconst_nan->isIdentical(e)) {
    vec.push_back(llvm::APFloat::getNaN(semantics));
  } else if (fpconst_zero_pos && fpconst_zero_pos->isIdentical(e)) {
    vec.push_back(llvm::APFloat::getZero(semantics));
  } else if (fpconst_zero_neg && fpconst_zero_neg->isIdentical(e)) {
    vec.push_back(llvm::APFloat::getZero(semantics, true));
  } else if (fpconst_inf_pos && fpconst_inf_pos->isIdentical(e)) {
    vec.push_back(llvm::APFloat::getInf(semantics));
  } else if (fpconst_inf_neg && fpconst_inf_neg->isIdentical(e)) {
    vec.push_back(llvm::APFloat::getInf(semantics, true));
  }

  return vec;
}

Expr AbsFpEncoding::zero(bool isNegative) {
  return constant(llvm::APFloat::getZero(semantics, isNegative));
}

Expr AbsFpEncoding::one(bool isNegative) {
  return constant(llvm::APFloat(semantics, isNegative ? -1 : 1));
}

Expr AbsFpEncoding::infinity(bool isNegative) {
  return constant(llvm::APFloat::getInf(semantics, isNegative));
}

Expr AbsFpEncoding::nan() {
  return constant(llvm::APFloat::getNaN(semantics));
}

Expr AbsFpEncoding::add(const Expr &_f1, const Expr &_f2) {
  usedOps.fpAdd = true;

  const auto &fp_id = zero(true);
  const auto fp_inf_pos = infinity();
  const auto fp_inf_neg = infinity(true);
  const auto fp_nan = nan();
  const auto bv_true = Expr::mkBV(1, 1);
  const auto bv_false = Expr::mkBV(0, 1);

  // Assume that all unspecified BVs are NaN
  const auto inf_value = ((Expr)fp_inf_pos).extract(value_bv_bits - 1, 0);
  const auto inf_type = bv_true;

  const auto f1_value = _f1.extract(value_bv_bits - 1, 0);
  const auto f1_type = _f1.extract(value_bv_bits, value_bv_bits);
  const auto f1 = Expr::mkIte(
      (f1_type == inf_type) & (f1_value != inf_value), fp_nan, _f1);

  const auto f2_value = _f2.extract(value_bv_bits - 1, 0);
  const auto f2_type = _f2.extract(value_bv_bits, value_bv_bits);
  const auto f2 = Expr::mkIte(
      (f2_type == inf_type) & (f2_value != inf_value), fp_nan, _f2);

  // Encode commutativity
  auto fp_add_res = getAddFn().apply({f1, f2}) + getAddFn().apply({f2, f1});
  auto fp_add_sign = fp_add_res.getMSB();
  auto fp_add_value = fp_add_res.extract(value_bv_bits - 1, 0);

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
    // There are two cases where we must override the sign bit of fp_add.
    // If signbit(f1) == 0 /\ signbit(f2) == 0, signbit(fpAdd(f1, f2)) = 0.
    // If signbit(f1) == 1 /\ signbit(f2) == 1, signbit(fpAdd(f1, f2)) = 1.
    // Otherwise, we can just use the arbitrary sign yielded from fp_add.
    Expr::mkIte(((f1.getMSB() == bv_false) & (f2.getMSB() == bv_false)),
      // pos + pos -> pos
      bv_false.concat(fp_add_value.zext(TYPE_BITS)),
    Expr::mkIte(((f1.getMSB() == bv_true) & (f2.getMSB() == bv_true)),
      // neg + neg -> neg
      bv_true.concat(fp_add_value.zext(TYPE_BITS)),
    Expr::mkIte(f1.extract(value_bv_bits - 1, 0) ==
                f2.extract(value_bv_bits - 1, 0),
      // x + -x -> 0.0
      zero(),
      fp_add_sign.concat(fp_add_value.zext(TYPE_BITS))
  ))))))))));
}

Expr AbsFpEncoding::mul(const Expr &f1, const Expr &f2) {
  usedOps.fpMul = true;

  auto fp_id = one();
  // if neither a nor b is 1.0, the result should be
  // an abstract and pairwise commutative value.
  // therefore we return fp_mul(f1, f2) + fp_mul(f2, f1)
  return Expr::mkIte(f1 == fp_id, f2,             // if f1 == 1.0, then f2
    Expr::mkIte(f2 == fp_id, f1,                  // elif f2 == 1.0 , then f1
      getMulFn().apply({f1, f2}) + getMulFn().apply({f2, f1})
          // else fp_mul(f1, f2) + fp_mul(f2, f1)
    )
  );
}

Expr AbsFpEncoding::multisetSum(const Expr &a, const Expr &n) {
  uint64_t length;
  if (!n.isUInt(length))
    assert("Only an array of constant length is supported.");

  auto elemtSort = a.select(Index(0)).sort();
  auto bag = Expr::mkEmptyBag(elemtSort);
  for (unsigned i = 0; i < length; i ++) {
    bag = bag.insert(a.select(Index(i)));
    bag = bag.simplify();
  }

  Expr result = getAssocSumFn()(bag);

  if (n.isNumeral())
    fp_sum_relations.push_back({bag, n, result});

  return result;
}

Expr AbsFpEncoding::sum(const Expr &a, const Expr &n) {
  usedOps.fpSum = true;

  if (getFpAddAssociativity() && useMultiset)
    return multisetSum(a, n);

  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr result = getSumFn()(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n), ai, zero(true))));

  if (getFpAddAssociativity() && n.isNumeral())
    fp_sum_relations.push_back({a, n, result});

  return result;
}

Expr AbsFpEncoding::dot(const Expr &a, const Expr &b, const Expr &n) {
  if (alFpDot == AbsLevelFpDot::FULLY_ABS) {
    usedOps.fpDot = true;
    auto i = (Expr)Index::var("idx", VarType::BOUND);

    Expr ai = a.select(i), bi = b.select(i);
    Expr identity = zero(true);
    // Encode commutativity: dot(a, b) = dot(b, a)
    Expr lhs = getDotFn().apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity))});
    Expr rhs = getDotFn().apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity))});
    return lhs + rhs;

  } else if (alFpDot == AbsLevelFpDot::SUM_MUL) {
    // usedOps.fpMul/fpSum will be updated by the fpMul()/fpSum() calls below
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, mul(ai, bi));

    return sum(arr, n);
  }
  llvm_unreachable("Unknown abstraction level for fp dot");
}

Expr AbsFpEncoding::getFpAssociativePrecondition() const {
  if (useMultiset) {
    // precondition between `bag equality <-> assoc_sumfn`
    Expr precond = Expr::mkBool(true);
    for (unsigned i = 0; i < fp_sum_relations.size(); i ++) {
      for (unsigned j = i + 1; j < fp_sum_relations.size(); j ++) {
        auto [abag, an, asum] = fp_sum_relations[i];
        auto [bbag, bn, bsum] = fp_sum_relations[j];
        uint64_t alen, blen;
        if (!an.isUInt(alen) || !bn.isUInt(blen) || alen != blen) continue;
        precond = precond & (abag == bbag).implies(asum == bsum);
      }
    }
    precond = precond.simplify();
    return precond;
  }

  // precondition between `hashfn <-> sumfn`
  Expr precond = Expr::mkBool(true);
  for (unsigned i = 0; i < fp_sum_relations.size(); i ++) {
    for (unsigned j = i + 1; j < fp_sum_relations.size(); j ++) {
      auto [a, an, asum] = fp_sum_relations[i];
      auto [b, bn, bsum] = fp_sum_relations[j];
      uint64_t alen, blen;
      if (!an.isUInt(alen) || !bn.isUInt(blen) || alen != blen) continue;

      auto domainSort = a.select(Index(0)).sort();
      FnDecl hashfn(domainSort, Index::sort(), freshName("fp_hash"));

      auto aVal = hashfn.apply(a.select(Index(0)));
      for (unsigned k = 1; k < alen; k ++)
        aVal = aVal + hashfn.apply(a.select(Index(k)));
      auto bVal = hashfn.apply(b.select(Index(0)));
      for (unsigned k = 1; k < blen; k ++)
        bVal = bVal + hashfn.apply(b.select(Index(k)));

      // precond: sumfn(A) != sumfn(B) -> hashfn(A) != hashfn(B)
      // This means if two summations are different, we can find concrete hash function that hashes into different value.
      auto associativity = (!(asum == bsum)).implies(!(aVal == bVal));
      precond = precond & associativity;
    }
  }
  precond = precond.simplify();
  return precond;
}



Expr getFpAssociativePrecondition() {
  // Calling this function doesn't make sense if add is not associative
  assert(isFpAddAssociative);

  Expr cond = Expr::mkBool(true);
  if (floatEnc)
    cond &= floatEnc->getFpAssociativePrecondition();
  // TODO: double

  return cond;
}



// ----- Integer operations ------


Expr intSum(const Expr &a, const Expr &n) {
  usedOps.intSum = true;

  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr zero = Integer(0, ai.bitwidth());

  FnDecl sumfn = getIntSumFn(ai.sort().bitwidth());
  Expr result = sumfn(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n), ai, zero)));

  return result;
}

Expr intDot(const Expr &a, const Expr &b, const Expr &n) {
  if (alIntDot == AbsLevelIntDot::FULLY_ABS) {
    usedOps.intDot = true;

    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    assert(ai.sort().bitwidth() == bi.sort().bitwidth());

    FnDecl dotfn = getIntDotFn(ai.sort().bitwidth());
    Expr zero = Expr::mkBV(0, ai.bitwidth());
    Expr lhs = dotfn.apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, zero)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, zero))});
    Expr rhs = dotfn.apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, zero)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, zero))});
    return lhs + rhs;

  } else if (alIntDot == AbsLevelIntDot::SUM_MUL) {
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, ai * bi);

    return intSum(arr, n);
  }
  llvm_unreachable("Unknown abstraction level for int dot");
}

}
