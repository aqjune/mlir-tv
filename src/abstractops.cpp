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

// ----- Constants and global vars for abstract floating point operations ------

namespace {
template<class OriginalTy>
class AbsFpValEncoding {
public:
  // NaNs, Infs, and +-0 are stored in separate variable
  // as they do not work well with map due to comparison issue
  optional<Expr> fpconst_zero_pos;
  optional<Expr> fpconst_zero_neg;
  optional<Expr> fpconst_nan;
  optional<Expr> fpconst_inf_pos;
  optional<Expr> fpconst_inf_neg;
  // Abstract representation of valid fp constants.
  map<OriginalTy, Expr> fpconst_absrepr;
  uint64_t fpconst_absrepr_num = 0;

  const static unsigned SIGN_BITS = 1;
  const static unsigned TYPE_BITS = 1;

  unsigned value_bv_bits;
  unsigned fp_bv_bits;
  uint64_t inf_value;
  uint64_t nan_value;
  uint64_t signed_value;

  vector<tuple<Expr, Expr, Expr>> fp_sum_relations;

private:
  // These are lazily created.
  optional<FnDecl> fp_sumfn;
  optional<FnDecl> fp_assoc_sumfn;
  optional<FnDecl> fp_dotfn;
  optional<FnDecl> fp_addfn;
  optional<FnDecl> fp_mulfn;
  std::string fn_suffix;

public:
  AbsFpValEncoding(unsigned valuebits, std::string &&fn_suffix):
      fn_suffix(move(fn_suffix)) {
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

  Sort sort() const {
    return Sort::bvSort(fp_bv_bits);
  }

  // Returns a fully abstract add fn fp_add(fty, fty) -> fty2
  // where fty is BV(fp_bv_bits) and fty2 is BV(fp_bv_bits - TYPE_BITS).
  // It is the user of this function that fills in TYPE_BITS.
  FnDecl getAddFn() {
    if (!fp_addfn) {
      auto fty = sort();
      auto fty2 = Sort::bvSort(fp_bv_bits - SIGN_BITS);
      fp_addfn.emplace({fty, fty}, fty2, "fp_add_" + fn_suffix);
    }
    return *fp_addfn;
  }

  // Returns a fully abstract add fn fp_add(fty, fty) -> fty2
  // where fty is BV(fp_bv_bits) and fty2 is BV(value_bv_bits).
  // It is the user of this function that fills in TYPE_BITS and SIGN_BITS.
  FnDecl getMulFn() {
    if (!fp_mulfn) {
      auto fty = sort();
      auto fty2 = Sort::bvSort(value_bv_bits);
      fp_mulfn.emplace({fty, fty}, fty2, "fp_mul_" + fn_suffix);
    }
    return *fp_mulfn;
  }

  FnDecl getAssocSumFn() {
    auto s = Expr::mkEmptyBag(sort()).sort();
    if (!fp_assoc_sumfn)
      fp_assoc_sumfn.emplace(s, sort(), "fp_assoc_sum_" + fn_suffix);
    return *fp_assoc_sumfn;
  }

  FnDecl getSumFn() {
    auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
    if (!fp_sumfn)
      fp_sumfn.emplace(arrs, sort(), "fp_sum_" + fn_suffix);
    return *fp_sumfn;
  }

  FnDecl getDotFn() {
    auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
    if (!fp_dotfn)
      fp_dotfn.emplace({arrs, arrs}, sort(), "fp_dot_" + fn_suffix);
    return *fp_dotfn;
  }
};

// TODO: double
optional<AbsFpValEncoding<float>> floatEnc;
}

// ----- Constants and global vars for abstract int operations ------

namespace {
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

static AbsLevelFpDot alFpDot;
static AbsLevelIntDot alIntDot;
static bool isFpAddAssociative;
static UsedAbstractOps usedOps;
static bool useMultiset;


UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstraction(
    AbsLevelFpDot afd, AbsLevelIntDot aid, bool addAssoc, unsigned fpBits) {
  alFpDot = afd;
  alIntDot = aid;
  isFpAddAssociative = addAssoc;
  memset(&usedOps, 0, sizeof(usedOps));

  floatEnc.emplace(fpBits == 1 ? fpBits : fpBits - 1, "float");
}

// A set of options that must not change the precision of validation.
void setEncodingOptions(bool use_multiset) {
  useMultiset = use_multiset;
}

bool getFpAddAssociativity() { return isFpAddAssociative; }


Sort fpSort() {
  return floatEnc->sort();
}

Expr fpConst(float f) {
  auto &enc = *floatEnc;

  if (isnan(f))
    return *enc.fpconst_nan;

  if (isinf(f))
    return signbit(f) ? *enc.fpconst_inf_neg : *enc.fpconst_inf_pos;

  if (f == 0.0f)
    return signbit(f) ? *enc.fpconst_zero_neg : *enc.fpconst_zero_pos;

  // We don't explicitly encode f
  auto itr = enc.fpconst_absrepr.find(f);
  if (itr != enc.fpconst_absrepr.end())
    return itr->second;

  uint64_t value_id;
  float abs_f = abs(f);
  if (abs_f == 1.0f) {
    value_id = 1;
  } else {
    assert(static_cast<uint64_t>(2 + enc.fpconst_absrepr_num) < enc.inf_value);
    value_id = 2 + enc.fpconst_absrepr_num++;
  }

  uint64_t bw = enc.fp_bv_bits;
  Expr e_pos = Expr::mkBV(value_id, bw);
  enc.fpconst_absrepr.emplace(abs_f, e_pos);
  Expr e_neg = Expr::mkBV(enc.signed_value | value_id, bw);
  enc.fpconst_absrepr.emplace(-abs_f, e_neg);

  return signbit(f) ? e_neg : e_pos;
}

vector<float> fpPossibleConsts(const Expr &e) {
  vector<float> vec;
  auto &enc = *floatEnc;

  for (auto &[k, v]: enc.fpconst_absrepr) {
    if (v.isIdentical(e))
      vec.push_back(k);
  }

  // for 'reserved' values that do not belong to fpconst_absrepr
  if (enc.fpconst_nan && enc.fpconst_nan->isIdentical(e)) {
    vec.push_back(nanf("0"));
  } else if (enc.fpconst_zero_pos && enc.fpconst_zero_pos->isIdentical(e)) {
    vec.push_back(0.0f);
  } else if (enc.fpconst_zero_neg && enc.fpconst_zero_neg->isIdentical(e)) {
    vec.push_back(-0.0f);
  } else if (enc.fpconst_inf_pos && enc.fpconst_inf_pos->isIdentical(e)) {
    vec.push_back(INFINITY);
  } else if (enc.fpconst_inf_neg && enc.fpconst_inf_neg->isIdentical(e)) {
    vec.push_back(-INFINITY);
  }

  return vec;
}

Expr fpAdd(const Expr &_f1, const Expr &_f2) {
  usedOps.fpAdd = true;
  auto &enc = *floatEnc;

  const auto fp_zero = Float(0.0f);
  const auto fp_id = Float(-0.0f);
  const auto fp_inf_pos = Float(INFINITY);
  const auto fp_inf_neg = Float(-INFINITY);
  const auto fp_nan = Float(nanf("0"));
  const auto bv_true = Expr::mkBV(1, 1);
  const auto bv_false = Expr::mkBV(0, 1);

  // Handle non-canonical NaNs
  const auto inf_value = ((Expr)fp_inf_pos).extract(enc.value_bv_bits - 1, 0);
  const auto inf_type = bv_true; 

  const auto f1_value = _f1.extract(enc.value_bv_bits - 1, 0);
  const auto f1_type = _f1.extract(enc.value_bv_bits, enc.value_bv_bits);
  const auto f1 = Expr::mkIte((f1_type == inf_type) & (f1_value != inf_value), fp_nan, _f1);

  const auto f2_value = _f2.extract(enc.value_bv_bits - 1, 0);
  const auto f2_type = _f2.extract(enc.value_bv_bits, enc.value_bv_bits);
  const auto f2 = Expr::mkIte((f2_type == inf_type) & (f2_value != inf_value), fp_nan, _f2);

  // Encode commutativity
  auto fp_add_res = enc.getAddFn().apply({f1, f2}) +
                    enc.getAddFn().apply({f2, f1});
  auto fp_add_sign = fp_add_res.getMSB();
  auto fp_add_value = fp_add_res.extract(enc.value_bv_bits - 1, 0);

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
      bv_false.concat(fp_add_value.zext(enc.TYPE_BITS)),
    Expr::mkIte(((f1.getMSB() == bv_true) & (f2.getMSB() == bv_true)),
      // neg + neg -> neg
      bv_true.concat(fp_add_value.zext(enc.TYPE_BITS)),
    Expr::mkIte(f1.extract(enc.value_bv_bits - 1, 0) ==
                f2.extract(enc.value_bv_bits - 1, 0),
      // x + -x -> 0.0
      fp_zero,
      fp_add_sign.concat(fp_add_value.zext(enc.TYPE_BITS))
  ))))))))));
}

Expr fpMul(const Expr &_f1, const Expr &_f2) {
  usedOps.fpMul = true;
  auto &enc = *floatEnc;

  auto fp_zero_pos = Float(0.0f);
  auto fp_zero_neg = Float(-0.0f);
  auto fp_id = Float(1.0f);
  auto fp_neg = Float(-1.0f);
  auto fp_inf_pos = Float(INFINITY);
  auto fp_inf_neg = Float(-INFINITY);
  auto fp_nan = Float(nanf("0"));
  auto bv_true = Expr::mkBV(1, 1);
  auto bv_false = Expr::mkBV(0, 1);

  // Handle non-canonical NaNs
  const auto inf_value = ((Expr)fp_inf_pos).extract(enc.value_bv_bits - 1, 0);
  const auto inf_type = bv_true; 

  const auto f1_value = _f1.extract(enc.value_bv_bits - 1, 0);
  const auto f1_type = _f1.extract(enc.value_bv_bits, enc.value_bv_bits);
  const auto f1 = Expr::mkIte((f1_type == inf_type) & (f1_value != inf_value), fp_nan, _f1);

  const auto f2_value = _f2.extract(enc.value_bv_bits - 1, 0);
  const auto f2_type = _f2.extract(enc.value_bv_bits, enc.value_bv_bits);
  const auto f2 = Expr::mkIte((f2_type == inf_type) & (f2_value != inf_value), fp_nan, _f2);

  // The sign bit(s) will be replaced in the next step,
  // so it is better to completely ignore the signs in this step.
  // (This is why there's so many | in the conditions...)
  // 
  // 1.0 * x -> x, -1.0 * x -> -x
  auto fpmul_res = Expr::mkIte((f1 == fp_id) | (f1 == fp_neg), f2,
  // x * 1.0 -> x, x * -1.0 -> -x
  Expr::mkIte((f2 == fp_id) | (f2 == fp_neg), f1,
  // NaN * x -> NaN
  Expr::mkIte(f1 == fp_nan, f1,
  // x * NaN -> NaN
  Expr::mkIte(f2 == fp_nan, f2,
  // +-Inf * +-0.0 -> NaN , +-Inf * x -> ?Inf (if x != 0.0)
  // IEEE 754-2019 section 7.2 'Invalid operation'
  Expr::mkIte((f1 == fp_inf_pos) | (f1 == fp_inf_neg),
    Expr::mkIte((f2 == fp_zero_pos) | (f2 == fp_zero_neg), fp_nan, fp_inf_pos),
  // +-0.0 * +-Inf -> NaN , x * +-Inf -> ?Inf (if x != 0.0)
  // IEEE 754-2019 section 7.2 'Invalid operation'
  Expr::mkIte((f2 == fp_inf_pos) | (f2 == fp_inf_neg),
    Expr::mkIte((f1 == fp_zero_pos) | (f1 == fp_zero_neg), fp_nan, fp_inf_pos),
  // +-0.0 * x -> ?0.0, x * +-0.0 -> ?0.0
  Expr::mkIte((f1 == fp_zero_pos) | (f1 == fp_zero_neg) | (f2 == fp_zero_pos) | (f2 == fp_zero_neg), 
    fp_zero_pos,
    // If both operands do not fall into any of the cases above,
    // use fp_mul for abstract representation.
    // But fp_mul only yields BV[VALUE_BITS], so we must prepend
    // sign bit(s) and type bit(s) at the fp_mul result.
    // For type bit(s), we can just assume that they are 0,
    // because the result of fp_mul is always some finite value
    // as Infs and NaNs are already handled in the previous Ites.
    // For sign bits, as written in the comment above,
    // it is safe to use any value as they will be overwritten anyway.
    // Now that we know using 0 for both SIGN_BITS and TYPE_BITS is fine,
    // we can simply zext(2) the fp_mul
    // to obtain BV[SIGN_BITS + TYPE_BITS + VALUE_BITS] we want!
    //
    // We want the result of fp_mul to be an abstract and pairwise commutative value.
    // therefore we return fp_mul(f1, f2) + fp_mul(f2, f1)
    (enc.getMulFn().apply({f1, f2}) + enc.getMulFn().apply({f2, f1})).zext(2)
  )))))));

  // And at last we replace the sign with signbit(f1) ^ signbit(f2)
  // pos * pos | neg * neg -> pos, pos * neg | neg * pos -> neg
  return Expr::mkIte(fpmul_res == fp_nan, fp_nan,
    Expr::mkIte(f1.getMSB() == f2.getMSB(),
      bv_false.concat(fpmul_res.extract(enc.value_bv_bits, 0)),
      bv_true.concat(fpmul_res.extract(enc.value_bv_bits, 0))    
  ));
}

static Expr fpMultisetSum(const Expr &a, const Expr &n) {
  uint64_t length;
  if (!n.isUInt(length))
    assert("Only an array of constant length is supported.");

  auto &enc = *floatEnc;
  auto elemtSort = a.select(Index(0)).sort();
  auto bag = Expr::mkEmptyBag(elemtSort);
  for (unsigned i = 0; i < length; i ++) {
    bag = bag.insert(a.select(Index(i)));
    bag = bag.simplify();
  }

  Expr result = enc.getAssocSumFn()(bag);

  if (n.isNumeral())
    enc.fp_sum_relations.push_back({bag, n, result});

  return result;
}

Expr fpSum(const Expr &a, const Expr &n) {
  usedOps.fpSum = true;
  // TODO: check that a.Sort is Index::Sort() -> Float::Sort()

  if (isFpAddAssociative && useMultiset)
    return fpMultisetSum(a, n);

  auto &enc = *floatEnc;

  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr identity = Float(-0.0);
  Expr result = enc.getSumFn()(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n), ai, identity)));

  if (isFpAddAssociative && n.isNumeral())
    enc.fp_sum_relations.push_back({a, n, result});

  return result;
}

Expr fpDot(const Expr &a, const Expr &b, const Expr &n) {
  auto &enc = *floatEnc;

  if (alFpDot == AbsLevelFpDot::FULLY_ABS) {
    usedOps.fpDot = true;
    auto i = (Expr)Index::var("idx", VarType::BOUND);

    Expr ai = a.select(i), bi = b.select(i);
    Expr identity = Float(-0.0);
    // Encode commutativity: dot(a, b) = dot(b, a)
    Expr lhs = enc.getDotFn().apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity))});
    Expr rhs = enc.getDotFn().apply({
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity)),
        Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity))});
    return lhs + rhs;

  } else if (alFpDot == AbsLevelFpDot::SUM_MUL) {
    // usedOps.fpMul/fpSum will be updated by the fpMul()/fpSum() calls below
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, fpMul(ai, bi));

    return fpSum(arr, n);
  }
  llvm_unreachable("Unknown abstraction level for fp dot");
}

static Expr getFpAssociativePrecondition(
    const decltype(floatEnc->fp_sum_relations) &fp_sum_relations) {
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
      FnDecl hashfn(Float::sort(), Index::sort(), freshName("fp_hash"));

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
    cond &= getFpAssociativePrecondition(floatEnc->fp_sum_relations);
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
