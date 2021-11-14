#include "abstractops.h"
#include "simplevalue.h"
#include "smt.h"
#include "utils.h"
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
aop::AbsLevelFpCast alFpCast;
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
    AbsLevelFpDot afd, AbsLevelFpCast afc, AbsLevelIntDot aid, bool addAssoc,
    unsigned floatBits, unsigned doubleBits) {
  alFpDot = afd;
  alFpCast = afc;
  alIntDot = aid;
  isFpAddAssociative = addAssoc;

  unsigned doubleLimitBits, doublePrecBits;
  if (afc == AbsLevelFpCast::PRECISE) {
    doubleLimitBits = 1;
    if (doubleBits > floatBits + 2) {
      doublePrecBits = doubleBits - floatBits - 1;
    } else {
      doublePrecBits = 1;
    }
  } else {
    doubleLimitBits = 0;
    doublePrecBits = 0;
  }

  floatEnc.emplace(llvm::APFloat::IEEEsingle(), floatBits, "float");
  doubleEnc.emplace(llvm::APFloat::IEEEdouble(), doubleLimitBits,
      doublePrecBits, &*floatEnc, "double");
}

// A set of options that must not change the precision of validation.
void setEncodingOptions(bool use_multiset) {
  useMultiset = use_multiset;
}

bool getFpAddAssociativity() { return isFpAddAssociative; }

AbsFpEncoding &getFloatEncoding() { return *floatEnc; }
AbsFpEncoding &getDoubleEncoding() { return *doubleEnc; }
AbsFpEncoding &getFpEncoding(mlir::Type ty) {
  if (ty.isF32()) {
    return getFloatEncoding();
  } else if (ty.isF64()) {
     return getDoubleEncoding();
  }
  llvm_unreachable("Unknown type");
}

AbsFpEncoding::AbsFpEncoding(const llvm::fltSemantics &semantics,
      unsigned limitbits, unsigned precbits, unsigned valuebits,
      AbsFpEncoding* smaller_fpty_enc, std::string &&fn_suffix)
     :semantics(semantics), fn_suffix(move(fn_suffix)) {
  assert(valuebits > 0);
  // BWs for casting
  limit_bv_bits = limitbits;
  prec_bv_bits = precbits;
  this->smaller_fpty_enc = smaller_fpty_enc;

  value_bv_bits = limitbits + precbits + valuebits;
  fp_bv_bits = SIGN_BITS + value_bv_bits;

  // INF: s11..10 where s is a sign bit
  const uint64_t inf_value = (1ull << (uint64_t)value_bv_bits) - 2;
  // NAN: s11..11 where s is a sign bit
  const uint64_t nan_value = (1ull << (uint64_t)value_bv_bits) - 1;
  const uint64_t signed_value = 1ull << (uint64_t)value_bv_bits;

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
  fp_ultfn.reset();
  fp_sum_relations.clear();
}

FnDecl AbsFpEncoding::getAddFn() {
  if (!fp_addfn) {
    auto fty = sort();
    fp_addfn.emplace({fty, fty}, fty, "fp_add_" + fn_suffix);
  }
  return *fp_addfn;
}

FnDecl AbsFpEncoding::getMulFn() {
  if (!fp_mulfn) {
    auto fty = Sort::bvSort(value_bv_bits);
    auto fty2 = Sort::bvSort(value_bv_bits);
    fp_mulfn.emplace({fty, fty}, fty2, "fp_mul_" + fn_suffix);
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

FnDecl AbsFpEncoding::getUltFn() {
  if (!fp_ultfn) {
    auto fty = Sort::bvSort(fp_bv_bits);
    auto fty2 = Sort::bvSort(1); // i1 type (boolean value)
    fp_ultfn.emplace({fty, fty}, fty2, "fp_ult_" + fn_suffix);
  }
  return *fp_ultfn;
}

FnDecl AbsFpEncoding::getExtendFn() {
  if (!fp_extendfn) {
    // In the fully abstract world, double and float have same bitwidth.
    auto fty = Sort::bvSort(fp_bv_bits);
    fp_extendfn.emplace({fty}, fty, "fp_extract_" + fn_suffix);
  }
  return *fp_extendfn;
}

uint64_t AbsFpEncoding::getSignBit() const {
  assert(value_bv_bits + SIGN_BITS == fp_bv_bits);
  return 1ull << value_bv_bits;
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
  // value_id is 0: zero, 1: one, 2: others
  if (abs_f.compare(llvm::APFloat(semantics, 1)) == llvm::APFloat::cmpEqual) {
    value_id = 1;
  } else {
    assert(static_cast<uint64_t>(2 + fpconst_absrepr_num) <
        (1ull << (uint64_t)value_bv_bits) - 2);
    value_id = 2 + fpconst_absrepr_num++;
  }

  Expr e_pos = Expr::mkBV(value_id, fp_bv_bits);
  fpconst_absrepr.emplace(abs_f, e_pos);
  Expr e_neg = Expr::mkBV(getSignBit() | value_id, fp_bv_bits);
  fpconst_absrepr.emplace(-abs_f, e_neg);

  return f.isNegative() ? e_neg : e_pos;
}

vector<pair<llvm::APFloat, Expr>> AbsFpEncoding::getAllConstants() const {
  vector<pair<llvm::APFloat, smt::Expr>> constants;
  for (auto &[k, v]: fpconst_absrepr) constants.emplace_back(k, v);

  if (fpconst_nan)
    constants.emplace_back(llvm::APFloat::getNaN(semantics), *fpconst_nan);
  if (fpconst_zero_pos)
    constants.emplace_back(llvm::APFloat::getZero(semantics), *fpconst_zero_pos);
  if (fpconst_zero_neg)
    constants.emplace_back(llvm::APFloat::getZero(semantics, true), *fpconst_zero_neg);
  if (fpconst_inf_pos)
    constants.emplace_back(llvm::APFloat::getInf(semantics), *fpconst_inf_pos);
  if (fpconst_inf_neg)
    constants.emplace_back(llvm::APFloat::getInf(semantics, true), *fpconst_inf_neg);

  return constants;
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
  llvm::APFloat apf(semantics, 1);
  if (isNegative)
    apf.changeSign();
  return constant(apf);
}

Expr AbsFpEncoding::infinity(bool isNegative) {
  return constant(llvm::APFloat::getInf(semantics, isNegative));
}

Expr AbsFpEncoding::nan() {
  return constant(llvm::APFloat::getNaN(semantics));
}

Expr AbsFpEncoding::isnan(const Expr &f) {
  // Modulo the sign bit, there is only one NaN representation in abs encoding.
  return f.extract(value_bv_bits - 1, 0) == nan().extract(value_bv_bits - 1, 0);
}

Expr AbsFpEncoding::abs(const Expr &f) {
  return Expr::mkBV(0, 1).concat(f.extract(fp_bv_bits - 2, 0));
}

Expr AbsFpEncoding::neg(const Expr &f) {
  auto sign = f.extract(fp_bv_bits - 1, fp_bv_bits - 1);
  auto sign_negated = sign ^ 1;
  return sign_negated.concat(f.extract(fp_bv_bits - 2, 0));
}

Expr AbsFpEncoding::add(const Expr &_f1, const Expr &_f2) {
  usedOps.fpAdd = true;

  const auto &fp_id = zero(true);
  const auto fp_inf_pos = infinity();
  const auto fp_inf_neg = infinity(true);
  const auto fp_nan = nan();
  const auto bv_true = Expr::mkBV(1, 1);
  const auto bv_false = Expr::mkBV(0, 1);

  // Handle non-canonical NaNs
  const auto f1 = Expr::mkIte(isnan(_f1), fp_nan, _f1);
  const auto f2 = Expr::mkIte(isnan(_f2), fp_nan, _f2);

  // Encode commutativity without loss of generality
  auto fp_add_res = getAddFn().apply({f1, f2}) + getAddFn().apply({f2, f1});
  // The result of addition cannot be NaN if inputs aren't.
  // This NaN case is specially treated below.
  // Simply redirect the result to zero.
  fp_add_res = Expr::mkIte(isnan(fp_add_res), zero(), fp_add_res);
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
    // 
    // There are two cases where we must override the sign bit of fp_add.
    // If signbit(f1) == 0 /\ signbit(f2) == 0, signbit(fpAdd(f1, f2)) = 0.
    // If signbit(f1) == 1 /\ signbit(f2) == 1, signbit(fpAdd(f1, f2)) = 1.
    // Otherwise, we can just use the arbitrary sign yielded from fp_add.
    Expr::mkIte(((f1.getMSB() == bv_false) & (f2.getMSB() == bv_false)),
      // pos + pos -> pos
      bv_false.concat(fp_add_value),
    Expr::mkIte(((f1.getMSB() == bv_true) & (f2.getMSB() == bv_true)),
      // neg + neg -> neg
      bv_true.concat(fp_add_value),
    Expr::mkIte(f1.extract(value_bv_bits - 1, 0) ==
                f2.extract(value_bv_bits - 1, 0),
      // x + -x -> 0.0
      zero(),
      fp_add_res
  ))))))))));
}

Expr AbsFpEncoding::mul(const Expr &_f1, const Expr &_f2) {
  usedOps.fpMul = true;

  auto fp_zero_pos = zero();
  auto fp_zero_neg = zero(true);
  auto fp_id = one();
  auto fp_minusone = one(true);
  auto fp_inf_pos = infinity();
  auto fp_inf_neg = infinity(true);
  auto fp_nan = nan();
  auto bv_true = Expr::mkBV(1, 1);
  auto bv_false = Expr::mkBV(0, 1);

  // Handle non-canonical NaNs
  const auto f1 = Expr::mkIte(isnan(_f1), fp_nan, _f1);
  const auto f2 = Expr::mkIte(isnan(_f2), fp_nan, _f2);
  const auto f1_nosign = f1.extract(fp_bv_bits - 2, 0);
  const auto f2_nosign = f2.extract(fp_bv_bits - 2, 0);

  // Encode commutativity of mul.
  // getMulFn()'s range is BV[VALUE_BITS] because it encodes absolute size of
  // mul.
  // We zero-extend 1 bit (SIGN-BIT) which is actually a dummy bit.
  auto mul_abs_res = (getMulFn().apply({f1_nosign, f2_nosign}) +
                  getMulFn().apply({f2_nosign, f1_nosign})).zext(1);
  // Absolute size of mul cannot be NaN (Inf * 0.0 and NaN * x will be special-
  // cased).
  mul_abs_res = Expr::mkIte(isnan(mul_abs_res), fp_id, mul_abs_res);

  // Calculate the absolute value of f1 * f2.
  // The sign bit(s) will be replaced in the next step,
  // so it is better to completely ignore the signs in this step.
  // (This is why there's so many | in the conditions...)
  // 
  // 1.0 * x -> x, -1.0 * x -> -x
  auto fpmul_res = Expr::mkIte((f1 == fp_id) | (f1 == fp_minusone), f2,
  // x * 1.0 -> x, x * -1.0 -> -x
  Expr::mkIte((f2 == fp_id) | (f2 == fp_minusone), f1,
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
  Expr::mkIte((f1 == fp_zero_pos) | (f1 == fp_zero_neg) | (f2 == fp_zero_pos) |
              (f2 == fp_zero_neg), 
    fp_zero_pos,
    // If both operands do not fall into any of the cases above,
    // use fp_mul for abstract representation.
    mul_abs_res
  )))))));

  // And at last we replace the sign with signbit(f1) ^ signbit(f2)
  // pos * pos | neg * neg -> pos, pos * neg | neg * pos -> neg
  return Expr::mkIte(fpmul_res == fp_nan, fp_nan,
    Expr::mkIte(f1.getMSB() == f2.getMSB(),
      bv_false.concat(fpmul_res.extract(value_bv_bits - 1, 0)),
      bv_true.concat(fpmul_res.extract(value_bv_bits - 1, 0))
  ));
}

Expr AbsFpEncoding::multisetSum(const Expr &a, const Expr &n) {
  uint64_t length;
  if (!n.isUInt(length))
    throw UnsupportedException("Only an array of constant length is supported.");

  auto elemtSort = a.select(Index(0)).sort();
  auto bag = Expr::mkEmptyBag(elemtSort);
  for (unsigned i = 0; i < length; i ++) {
    auto ai = a.select(Index(i));
    bag = bag.insert(Expr::mkIte(isnan(ai), nan(), ai));
    bag = bag.simplify();
  }

  Expr result = getAssocSumFn()(bag);
  fp_sum_relations.push_back({bag, n, result});

  return result;
}

Expr AbsFpEncoding::sum(const Expr &a, const Expr &n) {
  if (getFpAddAssociativity() && !n.isNumeral())
    throw UnsupportedException("Only an array of constant length is supported.");

  usedOps.fpSum = true;

  if (getFpAddAssociativity() && useMultiset)
    return multisetSum(a, n);

  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr result = getSumFn()(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n),
        Expr::mkIte(isnan(ai), nan(), ai), zero(true))));

  if (getFpAddAssociativity())
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

Expr AbsFpEncoding::fult(const Expr &f1, const Expr &f2) {
  usedOps.fpUlt = true;
  return getUltFn().apply({f1, f2});
}

Expr AbsFpEncoding::extend(const smt::Expr &f, aop::AbsFpEncoding &tgt) {
  usedOps.fpCastRound = true;

  if (value_bv_bits == tgt.value_bv_bits) {
    // Fully abstract encoding 
    return getExtendFn().apply(f);
  }

  assert(value_bv_bits < tgt.value_bv_bits &&
         "tgt cannot have smaller value_bv_bits than src");

  if (limit_bv_bits != 0 || prec_bv_bits != 0)
    throw UnsupportedException("Casting from middle-size type to large-size "
        "type is not supported");

  auto sign_bit = f.extract(fp_bv_bits - 1, value_bv_bits);
  auto limit_zero = Expr::mkBV(0, tgt.limit_bv_bits);
  auto value_bits = f.extract(value_bv_bits - 1, 0);
  auto prec_zero = Expr::mkBV(0, tgt.prec_bv_bits);

  auto extended_float = sign_bit.concat(limit_zero).concat(value_bits)
      .concat(prec_zero);
  assert(extended_float.bitwidth() == tgt.sort().bitwidth());
  return Expr::mkIte(isnan(f), tgt.nan(),
      Expr::mkIte(f == infinity(), tgt.infinity(),
      Expr::mkIte(f == infinity(true), tgt.infinity(true),
      extended_float)));
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

Expr getFpUltPrecondition() {
  Expr cond = Expr::mkBool(true);

  if (floatEnc) {
    auto constants = floatEnc->getAllConstants();
    for (auto &[const1, expr1]: constants) {
      for (auto &[const2, expr2]: constants) {
        if (const1.compare(const2) == llvm::APFloat::cmpLessThan ||
              const1.compare(const2) == llvm::APFloat::cmpUnordered) {
          cond = cond & floatEnc->fult(expr1, expr2) == Integer::boolTrue();
        }
      }
    }
  }
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
