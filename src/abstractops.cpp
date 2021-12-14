#include "abstractops.h"
#include "debug.h"
#include "simplevalue.h"
#include "smt.h"
#include "utils.h"
#include "value.h"
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
bool doUnrollIntSum;
unsigned maxUnrollFpSumBound;

optional<aop::AbsFpEncoding> floatEnc;
optional<aop::AbsFpEncoding> doubleEnc;

// ----- Constants and global vars for abstract int operations ------

aop::AbsLevelIntDot alIntDot;
map<unsigned, FnDecl> int_sumfn;
map<unsigned, FnDecl> int_dotfn;

// ----- Constants and global vars for abstract sumf operations ------
aop::AbsFpAddSumEncoding fpAddSum;

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

struct FPCastingInfo {
  // If false, the result of rounding is +inf
  bool zero_limit_bits;
  // If false, rounding loses lowest bits
  bool zero_prec_bits;
  bool is_rounded_upward;
};

optional<FPCastingInfo> getCastingInfo(llvm::APFloat fp_const) {
  assert(!fp_const.isNegative());

  auto semantics = llvm::APFloat::SemanticsToEnum(fp_const.getSemantics());
  bool zero_limit_bits = true, lost_info;

  if (semantics == llvm::APFloat::Semantics::S_IEEEsingle) {
    return nullopt;

  } else if (semantics == llvm::APFloat::Semantics::S_IEEEdouble) {
    auto fp_const_floor = fp_const;
    fp_const_floor.convert(llvm::APFloat::IEEEsingle(),
                      // floor to analyze rounding direction
                      llvm::APFloat::rmTowardZero, &lost_info);

    auto op_status = fp_const.convert(llvm::APFloat::IEEEsingle(),
                      // round to correctly analyze overflow
                      llvm::APFloat::rmNearestTiesToEven, &lost_info);
    if (op_status & llvm::APFloat::opOverflow) {
      zero_limit_bits = false;
    }
    bool zero_prec_bits = !lost_info;
    bool is_rounded_upward = (fp_const_floor != fp_const);

    return FPCastingInfo {zero_limit_bits, zero_prec_bits, is_rounded_upward};
  } else {
    throw UnsupportedException(
      "Cannot analyze casting information for this type");
  }
}
}


namespace aop {

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstraction(
    AbsLevelFpDot afd, AbsLevelFpCast afc, AbsLevelIntDot aid, AbsFpAddSumEncoding fas,
    bool addAssoc,
    bool unrollIntSum,
    unsigned unrollFpSumBound,
    unsigned floatNonConstsCnt, set<llvm::APFloat> floatConsts,
    unsigned doubleNonConstsCnt, set<llvm::APFloat> doubleConsts) {
  alFpDot = afd;
  alFpCast = afc;
  alIntDot = aid;
  fpAddSum = fas;
  doUnrollIntSum = unrollIntSum;
  maxUnrollFpSumBound = unrollFpSumBound;
  isFpAddAssociative = addAssoc;

  // without suffix f, it will become llvm::APFloat with double semantics
  // Note that 0.0 and 1.0 may already have been added during analysis.
  // 0.0 and 1.0 are necessary to prove several arithmetic properties,
  // so we're manually inserting 0.0 and 1.0
  // just in case they are not added during the analysis.
  floatConsts.emplace(0.0f);
  floatConsts.emplace(1.0f);
  doubleConsts.emplace(0.0);
  doubleConsts.emplace(1.0);

  // + 2: reserved for +NaN, +Inf; separately counted because they cannot be
  // included in set<APFloat>
  unsigned floatBits = log2_ceil(floatNonConstsCnt + floatConsts.size() + 2);
  floatEnc.emplace(llvm::APFloat::IEEEsingle(), floatBits, "float");
  floatEnc->addConstants(floatConsts);

  if (afc == AbsLevelFpCast::PRECISE) {
    unsigned consts_nonzero_limit = 0;
    unsigned const_nonzero_precs = 0, const_max_nonzero_precs = 0;

    // Visit non-negative fp consts by increasing order
    for (const auto& dbl_const : doubleConsts) {
      assert(!dbl_const.isNegative());

      auto casting_info = getCastingInfo(dbl_const);

      if (!casting_info->zero_limit_bits) {
        consts_nonzero_limit += 1;
      } else if (!casting_info->zero_prec_bits) {
        // count the maximum number of values
        // that converges to single value when rounded
        const_nonzero_precs += 1;
      } else {
        // const_nonzero_precs: # of fps requiring rounding
        const_max_nonzero_precs =
            max(const_nonzero_precs, const_max_nonzero_precs);
        const_nonzero_precs = 0;
      }
    }
    const_max_nonzero_precs = max(const_nonzero_precs, const_max_nonzero_precs);

    // all double variables may have same SVB but have different PBs
    const unsigned min_prec_bitwidth = max((uint64_t)1,
      max(log2_ceil(const_max_nonzero_precs), log2_ceil(doubleNonConstsCnt)));

    // Decide min_limit_bitwidth.
    // We're not going to simply use log2(consts_nonzero_limit) as limit bit
    // because precision bits can be utilized to reduce the whole bitwidth.
    // If limit bit is not zero, precision bits can be reused as floatBits.
    unsigned min_limit_bitwidth;
    const unsigned bitwidth_for_nonzero_limits =
        max(log2_ceil(consts_nonzero_limit), log2_ceil(doubleNonConstsCnt));

    if (bitwidth_for_nonzero_limits > floatBits + min_prec_bitwidth) {
      // Extend limit bits 
      // because many double constants require limit bit to be set.
      min_limit_bitwidth = 
          bitwidth_for_nonzero_limits - floatBits - min_prec_bitwidth + 1;
    } else {
      // reserve at least one bit for limit bit
      // for proper Inf/NaN handling
      min_limit_bitwidth = 1;
    }

    // 29: mantissa(double) - mantissa(float)
    // Using more than 29 precision bits is unnecessary
    // because such values do not exist in the real-world double
    const unsigned doublePrecBits = min(min_prec_bitwidth, 29u);
    // 32: bitwidth(double) - bitwidth(float)
    // Using more than 32 (limit + precision) bits is unnecessary
    // because such values do not exist in the real-world double
    // And this value will never overflow, because doublePrecBits is always <=29
    const unsigned doubleLimitBits = min(min_limit_bitwidth,
                                          32u - doublePrecBits);
    doubleEnc.emplace(llvm::APFloat::IEEEdouble(), doubleLimitBits,
        doublePrecBits, &*floatEnc, "double");
  } else {
    const unsigned doubleBits = 
        log2_ceil(doubleNonConstsCnt + doubleConsts.size() + 2);
    doubleEnc.emplace(llvm::APFloat::IEEEdouble(), doubleBits, "double");
  }
  doubleEnc->addConstants(doubleConsts);
}

// A set of options that must not change the precision of validation.
void setEncodingOptions(bool use_multiset) {
  useMultiset = use_multiset;
}

AbsLevelFpDot getAbsLevelFpDot() { return alFpDot; }
AbsLevelFpCast getAbsLevelFpCast() { return alFpCast; }
AbsLevelIntDot getAbsLevelIntDot() { return alIntDot; }
AbsFpAddSumEncoding getAbsFpAddSumEncoding() { return fpAddSum; }

bool getFpAddAssociativity() { return isFpAddAssociative; }

bool getFpCastIsPrecise() { return alFpCast == AbsLevelFpCast::PRECISE; }

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
      unsigned limit_bw, unsigned smaller_value_bw, unsigned prec_bw,
       std::string &&fnsuffix)
     :semantics(semantics), fn_suffix(move(fnsuffix)) {
  assert(smaller_value_bw > 0);
  verbose("AbsFpEncoding") << fn_suffix << ": limit bits: " << limit_bw
      << ", smaller value bits: " << smaller_value_bw << ", precision bits: "
      << prec_bw << '\n';
  // BWs for casting
  value_bit_info = { limit_bw, smaller_value_bw, prec_bw };
  value_bitwidth = value_bit_info.get_value_bitwidth();

  fp_bitwidth = SIGN_BITS + value_bitwidth;

  // INF: s11..10 where s is a sign bit
  const uint64_t inf_value = (1ull << (uint64_t)value_bitwidth) - 2;
  // NAN: s11..11 where s is a sign bit
  const uint64_t nan_value = (1ull << (uint64_t)value_bitwidth) - 1;
  const uint64_t signed_value = 1ull << (uint64_t)value_bitwidth;

  fpconst_nan = Expr::mkBV(nan_value, fp_bitwidth);
  fpconst_inf_pos = Expr::mkBV(inf_value, fp_bitwidth);
  fpconst_inf_neg = Expr::mkBV(signed_value + inf_value, fp_bitwidth);
  fpconst_zero_pos = Expr::mkBV(0, fp_bitwidth);
  fpconst_zero_neg = Expr::mkBV(signed_value + 0, fp_bitwidth);

  fp_sumfn.reset();
  fp_assoc_sumfn.reset();
  fp_dotfn.reset();
  fp_addfn.reset();
  fp_mulfn.reset();
  fp_divfn.reset();
  fp_ultfn.reset();
  fp_hashfn.reset();
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
    auto fty = Sort::bvSort(value_bitwidth);
    auto fty2 = Sort::bvSort(value_bitwidth);
    fp_mulfn.emplace({fty, fty}, fty2, "fp_mul_" + fn_suffix);
  }
  return *fp_mulfn;
}

FnDecl AbsFpEncoding::getDivFn() {
  if (!fp_mulfn) {
    auto fty = Sort::bvSort(value_bitwidth);
    auto fty2 = Sort::bvSort(value_bitwidth);
    fp_mulfn.emplace({fty, fty}, fty2, "fp_div_" + fn_suffix);
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
    auto fty = sort();
    auto fty2 = Sort::bvSort(1); // i1 type (boolean value)
    fp_ultfn.emplace({fty, fty}, fty2, "fp_ult_" + fn_suffix);
  }
  return *fp_ultfn;
}

FnDecl AbsFpEncoding::getExtendFn(const AbsFpEncoding &tgt) {
  if (!fp_extendfn) {
    auto src_fty = sort();
    auto tgt_fty = tgt.sort();
    fp_extendfn.emplace({src_fty}, tgt_fty, "fp_extend_" + fn_suffix);
  }
  return *fp_extendfn;
}

FnDecl AbsFpEncoding::getTruncateFn(const AbsFpEncoding &tgt) {
  if (!fp_truncatefn) {
    auto src_fty = sort();
    auto tgt_fty = tgt.sort();
    fp_truncatefn.emplace({src_fty}, tgt_fty, "fp_truncate_" + fn_suffix);
  }
  return *fp_truncatefn;
}

FnDecl AbsFpEncoding::getExpFn() {
  if (!fp_expfn) {
    auto fty = sort();
    fp_expfn.emplace({fty}, fty, "fp_exp_" + fn_suffix);
  }
  return *fp_expfn;
}

FnDecl AbsFpEncoding::getHashFnForAddAssoc() {
  if (!fp_hashfn) {
    auto fty = sort();
    fp_hashfn.emplace(fty, Sort::bvSort(getHashRangeBits()),
        "fp_hash_" + fn_suffix);
  } else {
    // Hash range bits must not be changed.
    assert(fp_hashfn->getRange().bitwidth() == getHashRangeBits());
  }
  return *fp_hashfn;
}

FnDecl AbsFpEncoding::getRoundDirFn() {
  if (!fp_rounddirfn) {
    auto src_fty = Sort::bvSort(value_bitwidth);
    auto tgt_fty = Sort::bvSort(1);
    fp_rounddirfn.emplace({src_fty}, tgt_fty, "fp_rounddir_" + fn_suffix);
  }
  return *fp_rounddirfn;
}

size_t AbsFpEncoding::getHashRangeBits() const {
  uint64_t numRelations = fp_sum_relations.size();
  uint64_t maxLength = 0;
  for (auto &rel: fp_sum_relations) {
    auto expr = get<1>(rel);
    uint64_t length;
    if (!expr.isUInt(length))
      length = Tensor::MAX_TENSOR_SIZE;
    if (maxLength < length)
      maxLength = length;
  }

  uint64_t bounds = numRelations * numRelations * maxLength;
  return max((uint64_t)1, log2_ceil(bounds));
}

uint64_t AbsFpEncoding::getSignBit() const {
  assert(value_bitwidth + SIGN_BITS == fp_bitwidth);
  return 1ull << value_bitwidth;
}

void AbsFpEncoding::addConstants(const set<llvm::APFloat>& const_set) {
  uint64_t value_id = 0;
  Expr small_value_bits = Expr::mkBV(0, value_bit_info.truncated_bitwidth);
  // prec_offset_map[smaller value]: next precision bit
  map<uint64_t, uint64_t> prec_offset_map;

  auto mkNonzero = [](Expr &&e) {
    return Expr::mkIte(e == 0, Expr::mkBV(1, e), e);
  };

  // Visit non-negative constants in increasing order.
  for (const auto& fp_const: const_set) {
    assert(!fp_const.isNegative() &&
            "const_set must only consist of non-negative consts!");
    if (fp_const.isZero()) {
      // 0.0 should not be added to absrepr
      continue;
    }

    optional<Expr> e_value;

    if (value_bit_info.limit_bitwidth == 0 &&
        value_bit_info.prec_bitwidth == 0) {
      // This encoding is the smallest encoding or does not support casting
      value_id += 1;
      // This naming convention is used by getFpTruncatePrecondition to relate
      // float and double constants
      e_value = Expr::mkVar(Sort::bvSort(value_bitwidth),
                            "fp_const_sval_" + to_string(value_id) + "_");
    } else {
      auto casting_info = getCastingInfo(fp_const);
      assert(casting_info.has_value() &&
             "this encoding requires casting info analysis for constants");
      assert(value_bit_info.limit_bitwidth > 0 && "limit bits cannot be zero");

      // default zero-init BVs
      auto limit_bits = Expr::mkBV(0, value_bit_info.limit_bitwidth);
      auto sv_bits = Expr::mkBV(0, value_bit_info.truncated_bitwidth);

      if (!casting_info->zero_limit_bits) {
        // Rounding becomes inf.
        value_id += 1;
        const unsigned int sv_prec_bitwidth =
          value_bit_info.truncated_bitwidth + value_bit_info.prec_bitwidth;
        auto sv_prec_bits = Expr::mkFreshVar(Sort::bvSort(sv_prec_bitwidth),
                            "fp_const_sval_prec_bits_");

        limit_bits = mkNonzero(
            Expr::mkFreshVar(limit_bits, "fp_const_limit_bits_"));
        e_value = limit_bits.concat(sv_prec_bits);

      } else if (!casting_info->zero_prec_bits) {
        // Rounding loses bits & the result becomes equivalent to the previous
        // rounding output.
        // e.g. fp_const is 2.2 (it was 2.1 in the prev. iter)
        // Do not change smaller_value and increment prec bit.
        auto itr = prec_offset_map.insert({value_id, 1}).first;
        uint64_t prec = itr->second;
        assert(prec < (1ull << value_bit_info.prec_bitwidth));

        auto prec_bits = mkNonzero(Expr::mkFreshVar(
            Sort::bvSort(value_bit_info.prec_bitwidth),
            "fp_const_prec_bits"));
        sv_bits = Expr::mkVar(sv_bits, 
                              "fp_const_sval_" + to_string(value_id) + "_");
        e_value = limit_bits.concat(sv_bits).concat(prec_bits);
        // Increase the next precision bit.
        itr->second++;

      } else {
        // Rounding does not lose bits, but will become a value other than the
        // previous one
        // e.g. fp_const is 3.0 (it was 2.2 in the prev. iter)
        // Assign different smaller_value
        value_id += 1;
        sv_bits = Expr::mkVar(sv_bits, 
                              "fp_const_sval_" + to_string(value_id) + "_");
        e_value = limit_bits.concat(sv_bits);
        // this encoding may not have prec bits
        if (value_bit_info.prec_bitwidth > 0) {
          auto prec_bits = Expr::mkBV(0, value_bit_info.prec_bitwidth);
          e_value = e_value->concat(prec_bits);
        }
      }
    }

    verbose("addConstants") << fp_const.convertToDouble() << ": " << *e_value
        << "\n";

    Expr e_pos = Expr::mkBV(0, SIGN_BITS).concat(*e_value);
    fpconst_absrepr.emplace(fp_const, e_pos);
    Expr e_neg = Expr::mkBV(1, SIGN_BITS).concat(*e_value);
    fpconst_absrepr.emplace(-fp_const, e_neg);
  }
}

Expr AbsFpEncoding::constant(const llvm::APFloat &f) const {
  if (f.isNaN())
    return *fpconst_nan;
  else if (f.isInfinity())
    return f.isNegative() ? *fpconst_inf_neg : *fpconst_inf_pos;
  else if (f.isPosZero())
    return *fpconst_zero_pos;
  else if (f.isNegZero())
    return *fpconst_zero_neg;

  // all other constant values in src and tgt IRs are added at analysis stage,
  // so this expression should never fail!
  auto itr = fpconst_absrepr.find(f);
  assert(itr != fpconst_absrepr.end()
          && "This constant does not have assigned abstract representation!");
  return itr->second;
}

vector<pair<llvm::APFloat, Expr>> AbsFpEncoding::getAllConstants() const {
  vector<pair<llvm::APFloat, smt::Expr>> constants;
  for (auto &[k, v]: fpconst_absrepr) constants.emplace_back(k, v);

  if (fpconst_nan)
    constants.emplace_back(llvm::APFloat::getNaN(semantics), *fpconst_nan);
  if (fpconst_zero_pos)
    constants.emplace_back(llvm::APFloat::getZero(semantics),
        *fpconst_zero_pos);
  if (fpconst_zero_neg)
    constants.emplace_back(llvm::APFloat::getZero(semantics, true),
        *fpconst_zero_neg);
  if (fpconst_inf_pos)
    constants.emplace_back(llvm::APFloat::getInf(semantics), *fpconst_inf_pos);
  if (fpconst_inf_neg)
    constants.emplace_back(llvm::APFloat::getInf(semantics, true),
        *fpconst_inf_neg);

  return constants;
}

vector<llvm::APFloat> AbsFpEncoding::possibleConsts(const Expr &e) const {
  vector<llvm::APFloat> vec;

  // expressions must be simplified in advance
  // because smt::Expr::isIdentical() fails to evaluate the identity
  // unless both lhs and rhs are in their canonical form!
  auto e_simp = e.simplify();

  for (auto &[k, v]: fpconst_absrepr) {
    if (v.simplify().isIdentical(e_simp))
      vec.push_back(k);
  }

  // for 'reserved' values that do not belong to fpconst_absrepr
  if (fpconst_nan &&
      getMagnitudeBits(*fpconst_nan).isIdentical(getMagnitudeBits(e_simp))) {
    vec.push_back(llvm::APFloat::getNaN(semantics));
  } else if (fpconst_zero_pos && fpconst_zero_pos->isIdentical(e_simp)) {
    vec.push_back(llvm::APFloat::getZero(semantics));
  } else if (fpconst_zero_neg && fpconst_zero_neg->isIdentical(e_simp)) {
    vec.push_back(llvm::APFloat::getZero(semantics, true));
  } else if (fpconst_inf_pos && fpconst_inf_pos->isIdentical(e_simp)) {
    vec.push_back(llvm::APFloat::getInf(semantics));
  } else if (fpconst_inf_neg && fpconst_inf_neg->isIdentical(e_simp)) {
    vec.push_back(llvm::APFloat::getInf(semantics, true));
  }

  return vec;
}

Expr AbsFpEncoding::zero(bool isNegative) const {
  return constant(llvm::APFloat::getZero(semantics, isNegative));
}

Expr AbsFpEncoding::one(bool isNegative) const {
  llvm::APFloat apf(semantics, 1);
  if (isNegative)
    apf.changeSign();
  return constant(apf);
}

Expr AbsFpEncoding::infinity(bool isNegative) const {
  return constant(llvm::APFloat::getInf(semantics, isNegative));
}

Expr AbsFpEncoding::nan() const {
  return constant(llvm::APFloat::getNaN(semantics));
}

Expr AbsFpEncoding::isnan(const Expr &f) {
  // Modulo the sign bit, there is only one NaN representation in abs encoding.
  return getMagnitudeBits(f) == getMagnitudeBits(nan());
}

Expr AbsFpEncoding::isinf(const Expr &f, bool isNegative) {
  return f == infinity(isNegative);
}

Expr AbsFpEncoding::iszero(const Expr &f, bool isNegative) {
  return f == zero(isNegative);
}

Expr AbsFpEncoding::abs(const Expr &f) {
  return Expr::mkBV(0, 1).concat(getMagnitudeBits(f));
}

Expr AbsFpEncoding::neg(const Expr &f) {
  auto sign = getSignBit(f);
  auto sign_negated = sign ^ 1;
  return sign_negated.concat(getMagnitudeBits(f));
}

Expr AbsFpEncoding::add(const Expr &_f1, const Expr &_f2) {
  if (fpAddSum == AbsFpAddSumEncoding::USE_SUM_ONLY) {
    auto i = Index::var("idx", VarType::BOUND);
    auto lambda = Expr::mkLambda(i, Expr::mkIte(i == Index::zero(), _f1, _f2));
    auto n = Index(2);
    return sum(lambda, n);
  }

  usedOps.fpAdd = true;

  const auto fp_id = zero(true);
  const auto fp_inf_pos = infinity();
  const auto fp_inf_neg = infinity(true);
  const auto fp_nan = nan();
  const auto bv_true = Expr::mkBV(1, 1);
  const auto bv_false = Expr::mkBV(0, 1);

  // Handle non-canonical NaNs (can have two different signs)
  const auto f1 = Expr::mkIte(isnan(_f1), fp_nan, _f1);
  const auto f2 = Expr::mkIte(isnan(_f2), fp_nan, _f2);

  // Encode commutativity without loss of generality
  // To avoid that the LSB of add(x, x) is always 0, encode separately.
  auto fp_add_res = Expr::mkIte(f1 == f2,
    getAddFn().apply({f1, f1}),
    getAddFn().apply({f1, f2}) + getAddFn().apply({f2, f1})
  );
  // The result of addition cannot be NaN if inputs aren't.
  // This NaN case is specially treated below.
  // Simply redirect the result to zero.
  fp_add_res = Expr::mkIte(isnan(fp_add_res), zero(), fp_add_res);
  auto fp_add_sign = getSignBit(fp_add_res);
  auto fp_add_value = getMagnitudeBits(fp_add_res);

  return Expr::mkIte(f1 == fp_id, f2,         // -0.0 + x -> x
    Expr::mkIte(f2 == fp_id, f1,              // x + -0.0 -> x
      Expr::mkIte(f1 == fp_nan, f1,           // NaN + x -> NaN
        Expr::mkIte(f2 == fp_nan, f2,         // x + NaN -> NaN
    // inf + -inf -> NaN, -inf + inf -> NaN
    // IEEE 754-2019 section 7.2 'Invalid operation'
    Expr::mkIte((isinf(f1, false) & isinf(f2, true)) |
                (isinf(f1, true) & isinf(f2, false)), fp_nan,
    // inf + x -> inf, -inf + x -> -inf (both commutative)
    // IEEE 754-2019 section 6.1 'Infinity arithmetic'
    Expr::mkIte(isinf(f1, false) | isinf(f1, true), f1,
      Expr::mkIte(isinf(f2, false) | isinf(f2, true), f2,
    // If both operands do not fall into any of the cases above,
    // use fp_add for abstract representation.
    // 
    // There are two cases where we must override the sign bit of fp_add.
    // If signbit(f1) == 0 /\ signbit(f2) == 0, signbit(fpAdd(f1, f2)) = 0.
    // If signbit(f1) == 1 /\ signbit(f2) == 1, signbit(fpAdd(f1, f2)) = 1.
    // Otherwise, we can just use the arbitrary sign yielded from fp_add.
    Expr::mkIte(((getSignBit(f1) == bv_false) & (getSignBit(f2) == bv_false)),
      // pos + pos -> pos
      bv_false.concat(fp_add_value),
    Expr::mkIte(((getSignBit(f1) == bv_true) & (getSignBit(f2) == bv_true)),
      // neg + neg -> neg
      bv_true.concat(fp_add_value),
    Expr::mkIte(getMagnitudeBits(f1) == getMagnitudeBits(f2),
      // x + -x -> 0.0
      zero(),
      fp_add_res
  ))))))))));
}

Expr AbsFpEncoding::mul(const Expr &_f1, const Expr &_f2) {
  usedOps.fpMul = true;

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
  const auto f1_nosign = getMagnitudeBits(f1);
  const auto f2_nosign = getMagnitudeBits(f2);

  // Encode commutativity of mul.
  // To avoid that the LSB of mul(x, x) is always 0, encode separately.
  auto mul_abs = Expr::mkIte(f1_nosign == f2_nosign,
    getMulFn().apply({f1_nosign, f1_nosign}),
    getMulFn().apply({f1_nosign, f2_nosign}) +
        getMulFn().apply({f2_nosign, f1_nosign})
  );
  // getMulFn()'s range is BV[VALUE_BITS] because it encodes absolute size of mul.
  // We zero-extend 1 bit (SIGN-BIT) which is actually a dummy bit.
  auto mul_abs_res = mul_abs.zext(1);
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
  Expr::mkIte(isinf(f1, false) | isinf(f1, true),
    Expr::mkIte(iszero(f2, false) | iszero(f2, true), fp_nan, fp_inf_pos),
  // +-0.0 * +-Inf -> NaN , x * +-Inf -> ?Inf (if x != 0.0)
  // IEEE 754-2019 section 7.2 'Invalid operation'
  Expr::mkIte(isinf(f2, false) | isinf(f2, true),
    Expr::mkIte(iszero(f1, false) | iszero(f1, true), fp_nan, fp_inf_pos),
  // +-0.0 * x -> ?0.0, x * +-0.0 -> ?0.0
  Expr::mkIte(iszero(f1, false) | iszero(f1, true) | iszero(f2, false) |
              iszero(f2, true), 
    zero(),
    // If both operands do not fall into any of the cases above,
    // use fp_mul for abstract representation.
    mul_abs_res
  )))))));

  // And at last we replace the sign with signbit(f1) ^ signbit(f2)
  // pos * pos | neg * neg -> pos, pos * neg | neg * pos -> neg
  return Expr::mkIte(fpmul_res == fp_nan, fp_nan,
    Expr::mkIte(getSignBit(f1) == getSignBit(f2),
      bv_false.concat(getMagnitudeBits(fpmul_res)),
      bv_true.concat(getMagnitudeBits(fpmul_res))
  ));
}

Expr AbsFpEncoding::div(const Expr &_f1, const Expr &_f2) {
  usedOps.fpDiv = true;

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
  const auto f1_nosign = getMagnitudeBits(f1);
  const auto f2_nosign = getMagnitudeBits(f2);

  auto div_abs = getDivFn().apply({f1_nosign, f2_nosign});
  // getDivFn()'s range is BV[VALUE_BITS] because it encodes absolute size of mul.
  // We zero-extend 1 bit (SIGN-BIT) which is actually a dummy bit.
  auto div_abs_res = div_abs.zext(1);
  // Absolute size of mul cannot be NaN
  // (0.0 / 0.0, Inf / Inf, and div between one or more NaN
  // will be special-cased).
  div_abs_res = Expr::mkIte(isnan(div_abs_res), fp_id, div_abs_res);

  // Calculate the absolute value of f1 / f2.
  // The sign bit(s) will be replaced in the next step,
  // so it is better to completely ignore the signs in this step.
  // (This is why there's so many | in the conditions...)
  // 
  // x / 1.0 -> x, x / -1.0 -> -x
  auto fpdiv_res = Expr::mkIte((f2 == fp_id) | (f2 == fp_minusone), f1,
  // NaN / x -> NaN
  Expr::mkIte(f1 == fp_nan, f1,
  // x / NaN -> NaN
  Expr::mkIte(f2 == fp_nan, f2,
  // +-Inf / +-Inf -> NaN, x / +-Inf -> ?0.0 (if x != Inf)
  // IEEE 754-2019 section 7.2 'Invalid operation'
  // IEEE 754-2019 section 6.1 'Infinity arithmetic'
  Expr::mkIte(isinf(f2, false) | isinf(f2, true),
    Expr::mkIte(isinf(f1, false) | isinf(f1, true), fp_nan, fp_zero_pos),
  // +-Inf / x -> ?Inf (if x != Inf)
  // IEEE 754-2019 section 6.1 'Infinity arithmetic'
  Expr::mkIte(isinf(f1, false) | isinf(f1, true), fp_inf_pos,
  // +-0.0 / +-0.0 -> NaN, x / +-0.0 -> ?Inf (if x != 0.0 | Inf)
  // IEEE 754-2019 section 7.2 'Invalid operation'
  // IEEE 754-2019 section 7.3 'Division by zero'
  // Division by zero should explicitly signal exception.
  // However, LLVM chooses to simply continue the execution without notifying
  Expr::mkIte(iszero(f2, false) | iszero(f2, true),
    Expr::mkIte(iszero(f1, false) | iszero(f1, true), fp_nan, fp_inf_pos),
    // If both operands do not fall into any of the cases above,
    // use fp_div for abstract representation.
    div_abs_res
  ))))));

  // And at last we replace the sign with signbit(f1) ^ signbit(f2)
  // pos / pos | neg / neg -> pos, pos / neg | neg / pos -> neg
  return Expr::mkIte(fpdiv_res == fp_nan, fp_nan,
    Expr::mkIte(f1.getMSB() == f2.getMSB(),
      bv_false.concat(getMagnitudeBits(fpdiv_res)),
      bv_true.concat(getMagnitudeBits(fpdiv_res))
  ));
}

Expr AbsFpEncoding::lambdaSum(const smt::Expr &a, const smt::Expr &n) {
  usedOps.fpSum = true;

  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);
  Expr result = getSumFn()(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(n),
        Expr::mkIte(isnan(ai), nan(), ai), zero(true))));

  if (getFpAddAssociativity())
    fp_sum_relations.push_back({a, n, result});

  return result;
}

Expr AbsFpEncoding::multisetSum(const Expr &a, const Expr &n) {
  usedOps.fpSum = true;

  uint64_t length;
  if (!n.isUInt(length))
    throw UnsupportedException(
        "Only an array of constant length is supported.");

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
    throw UnsupportedException(
        "Only an array of constant length is supported.");
  auto length = n.asUInt();

  optional<Expr> sumExpr;
  if (fpAddSum == AbsFpAddSumEncoding::USE_SUM_ONLY
      || fpAddSum == AbsFpAddSumEncoding::DEFAULT) {
    sumExpr = (getFpAddAssociativity() && useMultiset) ?
        multisetSum(a, n) :  lambdaSum(a, n);
  } else {
    if (!length || length > maxUnrollFpSumBound) {
      verbose("fpSum") << "ADD_ONLY applies only array length less than or"
                          " equals to " << maxUnrollFpSumBound << ".\n";
      verbose("fpSum") << "Fallback to lambdaSum...\n";
      sumExpr = lambdaSum(a, n);
    } else {
      verbose("fpSum") << "Sum of an array unrolled to fp_add.\n";
      auto sum = a.select(Index(0));
      for (auto i = 1; i < length; i++) {
        sum = add(sum, a.select(Index(i)));
        sum = sum.simplify();
      }
      sumExpr = sum;
    }
  }
  
  auto ret = Expr::mkIte(n == Index::zero(), zero(true),
      Expr::mkIte(n == Index::one(), a.select(Index(0)), *sumExpr));
  ret = ret.simplify();
  return ret;
}


Expr AbsFpEncoding::exp(const Expr &x) {
  // A very simple model. :)
  return Expr::mkIte(
      isnan(x) | (x == infinity()), x,
      Expr::mkIte(x == infinity(true), zero(),
      Expr::mkIte((x == zero()) | x == zero(true), one(),
      getExpFn().apply(x))));
}

Expr AbsFpEncoding::dot(const Expr &a, const Expr &b, const Expr &n) {
  if (alFpDot == AbsLevelFpDot::FULLY_ABS) {
    usedOps.fpDot = true;
    auto i = (Expr)Index::var("idx", VarType::BOUND);

    Expr ai = a.select(i), bi = b.select(i);
    Expr identity = zero(true);
    // Encode commutativity: dot(a, b) = dot(b, a)
    Expr arr1 = Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity));
    Expr arr2 = Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity));
    Expr lhs = getDotFn().apply({arr1, arr2});
    Expr rhs = getDotFn().apply({arr2, arr1});
    // To prevent the LSB of dot(x, x) from always being 0, separately deal with
    // the case using 'arr1 == arr2'.
    return Expr::mkIte(arr1 == arr2, lhs, lhs + rhs);

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
  return getUltFn().apply({f1, f2});
}

Expr AbsFpEncoding::extend(const smt::Expr &f, aop::AbsFpEncoding &tgt) {
  usedOps.fpCastRound = true;

  if (!getFpCastIsPrecise()) {
    // Fully abstract encoding 
    return getExtendFn(tgt).apply(f);
  }

  assert(value_bitwidth < tgt.value_bitwidth &&
         "tgt cannot have smaller value_bitwidth than src");

  if (value_bit_info.limit_bitwidth != 0 || value_bit_info.prec_bitwidth != 0)
    throw UnsupportedException("Casting from middle-size type to large-size "
        "type is not supported");

  auto sign_bit = getSignBit(f);
  auto limit_zero = Expr::mkBV(0, tgt.value_bit_info.limit_bitwidth);
  auto value_bits = getMagnitudeBits(f);
  
  auto extended_float = sign_bit.concat(limit_zero).concat(value_bits);
  if (tgt.value_bit_info.prec_bitwidth > 0) {
    auto prec_zero = Expr::mkBV(0, tgt.value_bit_info.prec_bitwidth);
    extended_float = extended_float.concat(prec_zero);
  }
      
  assert(extended_float.bitwidth() == tgt.sort().bitwidth());
  return Expr::mkIte(isnan(f), tgt.nan(),
      Expr::mkIte(f == infinity(), tgt.infinity(),
      Expr::mkIte(f == infinity(true), tgt.infinity(true),
      extended_float)));
}

Expr AbsFpEncoding::truncate(const smt::Expr &f, aop::AbsFpEncoding &tgt) {
  usedOps.fpCastRound = true;

  if (!getFpCastIsPrecise()) {
    // Fully abstract encoding
    return getTruncateFn(tgt).apply(f);
  }

  assert(value_bitwidth > tgt.value_bitwidth &&
        "tgt cannot have bigger value_bitwidth than src");

  if (tgt.value_bit_info.limit_bitwidth != 0 ||
      tgt.value_bit_info.prec_bitwidth != 0)
    throw UnsupportedException(
      "Truncating from large-size type to middle-size type is not supported");

  auto sign_bit = getSignBit(f);
  auto sign_pos = Expr::mkBV(0, SIGN_BITS);
  auto value_bits = getMagnitudeBits(f);
  auto limit_bits = getLimitBits(f);
  auto limit_zero = Expr::mkBV(0, limit_bits);
  auto prec_bits = getPrecisionBits(f);

  const auto round_dir = getRoundDirFn().apply(value_bits);
  const auto floored_value = getTruncatedBits(value_bits);
  const auto ceiled_value = floored_value + 1;

  const auto floored_float = sign_bit.concat(floored_value);
  const auto ceiled_float = sign_bit.concat(ceiled_value);
  assert(floored_float.bitwidth() == tgt.sort().bitwidth());
  const auto is_prec_zero = prec_bits ? *prec_bits == 0 : Expr::mkBool(true);

  return Expr::mkIte(isnan(f), tgt.nan(),
          Expr::mkIte(f == infinity(), tgt.infinity(),
          Expr::mkIte(f == infinity(true), tgt.infinity(true),
          Expr::mkIte(limit_bits != limit_zero,
            Expr::mkIte(sign_bit == sign_pos,
              tgt.infinity(), tgt.infinity(true)),
            Expr::mkIte(is_prec_zero, floored_float,
              Expr::mkIte(round_dir == Expr::mkBV(0, 1),
                floored_float, ceiled_float))))));
}

Expr AbsFpEncoding::getFpAssociativePrecondition() {
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

  vector<optional<Expr>> hashValues(fp_sum_relations.size());
  for (unsigned i = 0; i < fp_sum_relations.size(); i ++) {
    auto [a, an, asum] = fp_sum_relations[i];
    uint64_t alen;
    if (!an.isUInt(alen)) continue;

    auto hashfn = getHashFnForAddAssoc();
    auto aVal = Expr::mkBV(0, getHashRangeBits());

    for (unsigned j = 0; j < alen; j ++) {
      auto elem = (a.select(Index(j))).simplify();
      optional<Expr> current;
      for (unsigned k = 0; k < i; k ++) {
        auto [b, bn, bsum] = fp_sum_relations[k];
        auto other = bsum.simplify();
        if (elem.isIdentical(other)) current = hashValues[k];
      }
      aVal = aVal + current.value_or(hashfn.apply(elem));
    }
    hashValues[i] = aVal;
  }

  // precondition between `hashfn <-> sumfn`
  Expr precond = Expr::mkBool(true);
  for (unsigned i = 0; i < fp_sum_relations.size(); i ++) {
    for (unsigned j = i + 1; j < fp_sum_relations.size(); j ++) {
      auto [a, an, asum] = fp_sum_relations[i];
      auto [b, bn, bsum] = fp_sum_relations[j];
      uint64_t alen, blen;
      if (!an.isUInt(alen) || !bn.isUInt(blen)) continue;
      // if addf, sumfn are repective, we only consider same length array
      if (fpAddSum == AbsFpAddSumEncoding::DEFAULT && alen != blen) continue;

      auto aVal = *hashValues[i];
      auto bVal = *hashValues[j];
      // precond: sumfn(A) != sumfn(B) -> hashfn(A) != hashfn(B)
      // This means if two summations are different, we can find concrete hash function that hashes into different value.
      auto associativity = (!(asum == bsum)).implies(!(aVal == bVal));
      precond = precond & associativity;
    }
  }

  // To support summation without identity equals to orginal one sum([a])=sum([a, 0, 0])
  // add  precondition for hash(-0) = 0
  auto hashfn = getHashFnForAddAssoc();
  auto fpAddIdentity = zero(true);
  auto hashIdentity = Expr::mkBV(0, getHashRangeBits());
  precond = precond & (hashfn.apply(fpAddIdentity) == hashIdentity);

  precond = precond.simplify();
  return precond;
}

Expr AbsFpEncoding::getFpTruncatePrecondition(aop::AbsFpEncoding &tgt) {
  Expr precond = Expr::mkBool(true);

  // keep track of value_id of fp_const_sval on our own
  uint64_t value_id = 0;
  bool loses_info; // dummy
  auto prev_tgt_fp = llvm::APFloat(0.0f);
  for (const auto &[fp, absrepr] : fpconst_absrepr) {
    // only value bits are relevant in truncation
    if (fp.isNegative())
      continue;

    const auto casting_info = *getCastingInfo(fp);
    if (casting_info.zero_prec_bits) {
      auto tgt_fp = fp;
      tgt_fp.convert(tgt.semantics, llvm::APFloat::rmTowardZero, &loses_info);
      bool isEpsilon = 
        (tgt_fp.bitcastToAPInt() - prev_tgt_fp.bitcastToAPInt()).isOne();

      if (isEpsilon) {
        // remove the gap between two adjacent values
        const auto sv_bitwidth = value_bit_info.truncated_bitwidth;
        const auto prev_var = Expr::mkVar(Sort::bvSort(sv_bitwidth),
                                "fp_const_sval_" + to_string(value_id) + "_");
        const auto var = Expr::mkVar(Sort::bvSort(sv_bitwidth), 
                          "fp_const_sval_" + to_string(value_id + 1) + "_");
        precond &= var == (prev_var + 1);

        verbose("getFpTruncatePrecondition") << "("
            << prev_tgt_fp.convertToDouble() << ", "
            << tgt_fp.convertToDouble() << "): " << var << " == "
            << (prev_var + 1) << '\n';
      }
      value_id += 1;
      prev_tgt_fp = tgt_fp;
    } else {
      const auto value_bits = getMagnitudeBits(absrepr);
      if (casting_info.is_rounded_upward) {
        auto e = (getRoundDirFn().apply({value_bits}) == Expr::mkBV(1, 1));
        verbose("getFpTruncatePrecondition") << fp.convertToDouble() << ": "
            << e << '\n';

        precond &= e;
      } else {
        auto e = (getRoundDirFn().apply({value_bits}) == Expr::mkBV(0, 1));
        verbose("getFpTruncatePrecondition") << fp.convertToDouble() << ": "
            << e << '\n';

        precond &= e;
      }
    }
  }

  return precond;
}

Expr AbsFpEncoding::getFpConstantPrecondition() {
  Expr precond = Expr::mkBool(true);
  
  auto prev_fp = llvm::APFloat::getInf(semantics, true);
  auto prev_absrepr = infinity(true);
  bool firstItr = true;

  for (const auto &[fp, absrepr] : fpconst_absrepr) {
    assert(!fp.isInfinity() && !fp.isZero());

    if (!fp.isNegative()) {
      // SMT encoding of x and -x is equivalent modulo sign bit; exit early
      break;
    }

    precond &= prev_absrepr.ugt(absrepr);
    verbose("getFpConstantPrecondition") << prev_fp.convertToDouble()
        << " < " << fp.convertToDouble() << ": "
        << absrepr << " < " << prev_absrepr << "\n";

    prev_fp = fp;
    prev_absrepr = absrepr;
    firstItr = false;
  }
  if (!fpconst_absrepr.empty()) {
    precond &= prev_absrepr.ugt(zero(true));
    verbose("getFpConstantPrecondition") << prev_fp.convertToDouble()
        << " < -0.0: " << zero(true) << " < " << prev_absrepr << "\n";
  }

  return precond.simplify();
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

Expr getFpTruncatePrecondition() {
  // Calling this function doesn't make sense if casting is imprecise
  assert(alFpCast == AbsLevelFpCast::PRECISE);

  // if alFpCast is true, floatEnc and doubleEnc will exist
  Expr cond = doubleEnc->getFpTruncatePrecondition(*floatEnc);
  return cond;
}

Expr AbsFpEncoding::getSignBit(const smt::Expr &f) const {
  assert(fp_bitwidth - value_bitwidth == SIGN_BITS);
  return f.extract(fp_bitwidth - 1, value_bitwidth);
}

Expr AbsFpEncoding::getMagnitudeBits(const smt::Expr &f) const {
  return f.extract(value_bitwidth - 1, 0);
}

Expr AbsFpEncoding::getLimitBits(const smt::Expr &f) const {
  assert(value_bit_info.limit_bitwidth > 0);
  return f.extract(value_bitwidth - 1,
      value_bitwidth - value_bit_info.limit_bitwidth);
}

Expr AbsFpEncoding::getTruncatedBits(const smt::Expr &f) const {
  unsigned hw =
      value_bit_info.prec_bitwidth + value_bit_info.truncated_bitwidth - 1;
  return f.extract(hw, value_bit_info.prec_bitwidth);
}

optional<Expr> AbsFpEncoding::getPrecisionBits(const smt::Expr &f) const {
  if (value_bit_info.prec_bitwidth == 0)
    return nullopt;
  return f.extract(value_bit_info.prec_bitwidth - 1, 0);
}

Expr getFpConstantPrecondition() {
  Expr cond = Expr::mkBool(true);

  if (floatEnc) {
    cond &= floatEnc->getFpConstantPrecondition();
  }
  if (doubleEnc) {
    cond &= doubleEnc->getFpConstantPrecondition();
  }

  return cond.simplify();
}

// ----- Integer operations ------


Expr intSum(const Expr &a, const Expr &n) {
  auto i = Index::var("idx", VarType::BOUND);
  Expr ai = a.select(i);

  uint64_t n_const;
  if (doUnrollIntSum && n.isUInt(n_const)) {
    verbose("intSum") << "Unrolling sum whose size is " << n_const << "\n";
    Expr s = Expr::mkBV(0, ai.bitwidth());
    for (uint64_t j = 0; j < n_const; ++j) {
      s = s + a.select(Index(j));
    }
    return s;
  }

  usedOps.intSum = true;
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
