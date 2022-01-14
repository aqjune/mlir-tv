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
aop::Abstraction abstraction;

// ----- Constants and global vars for abstract floating point operations ------

bool isFpAddAssociative;
bool doUnrollIntSum;
bool hasArithProperties;
unsigned maxUnrollFpSumBound;

optional<aop::AbsFpEncoding> floatEnc;
optional<aop::AbsFpEncoding> doubleEnc;

// ----- Constants and global vars for abstract int operations ------

map<unsigned, FnDecl> int_sumfn;
map<unsigned, FnDecl> int_dotfn;

// ----- Constants and global vars for abstract sumf operations ------

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


pair<Expr, Expr> insertInitialValue(const Expr &a, const Expr &n,
    const Expr &initValue) {
  auto i = (Expr) Index::var("idx", VarType::BOUND);
  auto ai = a.select(i - 1);
  auto arr = Expr::mkLambda(i, Expr::mkIte(i.isZero(), initValue, ai));
  auto size = (Expr) n + 1;
  return {arr, size};
}

}


namespace aop {

UsedAbstractOps getUsedAbstractOps() { return usedOps; }

void setAbstraction(
    Abstraction abs,
    bool addAssoc,
    bool unrollIntSum,
    bool noArithProperties,
    unsigned unrollFpSumBound,
    unsigned floatNonConstsCnt, set<llvm::APFloat> floatConsts,
    unsigned doubleNonConstsCnt, set<llvm::APFloat> doubleConsts) {
  abstraction = abs;
  doUnrollIntSum = unrollIntSum;
  maxUnrollFpSumBound = unrollFpSumBound;
  hasArithProperties = !noArithProperties;
  isFpAddAssociative = addAssoc;

  assert(!addAssoc ||
      abs.fpAddSumEncoding == AbsFpAddSumEncoding::USE_SUM_ONLY);

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
  // Should not exceed 31 (limited by real-life float)
  unsigned floatBits =
      min((uint64_t) 31, log2_ceil(floatNonConstsCnt + floatConsts.size() + 2));
  floatEnc.emplace(llvm::APFloat::IEEEsingle(), floatBits, "float");
  floatEnc->addConstants(floatConsts);

  if (abstraction.fpCast == AbsLevelFpCast::PRECISE) {
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
    // doubleBits must be at least as large as floatBits, to represent all
    // float constants in double.
    const unsigned doubleBits = 
        max(log2_ceil(doubleNonConstsCnt + doubleConsts.size() + 2),
            (uint64_t)floatBits);
    doubleEnc.emplace(llvm::APFloat::IEEEdouble(), doubleBits, "double");
  }
  doubleEnc->addConstants(doubleConsts);
}

// A set of options that must not change the precision of validation.
void setEncodingOptions(bool use_multiset) {
  useMultiset = use_multiset;
}

bool getFpAddAssociativity() { return isFpAddAssociative; }

bool getFpCastIsPrecise() { 
  return abstraction.fpCast == AbsLevelFpCast::PRECISE;
}

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
  fp_hashfn.reset();
  fp_sums.clear();
  fp_pooling_sumfn.reset();
  fp_pooling_maxfn.reset();
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
    auto fty = hasArithProperties ? Sort::bvSort(value_bitwidth) : sort();
    fp_mulfn.emplace({fty, fty}, fty, "fp_mul_" + fn_suffix);
  }
  return *fp_mulfn;
}

FnDecl AbsFpEncoding::getDivFn() {
  if (!fp_mulfn) {
    auto fty = hasArithProperties ? Sort::bvSort(value_bitwidth) : sort();
    fp_mulfn.emplace({fty, fty}, fty, "fp_div_" + fn_suffix);
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
    // (initial Value, array1, array2)
    fp_dotfn.emplace({sort(), arrs, arrs}, sort(), "fp_dot_" + fn_suffix);
  return *fp_dotfn;
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

FnDecl AbsFpEncoding::getPoolingSumFn() {
  if (!fp_pooling_sumfn) {
    auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
    auto attr = Index::sort();
    // (Input, Kernel_Y Kernel_X, Stride_Y, Stride_X)
    fp_pooling_sumfn.emplace({arrs, attr, attr, attr, attr},
        arrs, "fp_pooling_sum_" + fn_suffix);
  }
  return *fp_pooling_sumfn;
}

FnDecl AbsFpEncoding::getPoolingMaxFn() {
  if (!fp_pooling_maxfn) {
    auto arrs = Sort::arraySort(Index::sort(), sort()).toFnSort();
    auto attr = Index::sort();
    // (Input, Kernel_Y Kernel_X, Stride_Y, Stride_X)
    fp_pooling_maxfn.emplace({arrs, attr, attr, attr, attr},
        arrs, "fp_pooling_max_" + fn_suffix);
  }
  return *fp_pooling_maxfn;
}

size_t AbsFpEncoding::getHashRangeBits() const {
  uint64_t numSums = fp_sums.size();
  uint64_t maxLength = 0;

  for (auto &rel: fp_sums) {
    maxLength = max(maxLength, rel.len);
  }

  uint64_t bounds = numSums * numSums * maxLength;
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

        auto inf_sv_value = (1 << sv_prec_bitwidth) -
                            (1 << value_bit_info.prec_bitwidth);
        auto limit_var = Expr::mkFreshVar(limit_bits, "fp_const_limit_bits_");
        limit_bits = Expr::mkIte(sv_prec_bits.uge(inf_sv_value), 
          limit_var,
          mkNonzero(move(limit_var)));
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

void AbsFpEncoding::evalConsts(smt::Model model) {
  for (auto &[k, v]: fpconst_absrepr) {
    v = model.eval(v);
  }

  if (fpconst_nan) {
    *fpconst_nan = model.eval(*fpconst_nan);
  } else if (fpconst_zero_pos) {
    *fpconst_zero_pos = model.eval(*fpconst_zero_pos);
  } else if (fpconst_zero_neg) {
    *fpconst_zero_neg = model.eval(*fpconst_zero_neg);
  } else if (fpconst_inf_pos) {
    *fpconst_inf_pos = model.eval(*fpconst_inf_pos);
  } else if (fpconst_inf_neg) {
    *fpconst_inf_neg = model.eval(*fpconst_inf_neg);
  }
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
  if (abstraction.fpAddSumEncoding == AbsFpAddSumEncoding::USE_SUM_ONLY) {
    auto i = Index::var("idx", VarType::BOUND);
    auto lambda = Expr::mkLambda(i, Expr::mkIte(i == Index::zero(), _f1, _f2));
    auto n = Index(2);
    return sum(lambda, n, {{_f1, _f2}});
  }

  usedOps.fpAdd = true;

  if (!hasArithProperties)
    return getAddFn().apply({_f1, _f2});

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
  auto fp_add_res = getAddFn().apply({f1, f2}) & getAddFn().apply({f2, f1});
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

  if (!hasArithProperties)
    return getMulFn().apply({_f1, _f2});

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
  auto mul_abs = getMulFn().apply({f1_nosign, f2_nosign}) &
                 getMulFn().apply({f2_nosign, f1_nosign});
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

  if (!hasArithProperties)
    return getDivFn().apply({_f1, _f2});

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
  // +-0.0 / x -> ?0.0, 
  Expr::mkIte(iszero(f1, false) | iszero(f1, true), fp_zero_pos,
  // x / x -> 1.0, x / -x -> -1.0 
  Expr::mkIte(f1_nosign == f2_nosign, fp_id,
  // If both operands do not fall into any of the cases above,
  // use fp_div for abstract representation.
  div_abs_res
  ))))))));

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

  uint64_t len;
  if (getFpAddAssociativity() && n.isUInt(len))
    fp_sums.push_back({a, {}, len, result});

  return result;
}

Expr AbsFpEncoding::lambdaSum(const vector<smt::Expr> &elems) {
  usedOps.fpSum = true;

  assert(elems.size() == 2 && "Currently supports sum of two elems only");

  auto i = Index::var("idx", VarType::BOUND);
  auto lambda = Expr::mkLambda(i, Expr::mkIte(i == 0, elems[0], elems[1]));
  Expr result = getSumFn()(lambda);

  if (getFpAddAssociativity())
    fp_sums.push_back({lambda, elems, elems.size(), result});

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
  fp_sums.push_back({bag, {}, length, result});

  return result;
}

Expr AbsFpEncoding::sum(const Expr &a, const Expr &n,
    optional<vector<smt::Expr>> &&elems,
    optional<smt::Expr> &&initValue) {
  if (getFpAddAssociativity() && !n.isNumeral())
    throw UnsupportedException(
        "Only an array of constant length is supported.");

  // If initValue is a non-identity value, add it to the beginning of the array.
  bool insertInitVal = initValue.has_value();
  if (insertInitVal)
    insertInitVal &= !((*initValue == zero(true)).isTrue());

  auto [arr, size] = insertInitVal ?
      insertInitialValue(a, n, *initValue) : make_pair(a, n);
  if (insertInitVal && elems.has_value()) {
    elems->insert(elems->begin(), *initValue);
  }

  auto length = size.asUInt();
  if (elems) {
    assert(length == elems->size());
  }

  optional<Expr> sumExpr;
  if (abstraction.fpAddSumEncoding == AbsFpAddSumEncoding::USE_SUM_ONLY
      || abstraction.fpAddSumEncoding == AbsFpAddSumEncoding::DEFAULT) {
    if (getFpAddAssociativity() && useMultiset)
      sumExpr = multisetSum(arr, size);
    else {
      if (elems)
        sumExpr = lambdaSum(*elems);
      else
        sumExpr = lambdaSum(arr, size);
    }
  } else {
    if (!length || length > maxUnrollFpSumBound) {
      verbose("fpSum") << "ADD_ONLY applies only array length less than or"
                          " equals to " << maxUnrollFpSumBound << ".\n";
      verbose("fpSum") << "Fallback to lambdaSum...\n";
      sumExpr = lambdaSum(arr, size);
    } else {
      verbose("fpSum") << "Sum of an array unrolled to fp_add.\n";
      auto sum = arr.select(Index(0));
      for (auto i = 1; i < length; i++) {
        sum = add(sum, arr.select(Index(i)));
        sum = sum.simplify();
      }
      sumExpr = sum;
    }
  }
  
  auto ret = Expr::mkIte(size == Index::zero(), zero(true),
      Expr::mkIte(size == Index::one(), arr.select(Index(0)), *sumExpr));
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

Expr AbsFpEncoding::dot(const Expr &a, const Expr &b,
    const Expr &n, std::optional<smt::Expr> &&initValue) {
  if (abstraction.fpDot == AbsLevelFpDot::FULLY_ABS) {
    usedOps.fpDot = true;
    auto i = (Expr)Index::var("idx", VarType::BOUND);

    Expr ai = a.select(i), bi = b.select(i);
    Expr identity = zero(true);
    // Encode commutativity: dot(a, b) = dot(b, a)
    Expr arr1 = Expr::mkLambda(i, Expr::mkIte(i.ult(n), ai, identity));
    Expr arr2 = Expr::mkLambda(i, Expr::mkIte(i.ult(n), bi, identity));

    return getDotFn().apply({initValue.value_or(identity), arr1, arr2}) &
      getDotFn().apply({initValue.value_or(identity), arr2, arr1});

  } else if (abstraction.fpDot == AbsLevelFpDot::SUM_MUL) {
    // usedOps.fpMul/fpSum will be updated by the fpMul()/fpSum() calls below
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, mul(ai, bi));

    return sum(arr, n, nullopt, move(initValue));
  }
  llvm_unreachable("Unknown abstraction level for fp dot");
}

Expr AbsFpEncoding::cmp(mlir::arith::CmpFPredicate pred,
                        const Expr &f1, const Expr &f2) {
  const Expr trueBV = Expr::mkBV(1, 1);
  const Expr falseBV = Expr::mkBV(0, 1);

  const Expr hasNaN = isnan(f1) | isnan(f2);

  const Expr f1Sign = getSignBit(f1), f2Sign = getSignBit(f2);
  const Expr f1Magn = getMagnitudeBits(f1), f2Magn = getMagnitudeBits(f2);

  const Expr cmpEQ = Expr::mkIte(f1 == f2 | (f1Magn.isZero() & f2Magn.isZero()),
                        trueBV, falseBV);
  const Expr cmpNE = Expr::mkIte(f1 != f2 & !(f1Magn.isZero() & f2Magn.isZero()),
                        trueBV, falseBV);
  const Expr cmpLT = Expr::mkIte(f1Sign.ugt(f2Sign), trueBV,
                      Expr::mkIte(f1Sign.ult(f2Sign), falseBV,
                      Expr::mkIte(f1Magn == f2Magn, falseBV,
                      Expr::mkIte(f1Magn.ult(f2Magn) ^ f1Sign.isNonZero(),
                        trueBV, falseBV))));
  const Expr cmpGT = Expr::mkIte(f1Sign.ult(f2Sign), trueBV,
                      Expr::mkIte(f1Sign.ugt(f2Sign), falseBV,
                      Expr::mkIte(f1Magn == f2Magn, falseBV,
                      Expr::mkIte(f1Magn.ugt(f2Magn) ^ f1Sign.isNonZero(),
                        trueBV, falseBV))));

  switch (pred) {
  case mlir::arith::CmpFPredicate::OEQ:
    return Expr::mkIte(hasNaN, falseBV, cmpEQ);
  case mlir::arith::CmpFPredicate::ONE:
    return Expr::mkIte(hasNaN, falseBV, cmpNE);
  case mlir::arith::CmpFPredicate::OLE:
    return Expr::mkIte(hasNaN, falseBV, cmpEQ | cmpLT);
  case mlir::arith::CmpFPredicate::OLT:
    return Expr::mkIte(hasNaN, falseBV, cmpLT);
  case mlir::arith::CmpFPredicate::OGE:
    return Expr::mkIte(hasNaN, falseBV, cmpEQ | cmpGT);
  case mlir::arith::CmpFPredicate::OGT:
    return Expr::mkIte(hasNaN, falseBV, cmpGT);
  case mlir::arith::CmpFPredicate::UEQ:
    return Expr::mkIte(hasNaN, trueBV, cmpEQ);
  case mlir::arith::CmpFPredicate::UNE:
    return Expr::mkIte(hasNaN, trueBV, cmpNE);
  case mlir::arith::CmpFPredicate::ULE:
    return Expr::mkIte(hasNaN, trueBV, cmpEQ | cmpLT);
  case mlir::arith::CmpFPredicate::ULT:
    return Expr::mkIte(hasNaN, trueBV, cmpLT);
  case mlir::arith::CmpFPredicate::UGE:
    return Expr::mkIte(hasNaN, trueBV, cmpEQ | cmpGT);
  case mlir::arith::CmpFPredicate::UGT:
    return Expr::mkIte(hasNaN, trueBV, cmpGT);
  case mlir::arith::CmpFPredicate::ORD:
    return Expr::mkIte(hasNaN, falseBV, trueBV);
  case mlir::arith::CmpFPredicate::UNO:
    return Expr::mkIte(hasNaN, trueBV, falseBV);
  case mlir::arith::CmpFPredicate::AlwaysTrue:
    return trueBV;
  case mlir::arith::CmpFPredicate::AlwaysFalse:
    return falseBV;
  default:
    throw UnsupportedException("Invalid cmpf predicate");
  }
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

  const auto is_truncated_value_inf_or_nan =
              floored_value.uge(tgt.getMagnitudeBits(tgt.infinity()));
  return Expr::mkIte(isnan(f), tgt.nan(),
          Expr::mkIte(f == infinity(), tgt.infinity(),
          Expr::mkIte(f == infinity(true), tgt.infinity(true),
          Expr::mkIte(limit_bits != limit_zero |
                      is_truncated_value_inf_or_nan,
            Expr::mkIte(sign_bit == sign_pos,
              tgt.infinity(), tgt.infinity(true)),
            Expr::mkIte(is_prec_zero, floored_float,
              Expr::mkIte(round_dir == Expr::mkBV(0, 1),
                floored_float, ceiled_float))))));
}

Expr AbsFpEncoding::avgPool(const Expr &arr,
    const Expr &kernelY, const Expr &kernelX,
    const Expr &strideY, const Expr &strideX) {
  return getPoolingSumFn().apply({arr, kernelY, kernelX, strideY, strideX});
}

Expr AbsFpEncoding::maxPool(const Expr &arr,
    const Expr &kernelY, const Expr &kernelX,
    const Expr &strideY, const Expr &strideX) {
  return getPoolingMaxFn().apply({arr, kernelY, kernelX, strideY, strideX});
}

Expr AbsFpEncoding::getFpAssociativePrecondition() {
  if (useMultiset) {
    // precondition between `bag equality <-> assoc_sumfn`
    Expr precond = Expr::mkBool(true);
    for (unsigned i = 0; i < fp_sums.size(); i ++) {
      for (unsigned j = i + 1; j < fp_sums.size(); j ++) {
        auto [abag, aelems, alen, asum] = fp_sums[i];
        auto [bbag, belems, blen, bsum] = fp_sums[j];

        precond = precond & (abag == bbag).implies(asum == bsum);
      }
    }

    // precondition for bags union
    for (unsigned i = 0; i < fp_sums.size(); i ++) {
      for (unsigned j = 0; j < i; j ++) {
        for (unsigned k = j + 1; k < i; k ++) {
          auto [abag, aelems, alen, asum] = fp_sums[i];
          auto [bbag, belems, blen, bsum] = fp_sums[j];
          auto [cbag, celems, clen, csum] = fp_sums[k];

          if (alen != blen + clen) continue;
          auto precondition = (bbag.bagUnion(cbag) == abag)
            .implies(add(bsum, csum) == asum);
          precond = precond & precondition;
        }
      }
    }

    return precond;
  }

  vector<optional<Expr>> hashValues(fp_sums.size());
  auto hashfn = getHashFnForAddAssoc();

  for (unsigned i = 0; i < fp_sums.size(); i ++) {
    const auto &[a, aelems, alen, asum] = fp_sums[i];

    auto aVal = Expr::mkBV(0, getHashRangeBits());

    for (unsigned j = 0; j < alen; j ++) {
      auto elem = !aelems.empty() ? aelems[j] : a.select(Index(j));

      optional<Expr> current;
      for (unsigned k = 0; k < i; k ++) {
        const auto &[b, belems, blen, bsum] = fp_sums[k];
        if ((bsum == elem).simplify().isTrue())
          current = hashValues[k];
      }
      aVal = aVal + current.value_or(hashfn.apply(elem));
    }
    hashValues[i] = aVal;
  }

  // precondition between `hashfn <-> sumfn`
  Expr precond = Expr::mkBool(true);
  for (unsigned i = 0; i < fp_sums.size(); i ++) {
    const auto &[a, aelems, alen, asum] = fp_sums[i];

    for (unsigned j = i + 1; j < fp_sums.size(); j ++) {
      const auto &[b, belems, blen, bsum] = fp_sums[j];

      // if addf, sumfn are repective, we only consider same length array
      if (abstraction.fpAddSumEncoding == AbsFpAddSumEncoding::DEFAULT &&
          alen != blen)
        continue;

      auto aVal = *hashValues[i];
      auto bVal = *hashValues[j];
      // precond: sumfn(A) != sumfn(B) -> hashfn(A) != hashfn(B)
      // This means if two summations are different, we can find concrete hash
      // function that hashes into different value.
      auto associativity = (!(asum == bsum)).implies(!(aVal == bVal));
      precond = precond & associativity;
    }
  }

  // To support summation without identity equals to orginal one
  //   sum([a])=sum([a, 0, 0])
  // add a precondition for hash(-0) = 0
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

  if (doubleEnc)
    cond &= doubleEnc->getFpAssociativePrecondition();

  return cond;
}

Expr getFpTruncatePrecondition() {
  // Calling this function doesn't make sense if casting is imprecise
  assert(abstraction.fpCast == AbsLevelFpCast::PRECISE);

  // if fpCast is true, floatEnc and doubleEnc will exist
  Expr cond = doubleEnc->getFpTruncatePrecondition(*floatEnc);
  return cond;
}

void evalConsts(smt::Model model) {
  if (floatEnc)
    floatEnc->evalConsts(model);

  if (doubleEnc)
    doubleEnc->evalConsts(model);
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


Expr intSum(const Expr &a, const Expr &n,
    std::optional<smt::Expr> &&initValue) {
  
  auto [arr, size] = initValue ?
      insertInitialValue(a, n, *initValue) : make_pair(a, n);
  auto i = Index::var("idx", VarType::BOUND);
  Expr arri = arr.select(i);

  uint64_t length;
  if (doUnrollIntSum && size.isUInt(length)) {
    verbose("intSum") << "Unrolling sum whose size is " << length << "\n";
    Expr s = Expr::mkBV(0, arri.bitwidth());
    for (uint64_t j = 0; j < length; ++j) {
      s = s + arr.select(Index(j));
    }
    return s;
  }

  usedOps.intSum = true;
  Expr zero = Integer(0, arri.bitwidth());

  FnDecl sumfn = getIntSumFn(arri.sort().bitwidth());
  Expr result = sumfn(
      Expr::mkLambda(i, Expr::mkIte(((Expr)i).ult(size), arri, zero)));

  return result;
}

Expr intDot(const Expr &a, const Expr &b,
    const Expr &n, std::optional<smt::Expr> &&initValue) {
  if (abstraction.intDot == AbsLevelIntDot::FULLY_ABS) {
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

  } else if (abstraction.intDot == AbsLevelIntDot::SUM_MUL) {
    auto i = (Expr)Index::var("idx", VarType::BOUND);
    Expr ai = a.select(i), bi = b.select(i);
    Expr arr = Expr::mkLambda(i, ai * bi);

    return intSum(arr, n, move(initValue));
  }
  llvm_unreachable("Unknown abstraction level for int dot");
}

}
