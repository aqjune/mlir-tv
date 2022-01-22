#pragma once

#include "smt.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinOps.h"
#include <vector>
#include <set>

namespace aop {

struct UsedAbstractOps {
  // Float ops
  bool fpDot;
  bool fpAdd;
  bool fpMul;
  bool fpDiv;
  bool fpSum;
  bool fpCastRound;
  // Int ops
  bool intDot;
  bool intSum;
};
UsedAbstractOps getUsedAbstractOps();

enum class AbsLevelFpDot {
  FULLY_ABS = 0, // FP Dot is a fully unknown function
  SUM_MUL   = 1, // FP Dot is a summation of pairwisely multiplied values
};

enum class AbsLevelFpCast {
  FULLY_ABS = 0, // FP Cast is a fully unknown function
  PRECISE   = 1, // FP Cast's semantics is precisely encoded
};

enum class AbsLevelIntDot {
  FULLY_ABS = 0, // Int Dot is a fully unknown function
  SUM_MUL   = 1, // Int Dot is a summation of pairwisely multiplied values
};

enum class AbsFpAddSumEncoding {
  USE_SUM_ONLY = 0, // When --associativity is given, encode an addition as sum
  DEFAULT = 1, // Encode addition using fp_add, fp_sum respectivly
               // (no relation between them)
  UNROLL_TO_ADD = 2, // Unroll sum to add if the size of array is small enough
                     // This is more concrete semantics than DEFAULT.
};

struct Abstraction {
  AbsLevelFpDot fpDot;
  AbsLevelFpCast fpCast;
  AbsLevelIntDot intDot;
  AbsFpAddSumEncoding fpAddSumEncoding;
};

// unrollIntSum: Fully unroll sum(arr) where arr is an int array of const size
//               as arr[0] + arr[1] + .. + arr[len-1]?
// unrollFpSumBound: If AbsFpAddSumEncoding is UNROLL_TO_ADD, specify the max.
//                   size of an array to unroll
// floatNonConstsCnt: # of non-constant distinct f32 values necessary to
// validate the transformation.
// NOTE: This resets the used abstract ops record, but does not reset encoding
//    options (see setEncodingOptions).
void setAbstraction(Abstraction abs,
                    bool isFpAddAssociative,
                    bool unrollIntSum,
                    bool noArithProperties,
                    unsigned unrollFpSumBound,
                    unsigned floatNonConstsCnt,
                    std::set<llvm::APFloat> floatConsts,
                    bool floatHasInfOrNaN,
                    unsigned doubleNonConstsCnt,
                    std::set<llvm::APFloat> doubleConsts,
                    bool doubleHasInfOrNaN);
// A set of options that must not change the precision of validation.
// useMultiset: To encode commutativity of fp summation, use multiset?
void setEncodingOptions(bool useMultiset);

bool getFpAddAssociativity();
bool getFpCastIsPrecise();

smt::Expr getFpTruncatePrecondition();
smt::Expr getFpAssociativePrecondition();
smt::Expr getFpConstantPrecondition();

void evalConsts(smt::Model model);

smt::Expr intSum(const smt::Expr &arr, const smt::Expr &n,
    std::optional<smt::Expr> &&initValue = std::nullopt);
smt::Expr intDot(const smt::Expr &arr1, const smt::Expr &arr2,
                 const smt::Expr &n,
                 std::optional<smt::Expr> &&initValue = std::nullopt);

class AbsFpEncoding {
private:
  const llvm::fltSemantics &semantics;

  // NaNs, Infs, and +-0 are stored in separate variable
  // as they do not work well with map due to comparison issue
  std::optional<smt::Expr> fpconst_zero_pos;
  std::optional<smt::Expr> fpconst_zero_neg;
  std::optional<smt::Expr> fpconst_nan;
  std::optional<smt::Expr> fpconst_inf_pos;
  std::optional<smt::Expr> fpconst_inf_neg;
  // float::MIN/MAX are stored in separate variable
  // as they must be reserved a fixed value for correct validation
  std::optional<smt::Expr> fpconst_min; // -float::MAX
  std::optional<smt::Expr> fpconst_max;
  // Abstract representation of valid fp constants (except +-0.0, min, max).
  std::map<llvm::APFloat, smt::Expr> fpconst_absrepr;

  const static unsigned SIGN_BITS = 1;
  // The BV width of abstract fp encoding.
  // fp_bv_bits = SIGN_BITS + value_bv_bits (magnitude)
  unsigned fp_bitwidth;
  unsigned value_bitwidth;

  // Bits for casting.
  struct ValueBitInfo {
    unsigned limit_bitwidth;
    unsigned truncated_bitwidth;
    unsigned prec_bitwidth;

    unsigned get_value_bitwidth() {
      return limit_bitwidth + truncated_bitwidth + prec_bitwidth;
    }
  };
  ValueBitInfo value_bit_info;

  // A summation of an array having static length
  struct FpSumInfo {
    smt::Expr arr;
    // arrElems can be empty.
    std::vector<smt::Expr> arrElems;
    uint64_t len;
    smt::Expr sumExpr;
  };
  std::vector<FpSumInfo> fp_sums;

  // These are lazily created.
  std::optional<smt::FnDecl> fp_sumfn;
  std::optional<smt::FnDecl> fp_assoc_sumfn;
  std::optional<smt::FnDecl> fp_dotfn;
  std::optional<smt::FnDecl> fp_addfn;
  std::optional<smt::FnDecl> fp_mulfn;
  std::optional<smt::FnDecl> fp_divfn;
  std::optional<smt::FnDecl> fp_extendfn;
  std::optional<smt::FnDecl> fp_truncatefn;
  std::optional<smt::FnDecl> fp_expfn;
  std::optional<smt::FnDecl> fp_hashfn;
  std::optional<smt::FnDecl> fp_rounddirfn;
  std::optional<smt::FnDecl> fp_maxfn;
  std::optional<smt::FnDecl> fp_sint32tofp_fn;
  std::string fn_suffix;

private:
  AbsFpEncoding(const llvm::fltSemantics &semantics,
      unsigned limit_bw, unsigned smaller_value_bw, unsigned prec_bw,
      std::string &&fn_suffix);

public:
  AbsFpEncoding(const llvm::fltSemantics &semantics, unsigned value_bw,
      std::string &&fn_suffix)
      : AbsFpEncoding(semantics, 0u, value_bw, 0u, std::move(fn_suffix)) {}
  // Use smaller_fpty_enc's value_bv_bits to calculate this type's value_bv_bits
  AbsFpEncoding(const llvm::fltSemantics &semantics,
      unsigned limit_bw, unsigned prec_bw, AbsFpEncoding* smaller_fpty_enc,
      std::string &&fn_suffix)
      : AbsFpEncoding(semantics, limit_bw, smaller_fpty_enc->value_bitwidth,
        prec_bw, std::move(fn_suffix)) {}
  // Copying this object badly interacts with how CVC5 treats the term objects.
  AbsFpEncoding(const AbsFpEncoding &) = delete;

  smt::Sort sort() const {
    return smt::Sort::bvSort(fp_bitwidth);
  }

private:
  smt::FnDecl getAddFn();
  smt::FnDecl getMulFn();
  smt::FnDecl getDivFn();
  smt::FnDecl getAssocSumFn();
  smt::FnDecl getSumFn();
  smt::FnDecl getDotFn();
  smt::FnDecl getExtendFn(const AbsFpEncoding &tgt);
  smt::FnDecl getTruncateFn(const AbsFpEncoding &tgt);
  smt::FnDecl getExpFn();
  smt::FnDecl getHashFnForAddAssoc();
  smt::FnDecl getRoundDirFn();
  smt::FnDecl getMaxFn();
  smt::FnDecl getInt32ToFpFn();

  size_t getHashRangeBits() const;
  uint64_t getSignBit() const;

public:
  void addConstants(const std::set<llvm::APFloat>& const_set);
  smt::Expr constant(const llvm::APFloat &f) const;
  smt::Expr zero(bool isNegative = false, bool isFpaValue = false) const;
  smt::Expr one(bool isNegative = false, bool isFpaValue = false) const;
  smt::Expr infinity(bool isNegative = false, bool isFpaValue = false) const;
  smt::Expr nan() const;
  smt::Expr largest(bool isNegative = false, bool isFpaValue = false) const;

  std::vector<std::pair<llvm::APFloat, smt::Expr>> getAllConstants() const;
  void evalConsts(smt::Model model);
  std::vector<llvm::APFloat> possibleConsts(const smt::Expr &e) const;
  smt::Expr isnan(const smt::Expr &f);
  smt::Expr iszero(const smt::Expr &f, bool isNegative);
  smt::Expr isinf(const smt::Expr &f, bool isNegative);

  smt::Expr abs(const smt::Expr &f);
  smt::Expr neg(const smt::Expr &f);
  smt::Expr add(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr mul(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr div(const smt::Expr &f1, const smt::Expr &f2);
  // If elems != nullopt, a must be
  //    lambda i, ite(i=0, elems[0], ite(i = 1, elems[1], ...))
  smt::Expr sum(const smt::Expr &a, const smt::Expr &n,
      std::optional<std::vector<smt::Expr>> &&elems = std::nullopt,
      std::optional<smt::Expr> &&initValue = std::nullopt);
  smt::Expr max(const smt::Expr &arr, const smt::Expr &n,
      std::optional<smt::Expr> &&initValue = std::nullopt);
  smt::Expr exp(const smt::Expr &x);
  smt::Expr dot(const smt::Expr &a, const smt::Expr &b,
      const smt::Expr &n, std::optional<smt::Expr> &&initValue = std::nullopt);
  smt::Expr extend(const smt::Expr &f, aop::AbsFpEncoding &tgt);
  smt::Expr truncate(const smt::Expr &f, aop::AbsFpEncoding &tgt);
  smt::Expr castFromSignedInt(const smt::Expr &integer);
  smt::Expr cmp(mlir::arith::CmpFPredicate pred, const smt::Expr &f1,
      const smt::Expr &f2);
  smt::Expr getFpAssociativePrecondition();
  smt::Expr getFpTruncatePrecondition(aop::AbsFpEncoding &tgt);
  smt::Expr getFpConstantPrecondition();

private:
  smt::Expr lambdaSum(const smt::Expr &a, const smt::Expr &n);
  smt::Expr lambdaSum(const std::vector<smt::Expr> &elems);
  smt::Expr multisetSum(const smt::Expr &a, const smt::Expr &n);

  smt::Expr getSignBit(const smt::Expr &f) const;
  smt::Expr getMagnitudeBits(const smt::Expr &f) const;
  smt::Expr getLimitBits(const smt::Expr &f) const;
  smt::Expr getTruncatedBits(const smt::Expr &f) const;
  std::optional<smt::Expr> getPrecisionBits(const smt::Expr &f) const;
};

AbsFpEncoding &getFloatEncoding();
AbsFpEncoding &getDoubleEncoding();
AbsFpEncoding &getFpEncoding(mlir::Type);

};
