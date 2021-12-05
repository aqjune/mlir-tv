#pragma once

#include "smt.h"
#include "llvm/ADT/APFloat.h"
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
  bool fpUlt;
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

// unrollIntSum: Fully unroll sum(arr) where arr is an int array of const size
//               as arr[0] + arr[1] + .. + arr[len-1]?
// floatNonConstsCnt: # of non-constant distinct f32 values necessary to
// validate the transformation.
// NOTE: This resets the used abstract ops record.
void setAbstraction(AbsLevelFpDot, AbsLevelFpCast, AbsLevelIntDot,
                    bool isFpAddAssociative,
                    bool unrollIntSum,
                    unsigned floatNonConstsCnt,
                    std::set<llvm::APFloat> floatConsts,
                    unsigned doubleNonConstsCnt,
                    std::set<llvm::APFloat> doubleConsts);
// A set of options that must not change the precision of validation.
// useMultiset: To encode commutativity of fp summation, use multiset?
void setEncodingOptions(bool useMultiset);

bool getFpAddAssociativity();

smt::Expr getFpAssociativePrecondition();
smt::Expr getFpUltPrecondition();

smt::Expr intSum(const smt::Expr &arr, const smt::Expr &n);
smt::Expr intDot(const smt::Expr &arr1, const smt::Expr &arr2,
                 const smt::Expr &n);

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
  // Abstract representation of valid fp constants.
  std::map<llvm::APFloat, smt::Expr> fpconst_absrepr;
  uint64_t fpconst_absrepr_num = 0;

  const static unsigned SIGN_BITS = 1;
  // The BV width of abstract fp encoding.
  // fp_bv_bits = SIGN_BITS + value_bv_bits
  unsigned fp_bitwidth;
  unsigned value_bitwidth;

  // Bits for casting.
  struct ValueBitInfo {
    unsigned limit_bitwidth;
    unsigned smaller_value_bitwidth;
    unsigned prec_bitwidth;

    unsigned get_value_bitwidth() {
      return limit_bitwidth + smaller_value_bitwidth + prec_bitwidth;
    }
  };
  ValueBitInfo value_bit_info;

  std::vector<std::tuple<smt::Expr, smt::Expr, smt::Expr>> fp_sum_relations;

  // These are lazily created.
  std::optional<smt::FnDecl> fp_sumfn;
  std::optional<smt::FnDecl> fp_assoc_sumfn;
  std::optional<smt::FnDecl> fp_dotfn;
  std::optional<smt::FnDecl> fp_addfn;
  std::optional<smt::FnDecl> fp_mulfn;
  std::optional<smt::FnDecl> fp_divfn;
  std::optional<smt::FnDecl> fp_ultfn;
  std::optional<smt::FnDecl> fp_extendfn;
  std::optional<smt::FnDecl> fp_truncatefn;
  std::optional<smt::FnDecl> fp_expfn;
  std::optional<smt::FnDecl> fp_hashfn;
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

  smt::Sort sort() const {
    return smt::Sort::bvSort(fp_bitwidth);
  }

private:
  // Returns a fully abstract add fn fp_add(fty, fty) -> fty2
  // where fty is BV(fp_bv_bits) and fty2 is BV(fp_bv_bits - TYPE_BITS).
  // It is the user of this function that fills in TYPE_BITS.
  smt::FnDecl getAddFn();
  // Returns a fully abstract mul fn fp_mul(value_bv_bits-1, value_bv_bits-1)
  // -> BV(value_bv_bits)
  smt::FnDecl getMulFn();
  // Returns a fully abstract div fn fp_mul(value_bv_bits-1, value_bv_bits-1)
  // -> BV(value_bv_bits)
  smt::FnDecl getDivFn();
  smt::FnDecl getAssocSumFn();
  smt::FnDecl getSumFn();
  smt::FnDecl getDotFn();
  smt::FnDecl getUltFn();
  smt::FnDecl getExtendFn(const AbsFpEncoding &tgt);
  smt::FnDecl getTruncateFn(const AbsFpEncoding &tgt);
  smt::FnDecl getExpFn();
  smt::FnDecl getHashFnForAddAssoc();

  size_t getHashRangeBits() const;
  uint64_t getSignBit() const;

public:
  void addConstants(const std::set<llvm::APFloat>& const_set);
  smt::Expr constant(const llvm::APFloat &f) const;
  smt::Expr zero(bool isNegative = false) const;
  smt::Expr one(bool isNegative = false) const;
  smt::Expr infinity(bool isNegative = false) const;
  smt::Expr nan() const;

  std::vector<std::pair<llvm::APFloat, smt::Expr>> getAllConstants() const;
  std::vector<llvm::APFloat> possibleConsts(const smt::Expr &e) const;
  smt::Expr isnan(const smt::Expr &f);
  smt::Expr abs(const smt::Expr &f);
  smt::Expr neg(const smt::Expr &f);
  smt::Expr add(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr mul(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr div(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr sum(const smt::Expr &a, const smt::Expr &n);
  smt::Expr exp(const smt::Expr &x);
  smt::Expr dot(const smt::Expr &a, const smt::Expr &b, const smt::Expr &n);
  smt::Expr fult(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr extend(const smt::Expr &f, aop::AbsFpEncoding &tgt);
  smt::Expr truncate(const smt::Expr &f, aop::AbsFpEncoding &tgt);
  smt::Expr getFpAssociativePrecondition();

private:
  smt::Expr multisetSum(const smt::Expr &a, const smt::Expr &n);
};

AbsFpEncoding &getFloatEncoding();
AbsFpEncoding &getDoubleEncoding();
AbsFpEncoding &getFpEncoding(mlir::Type);

};
