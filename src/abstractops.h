#pragma once

#include "smt.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinOps.h"
#include <vector>

namespace aop {

struct UsedAbstractOps {
  // Float ops
  bool fpDot;
  bool fpAdd;
  bool fpMul;
  bool fpSum;
  bool fpUlt;
  // Int ops
  bool intDot;
  bool intSum;
};
UsedAbstractOps getUsedAbstractOps();

enum class AbsLevelFpDot {
  FULLY_ABS = 0, // FP Dot is a fully unknown function
  SUM_MUL   = 1, // FP Dot is a summation of pairwisely multiplied values
};

enum class AbsLevelIntDot {
  FULLY_ABS = 0, // Int Dot is a fully unknown function
  SUM_MUL   = 1, // Int Dot is a summation of pairwisely multiplied values
};

// This resets the used abstract ops record.
void setAbstraction(AbsLevelFpDot, AbsLevelIntDot, bool isFpAddAssociative,
                    unsigned floatBits, unsigned doubleBits);
void setEncodingOptions(bool use_multiset);

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
  const static unsigned TYPE_BITS = 1;

  unsigned value_bv_bits;
  unsigned fp_bv_bits;
  uint64_t inf_value;
  uint64_t nan_value;
  uint64_t signed_value;

  std::vector<std::tuple<smt::Expr, smt::Expr, smt::Expr>> fp_sum_relations;

  // These are lazily created.
  std::optional<smt::FnDecl> fp_sumfn;
  std::optional<smt::FnDecl> fp_assoc_sumfn;
  std::optional<smt::FnDecl> fp_dotfn;
  std::optional<smt::FnDecl> fp_addfn;
  std::optional<smt::FnDecl> fp_mulfn;
  std::optional<smt::FnDecl> fp_ultfn;
  std::string fn_suffix;

public:
  AbsFpEncoding(const llvm::fltSemantics &semantics, unsigned valuebits,
      std::string &&fn_suffix);

  smt::Sort sort() const {
    return smt::Sort::bvSort(fp_bv_bits);
  }

private:
  // Returns a fully abstract add fn fp_add(fty, fty) -> fty2
  // where fty is BV(fp_bv_bits) and fty2 is BV(fp_bv_bits - TYPE_BITS).
  // It is the user of this function that fills in TYPE_BITS.
  smt::FnDecl getAddFn();
  // Returns a fully abstract mul fn fp_mul(value_bv_bits-1, value_bv_bits-1)
  // -> BV(value_bv_bits)
  smt::FnDecl getMulFn();
  smt::FnDecl getAssocSumFn();
  smt::FnDecl getSumFn();
  smt::FnDecl getDotFn();
  smt::FnDecl getUltFn();

public:
  smt::Expr constant(const llvm::APFloat &f);
  smt::Expr zero(bool isNegative = false);
  smt::Expr one(bool isNegative = false);
  smt::Expr infinity(bool isNegative = false);
  smt::Expr nan();

  std::vector<std::pair<llvm::APFloat, smt::Expr>> getAllConstants() const;
  std::vector<llvm::APFloat> possibleConsts(const smt::Expr &e) const;
  smt::Expr isnan(const smt::Expr &f);
  smt::Expr abs(const smt::Expr &f);
  smt::Expr add(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr mul(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr sum(const smt::Expr &a, const smt::Expr &n);
  smt::Expr dot(const smt::Expr &a, const smt::Expr &b, const smt::Expr &n);
  smt::Expr fult(const smt::Expr &f1, const smt::Expr &f2);
  smt::Expr getFpAssociativePrecondition() const;

private:
  smt::Expr multisetSum(const smt::Expr &a, const smt::Expr &n);
};

AbsFpEncoding &getFloatEncoding();
AbsFpEncoding &getDoubleEncoding();
AbsFpEncoding &getFpEncoding(mlir::Type);

};
