#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>

namespace smt {
extern z3::context ctx;

z3::expr get1DSize(const std::vector<z3::expr> &dims);
std::vector<z3::expr> from1DIdx(
    z3::expr idx1d, const std::vector<z3::expr> &dims);
std::vector<z3::expr> simplifyList(const std::vector<z3::expr> &exprs);

z3::expr to1DIdx(const std::vector<z3::expr> &idxs,
                 const std::vector<z3::expr> &dims);
z3::expr to1DIdxWithLayout(const std::vector<z3::expr> &idxs, z3::expr layout);
z3::expr fitsInDims(const std::vector<z3::expr> &idxs,
                    const std::vector<z3::expr> &sizes);
z3::expr_vector toExprVector(const std::vector<z3::expr> &vec);
std::string or_omit(const z3::expr &e);
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<z3::expr> &es);