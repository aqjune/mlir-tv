#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>

extern z3::context ctx;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<z3::expr> &es);

z3::expr get1DSize(const std::vector<z3::expr> &dims);
std::vector<z3::expr> from1DIdx(
    z3::expr idx1d, const std::vector<z3::expr> &dims);
std::vector<z3::expr> simplifyList(const std::vector<z3::expr> &exprs);