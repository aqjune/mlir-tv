#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"
#include <vector>

extern z3::context ctx;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e);
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<z3::expr> &es);