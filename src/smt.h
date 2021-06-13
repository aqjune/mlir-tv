#pragma once

#include "llvm/Support/raw_ostream.h"
#include "z3++.h"

extern z3::context ctx;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e);