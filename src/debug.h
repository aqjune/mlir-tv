#pragma once

#include "llvm/Support/raw_ostream.h"
#include <string>

void setVerbose(bool vb);

llvm::raw_ostream &verbose(const std::string &prefix);