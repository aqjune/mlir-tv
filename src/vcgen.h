#pragma once

#include "memory.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <algorithm>

class Results {
public:
  enum Code {
    SUCCESS = 0, TIMEOUT, RETVALUE, UB
  };

  // Returns true if the value equals zero.
  bool succeeded() const { return code == SUCCESS; }
  // Returns true if the value is other than zero.
  bool failed() const { return !succeeded(); }
  // Returns status code
  Code getCode() const { return code; }

  // set default value to success
  Results(Code code = SUCCESS) : code(code) {}

  // get the worse result
  Results merge (const Results &RHS) {
    code = std::max(code, RHS.code);
    return *this;
  }

public:
  Code code;
};

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
    const std::string &dump_smt_to,
    unsigned int numBlocks,
    MemEncoding encoding);
