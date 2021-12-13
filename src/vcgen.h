#pragma once

#include "memory.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <algorithm>

class Results {
public:
  enum Code {
    SUCCESS = 0,
    TIMEOUT = 101,
    RETVALUE = 102,
    UB = 103,
    INCONSISTENT = 104
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

const int UNSUPPORTED_EXIT_CODE = 91;

enum class VerificationStep {
  UB,
  RetValue,
  Memory
};

Results validate(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt);
