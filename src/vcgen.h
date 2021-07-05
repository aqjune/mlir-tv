#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <algorithm>

struct Results {
public:
    static Results success() {
        return Results(0);
    }
    static Results failure(int value = 1) {
        return Results(value);
    }

    // Returns true if the value equals zero.
    bool succeeded() const { return value == 0; }
    // Returns true if the value is other than zero.
    bool failed() const { return !succeeded(); }

    // set default value to zero
    Results() : value(0) {}

    Results &operator &= (const Results &RHS) {
        value = std::max(value, RHS.value);
        return *this;
    }

    Results &operator |= (const Results &RHS) {
        value = std::min(value, RHS.value);
        return *this;
    }

private:
    Results(int value) : value(value) {}
    int value;
};

// Note: to avoid conflict with LogicalResults inlining function, here I use abbreviations
inline Results succ() {
  return Results::success();
}

inline Results fail(int value) {
    return Results::failure(value);
}

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const std::string &dump_smt_to);