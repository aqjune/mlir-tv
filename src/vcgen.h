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

    Results merge (const Results &RHS) {
        value = std::max(value, RHS.value);
        return *this;
    }

private:
    Results(int value) : value(value) {}
    int value;
};

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const std::string &dump_smt_to);
