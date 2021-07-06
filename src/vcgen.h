#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <algorithm>

struct Results {
public:
    static Results success() {
        return Results(0);
    }
    static Results failure(int code = 1) {
        return Results(code);
    }

    // Returns true if the value equals zero.
    bool succeeded() const { return code == 0; }
    // Returns true if the value is other than zero.
    bool failed() const { return !succeeded(); }
    // Returns status code
    int getCode() const { return code; }

    // set default value to zero
    Results() : code(0) {}

    Results merge (const Results &RHS) {
        code = std::max(code, RHS.code);
        return *this;
    }

private:
    Results(int code) : code(code) {}
    int code;
};

Results verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const std::string &dump_smt_to);
