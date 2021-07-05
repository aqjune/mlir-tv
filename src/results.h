#pragma once

#include <algorithm>
using namespace std;

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
        value = max(value, RHS.value);
        return *this;
    }
    Results &operator |= (const Results &RHS) {
        value = min(value, RHS.value);
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