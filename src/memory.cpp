#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace std;

MemBlock::MemBlock(unsigned bid):
    bid(bid),
    array(ctx.constant(
        ("blk" + to_string(bid)).c_str(),
        ctx.array_sort(Index::sort(), Float::sort()))),
    numelem(ctx.constant(("numelem" + to_string(bid)).c_str(), Index::sort())),
    isConstant(ctx.bool_const(("isConstant" + to_string(bid)).c_str())) {
}

z3::expr MemBlock::store(const z3::expr &f32val, const z3::expr &idx) {
  array = z3::store(array, idx, f32val);
  return z3::ult(idx, numelem) && !isConstant;
}

pair<z3::expr, z3::expr> MemBlock::load(const z3::expr &idx) const {
  return {z3::select(array, idx), z3::ult(idx, numelem)};
}
