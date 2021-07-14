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
    writable(ctx.bool_const(("writable" + to_string(bid)).c_str())),
    numelem(ctx.constant(("numelem" + to_string(bid)).c_str(), Index::sort())) {
}

z3::expr MemBlock::store(const z3::expr &f32val, const z3::expr &idx) {
  array = z3::store(array, idx, f32val);
  return z3::ult(idx, numelem) && writable;
}

pair<z3::expr, z3::expr> MemBlock::load(const z3::expr &idx) const {
  return {z3::select(array, idx), z3::ult(idx, numelem)};
}

MemBlock Memory::getMemBlock(const z3::expr &bid) const {
  // Currently we only support 1 memblocks. This constarints will be relaxed later.
  return mb0;
}
