#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace std;

static unsigned int ulog2(unsigned int numBlocks) {
  if (numBlocks == 0)
    return 0;
  return (unsigned int) ceil(log2(std::max(numBlocks, (unsigned int) 2)));
}

z3::expr MemBlock::store(const z3::expr &f32val, const z3::expr &idx) {
  array = z3::store(array, idx, f32val);
  return z3::ult(idx, numelem) && writable;
}

pair<z3::expr, z3::expr> MemBlock::load(const z3::expr &idx) const {
  return {z3::select(array, idx), z3::ult(idx, numelem)};
}

Memory::Memory(unsigned int numBlocks):
  bits(ulog2(numBlocks)),
  numBlocks(numBlocks),
  arrayMap(ctx.constant("arrayMap",
    ctx.array_sort(ctx.bv_sort(bits), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMap(ctx.constant("writableMap",
    ctx.array_sort(ctx.bv_sort(bits), ctx.bool_sort()))),
  numelemMap(ctx.constant("numelemMap",
    ctx.array_sort(ctx.bv_sort(bits), Index::sort()))) {}

MemBlock Memory::getMemBlock(const z3::expr &bid) const {
  z3::expr array = z3::select(arrayMap, bid);
  z3::expr writable = z3::select(writableMap, bid);
  z3::expr numelem = z3::select(numelemMap, bid);
  return MemBlock(array, writable, numelem);
}

void Memory::updateMemBlock(const z3::expr &bid, bool writable) {
  z3::store(writableMap, bid, ctx.bool_val(writable));
}
