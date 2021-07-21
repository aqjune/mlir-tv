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

Memory::Memory(unsigned int numBlocks):
  bits(ulog2(numBlocks)),
  numBlocks(numBlocks),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bits), Index::sort()))) {}

MemBlock Memory::getMemBlock(const z3::expr &bid) const {
  z3::expr array = z3::select(arrayMaps, bid);
  z3::expr writable = z3::select(writableMaps, bid);
  z3::expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

 void Memory::setWritable(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
 }

 z3::expr Memory::store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx) {
  const auto block = getMemBlock(bid);
  arrayMaps = z3::store(arrayMaps, bid, z3::store(block.array, idx, f32val));
  return z3::ult(idx, block.numelem) && block.writable;
 }

 std::pair<z3::expr, z3::expr> Memory::load(const z3::expr &bid, const z3::expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
 }
