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

Memory* Memory::create(unsigned int numBlocks, MemEncoding encoding) {
  switch(encoding) {
    case MemEncoding::SINGLE_ARRAY:
      return new SingleArrayMemory(numBlocks);
    case MemEncoding::MULTIPLE_ARRAY:
      return new MultipleArrayMemory(numBlocks);
  }
}

SingleArrayMemory::SingleArrayMemory(unsigned int numBlocks):
  Memory(ulog2(numBlocks), numBlocks),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bits), Index::sort()))) {}

MemBlock SingleArrayMemory::getMemBlock(const z3::expr &bid) const {
  z3::expr array = z3::select(arrayMaps, bid);
  z3::expr writable = z3::select(writableMaps, bid);
  z3::expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

void SingleArrayMemory::setWritable(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

z3::expr SingleArrayMemory::store(const z3::expr &f32val,
  const z3::expr &bid, const z3::expr &idx) {
  const auto block = getMemBlock(bid);
  arrayMaps = z3::store(arrayMaps, bid, z3::store(block.array, idx, f32val));
  return z3::ult(idx, block.numelem) && block.writable;
}

std::pair<z3::expr, z3::expr> SingleArrayMemory::load(
  const z3::expr &bid, const z3::expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
}

MultipleArrayMemory::MultipleArrayMemory(unsigned int numBlocks):
  Memory(ulog2(numBlocks), numBlocks),
  arrayMaps(numBlocks, ctx.constant("arrayMaps",
    ctx.array_sort(Index::sort(), Float::sort()))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bits), Index::sort()))) {}

void MultipleArrayMemory::setWritable(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

z3::expr MultipleArrayMemory::store(const z3::expr &f32val,
  const z3::expr &bid, const z3::expr &idx) {
  for (int i = 0; i < numBlocks; i ++) {
    arrayMaps[i] = z3::ite(
      bid == ctx.bv_val(i, bits),
      z3::store(arrayMaps[i], idx, f32val),
      arrayMaps[i]
    );
  }
  z3::expr numelem = z3::select(numelemMaps, bid);
  z3::expr writable = z3::select(writableMaps, bid);
  return z3::ult(idx, numelem) && writable;
}

std::pair<z3::expr, z3::expr> MultipleArrayMemory::load(
  const z3::expr &bid, const z3::expr &idx) const {
  z3::expr value = z3::select(arrayMaps[0], idx);
  for (int i = 1; i < numBlocks; i ++) {
    value = z3::ite(
      bid == ctx.bv_val(i, bits),
      z3::select(arrayMaps[i], idx),
      value
    );
  }
  z3::expr numelem = z3::select(numelemMaps, bid);
  z3::expr success = z3::ult(idx, numelem);
  return {value, success};
}
