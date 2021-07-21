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
    case MemEncoding::SINGLE:
      return new SingleArrayMemory(numBlocks);
    case MemEncoding::MULTIPLE:
      return new MultipleArrayMemory(numBlocks);
  }
}

SingleArrayMemory::SingleArrayMemory(unsigned int numBlocks):
  bits(ulog2(numBlocks)),
  numBlocks(numBlocks),
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
  bits(ulog2(numBlocks)),
  numBlocks(numBlocks),
  arrayMaps(numBlocks, ctx.constant("arrayMaps",
    ctx.array_sort(Index::sort(), Float::sort()))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bits), Index::sort()))) {}

MemBlock MultipleArrayMemory::getMemBlock(const z3::expr &bid) const {
  z3::expr array(ctx);
  if (numBlocks == 1) {
    array = arrayMaps[0];
  } else {
    // Z3 if-then-else encoding
    assert(numBlocks >= 2);
    array = z3::ite(
      bid == ctx.bv_val(numBlocks - 2, bits),
      arrayMaps[numBlocks - 2], arrayMaps[numBlocks - 1]);
    for (int i = (int)numBlocks - 3; i >= 0; i --)
      array = z3::ite(bid == ctx.bv_val(i, bits), arrayMaps[i], array);
  }
  z3::expr writable = z3::select(writableMaps, bid);
  z3::expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

void MultipleArrayMemory::setWritable(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

z3::expr MultipleArrayMemory::store(const z3::expr &f32val,
  const z3::expr &bid, const z3::expr &idx) {
  const auto block = getMemBlock(bid);
  for (int i = 0; i < numBlocks; i ++) {
    arrayMaps[i] = z3::ite(
      bid == ctx.bv_val(i, bits),
      z3::store(block.array, idx, f32val), arrayMaps[i]);
  }
  return z3::ult(idx, block.numelem) && block.writable;
}

std::pair<z3::expr, z3::expr> MultipleArrayMemory::load(
  const z3::expr &bid, const z3::expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
}
