#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace std;

static int log2(unsigned int NUM_BLOCKS) {
  return (int)floor(log2(max((int)NUM_BLOCKS - 1, 1))) + 1;
}

Memory* Memory::create(unsigned int NUM_BLOCKS, MemType type) {
  switch(type) {
    case MemType::SINGLE:
      return new SingleArrayMemory(NUM_BLOCKS);
    case MemType::MULTIPLE:
      return new MultipleArrayMemory(NUM_BLOCKS);
  }
}

SingleArrayMemory::SingleArrayMemory(unsigned int NUM_BLOCKS):
  BID_BITS(log2(NUM_BLOCKS)),
  NUM_BLOCKS(NUM_BLOCKS),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), Index::sort()))) {}

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

MultipleArrayMemory::MultipleArrayMemory(unsigned int NUM_BLOCKS):
  BID_BITS(log2(NUM_BLOCKS)),
  NUM_BLOCKS(NUM_BLOCKS),
  arrayMaps(NUM_BLOCKS, ctx.constant("arrayMaps",
    ctx.array_sort(Index::sort(), Float::sort()))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), Index::sort()))) {}

MemBlock MultipleArrayMemory::getMemBlock(const z3::expr &bid) const {
  z3::expr array(ctx);
  if (NUM_BLOCKS == 1) {
    array = arrayMaps[0];
  } else {
    // Z3 if-then-else encoding
    assert(NUM_BLOCKS >= 2);
    array = z3::ite(
      bid == ctx.bv_val(NUM_BLOCKS - 2, BID_BITS),
      arrayMaps[NUM_BLOCKS - 2], arrayMaps[NUM_BLOCKS - 1]);
    for (int i = (int)NUM_BLOCKS - 3; i >= 0; i --)
      array = z3::ite(bid == ctx.bv_val(i, BID_BITS), arrayMaps[i], array);
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
  for (int i = 0; i < NUM_BLOCKS; i ++) {
    arrayMaps[i] = z3::ite(
      bid == ctx.bv_val(i, BID_BITS),
      z3::store(block.array, idx, f32val), arrayMaps[i]);
  }
  return z3::ult(idx, block.numelem) && block.writable;
}

std::pair<z3::expr, z3::expr> MultipleArrayMemory::load(
  const z3::expr &bid, const z3::expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
}
