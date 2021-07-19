#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace std;

Memory::Memory(unsigned int NUM_BLOCKS):
  NUM_BLOCKS(NUM_BLOCKS),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), Index::sort()))) {}

Memory::MemBlock Memory::getMemBlock(const z3::expr &bid) const {
  z3::expr array = z3::select(arrayMaps, bid);
  z3::expr writable = z3::select(writableMaps, bid);
  z3::expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

z3::expr Memory::getNumElementsOfMemBlock(const z3::expr &bid) const {
  return getMemBlock(bid).numelem;
}

void Memory::updateMemBlock(const z3::expr &bid, bool writable) {
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

////

NewMemory::NewMemory(unsigned int NUM_BLOCKS):
  NUM_BLOCKS(NUM_BLOCKS),
  arrayMaps(NUM_BLOCKS, ctx.constant("arrayMaps",
    ctx.array_sort(Index::sort(), Float::sort()))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), Index::sort()))) {}

NewMemory::MemBlock NewMemory::getMemBlock(const z3::expr &bid) const {
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

z3::expr NewMemory::getNumElementsOfMemBlock(const z3::expr &bid) const {
  return getMemBlock(bid).numelem;
}

void NewMemory::updateMemBlock(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

z3::expr NewMemory::store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx) {
  const auto block = getMemBlock(bid);
  for (int i = 0; i < NUM_BLOCKS; i ++) {
    arrayMaps[i] = z3::ite(
      bid == ctx.bv_val(i, BID_BITS),
      z3::store(block.array, idx, f32val), arrayMaps[i]);
  }
  return z3::ult(idx, block.numelem) && block.writable;
}

std::pair<z3::expr, z3::expr> NewMemory::load(const z3::expr &bid, const z3::expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
}
