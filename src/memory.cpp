#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace std;

static int log2(unsigned int NUM_BLOCKS) {
  return (int)floor(log2(max((int)NUM_BLOCKS - 1, 1))) + 1;
}

Memory::Memory(unsigned int NUM_BLOCKS):
  BID_BITS(log2(NUM_BLOCKS)),
  NUM_BLOCKS(NUM_BLOCKS),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(BID_BITS), Index::sort()))) {}

MemBlock Memory::getMemBlock(const z3::expr &bid) const {
  llvm::outs() << "BID: " << bid << "\n";
  // llvm::outs() << "BID: " << bid.ctx() == arrayMaps.ctx() << "\n";
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
  //  llvm::outs() << block.array << "\n";
  //  llvm::outs() << block.numelem << "\n";
   return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
 }
