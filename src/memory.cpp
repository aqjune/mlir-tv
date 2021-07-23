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
  default:
    llvm_unreachable("Unknown memory encoding");
  }
}

pair<z3::expr, std::vector<z3::expr>>
SingleArrayMemory::refines(const Memory &other) const {
  auto bid = ctx.bv_const("bid", bidBits);
  auto offset = Index("offset", true);
  auto [srcValue, srcSuccess] = load(bid, offset);
  auto srcWritable = getWritable(bid);
  auto [tgtValue, tgtSuccess] = other.load(bid, offset);
  auto tgtWritable = other.getWritable(bid);
  // define memory refinement using writable refinement and value refinement
  auto wRefinement = z3::implies(srcWritable, tgtWritable);
  auto vRefinement = z3::eq(tgtValue, srcValue);
  auto refinement = z3::implies(tgtSuccess, srcSuccess && wRefinement && vRefinement);
  return {refinement, {bid, offset}};
}

SingleArrayMemory::SingleArrayMemory(unsigned int numBlocks):
  Memory(ulog2(numBlocks), numBlocks),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), Index::sort()))) {}

MemBlock SingleArrayMemory::getMemBlock(const z3::expr &bid) const {
  z3::expr array = z3::select(arrayMaps, bid);
  z3::expr writable = z3::select(writableMaps, bid);
  z3::expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

void SingleArrayMemory::setWritable(const z3::expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

z3::expr SingleArrayMemory::getWritable(const z3::expr &bid) const {
  return z3::select(writableMaps, bid);
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
    Memory(ulog2(numBlocks), numBlocks) {
  for (unsigned i = 0; i < numBlocks; ++i) {
    auto suffix = [&](const string &s) {
      return s + to_string(i);
    };
    arrays.push_back(ctx.constant(suffix("array").c_str(),
        ctx.array_sort(Index::sort(), Float::sort())));
    writables.push_back(ctx.bool_const(suffix("writable").c_str()));
    numelems.push_back(ctx.bv_const(suffix("numelems").c_str(), Index::BITS));
  }
}

z3::expr MultipleArrayMemory::itebid(
    const z3::expr &bid, function<z3::expr(unsigned)> fn) const {
  assert(numBlocks > 0);
  assert(bid.get_sort().is_bv() && bid.get_sort().bv_size() == getBIDBits());

  uint64_t const_bid;
  if (bid.is_numeral_u64(const_bid))
    return fn(const_bid);

  const unsigned bits = bid.get_sort().bv_size();
  unsigned curbid = numBlocks - 1;
  z3::expr val = fn(curbid);

  while (curbid) {
    curbid--;
    val = z3::ite(bid == ctx.bv_val(curbid, bits), fn(curbid), val);
  }

  return val;
}

void MultipleArrayMemory::update(
    const z3::expr &bid, function<z3::expr*(unsigned)> getExprToUpdate,
    function<z3::expr(unsigned)> getUpdatedValue) const {
  assert(numBlocks > 0);
  assert(bid.get_sort().is_bv() && bid.get_sort().bv_size() == getBIDBits());

  uint64_t const_bid;
  if (bid.is_numeral_u64(const_bid)) {
    *getExprToUpdate(const_bid) = getUpdatedValue(const_bid);
    return;
  }

  const unsigned bits = getBIDBits();
  for (unsigned i = 0; i < numBlocks; ++i) {
    z3::expr *expr = getExprToUpdate(i);
    assert(expr);
    *expr = z3::ite(bid == ctx.bv_val(i, bits), getUpdatedValue(i), *expr);
  }
}

z3::expr MultipleArrayMemory::getNumElementsOfMemBlock(
    const z3::expr &bid) const {
  return itebid(bid, [&](auto ubid) { return numelems[ubid]; });
}

void MultipleArrayMemory::setWritable(const z3::expr &bid, bool writable) {
  update(bid, [&](unsigned ubid) { return &writables[ubid]; },
      [&](auto) { return ctx.bool_val(writable); });
}

z3::expr MultipleArrayMemory::getWritable(const z3::expr &bid) const {
  return itebid(bid, [&](auto ubid) { return writables[ubid]; });
}

z3::expr MultipleArrayMemory::store(const z3::expr &f32val,
    const z3::expr &bid, const z3::expr &idx) {
  update(bid, [&](auto ubid) { return &arrays[ubid]; },
      [&](auto ubid) { return z3::store(arrays[ubid], idx, f32val); });

  return z3::ult(idx, getNumElementsOfMemBlock(bid)) && getWritable(bid);
}

std::pair<z3::expr, z3::expr> MultipleArrayMemory::load(
    unsigned ubid, const z3::expr &idx) const {
  assert(ubid < numBlocks);

  z3::expr success = z3::ult(idx, getNumElementsOfMemBlock(ubid));
  return {z3::select(arrays[ubid], idx), success};
}

std::pair<z3::expr, z3::expr> MultipleArrayMemory::load(
    const z3::expr &bid, const z3::expr &idx) const {
  z3::expr value = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).first; });
  z3::expr success = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).second; });
  return {value, success};
}

pair<z3::expr, std::vector<z3::expr>>
MultipleArrayMemory::refines(const Memory &other0) const {
  // NOTE: We cannot use dynamic_cast because we disabled -fno-rtti to link to
  // a plain LLVM.
  const MultipleArrayMemory &other =
      *static_cast<const MultipleArrayMemory *>(&other0);
  assert(other.numBlocks == numBlocks);

  auto bid = ctx.bv_const("bid", bidBits);
  auto offset = Index("offset", true);

  auto refines = [this, &other, &bid, &offset](unsigned ubid) {
    auto [srcValue, srcSuccess] = load(ubid, offset);
    auto srcWritable = getWritable(ubid);
    auto [tgtValue, tgtSuccess] = other.load(ubid, offset);
    auto tgtWritable = other.getWritable(ubid);

    auto wRefinement = z3::implies(srcWritable, tgtWritable);
    auto vRefinement = z3::eq(tgtValue, srcValue);
    return z3::implies(tgtSuccess, srcSuccess && wRefinement && vRefinement);
  };

  return { itebid(bid, refines), {bid, offset}};
}
