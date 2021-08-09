#include "memory.h"
#include "smt.h"
#include "value.h"
#include <string>

using namespace smt;
using namespace std;

static unsigned int ulog2(unsigned int numBlocks) {
  if (numBlocks == 0)
    return 0;
  return (unsigned int) ceil(log2(std::max(numBlocks, (unsigned int) 2)));
}

Memory* Memory::create(
    unsigned int numGlobalBlocks,
    unsigned int maxLocalBlocks,
    MemEncoding encoding) {
  switch(encoding) {
  case MemEncoding::SINGLE_ARRAY:
    return new SingleArrayMemory(numGlobalBlocks, maxLocalBlocks);
  case MemEncoding::MULTIPLE_ARRAY:
    return new MultipleArrayMemory(numGlobalBlocks, maxLocalBlocks);
  default:
    llvm_unreachable("Unknown memory encoding");
  }
}

expr Memory::isGlobalBlock(const expr &bid) const {
  return z3::ult(bid, numGlobalBlocks);
}

expr Memory::isLocalBlock(const expr &bid) const {
  return !isGlobalBlock(bid);
}

pair<expr, std::vector<expr>>
SingleArrayMemory::refines(const Memory &other) const {
  auto bid = ctx.bv_const("bid", bidBits);
  auto offset = Index("offset", true);
  auto [srcValue, srcSuccess] = load(bid, offset);
  auto srcWritable = getWritable(bid);
  auto [tgtValue, tgtSuccess] = other.load(bid, offset);
  auto tgtWritable = other.getWritable(bid);
  // define memory refinement using writable refinement and value refinement
  auto wRefinement = z3::implies(srcWritable, tgtWritable);
  auto vRefinement = (tgtValue == srcValue);
  auto refinement = z3::implies(tgtSuccess, srcSuccess && wRefinement && vRefinement);
  return {z3::implies(isGlobalBlock(bid), refinement), {bid, offset}};
}

SingleArrayMemory::SingleArrayMemory(
    unsigned int numGlobalBlocks, unsigned int maxLocalBlocks):
  Memory(numGlobalBlocks, maxLocalBlocks, ulog2(numGlobalBlocks + maxLocalBlocks)),
  arrayMaps(ctx.constant("arrayMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), ctx.array_sort(Index::sort(), Float::sort())))),
  writableMaps(ctx.constant("writableMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), ctx.bool_sort()))),
  numelemMaps(ctx.constant("numelemMaps",
    ctx.array_sort(ctx.bv_sort(bidBits), Index::sort()))) {}

MemBlock SingleArrayMemory::getMemBlock(const expr &bid) const {
  expr array = z3::select(arrayMaps, bid);
  expr writable = z3::select(writableMaps, bid);
  expr numelem = z3::select(numelemMaps, bid);
  return MemBlock(array, writable, numelem);
}

expr SingleArrayMemory::addLocalBlock(const expr &numelem) {
  assert(numLocalBlocks <= maxLocalBlocks);
  auto bid = ctx.bv_val(numGlobalBlocks + numLocalBlocks, bidBits);
  numelemMaps = z3::store(numelemMaps, bid, numelem);
  numLocalBlocks ++;

  return bid;
}

void SingleArrayMemory::setWritable(const expr &bid, bool writable) {
  writableMaps = z3::store(writableMaps, bid, ctx.bool_val(writable));
}

expr SingleArrayMemory::getWritable(const expr &bid) const {
  return z3::select(writableMaps, bid);
}

expr SingleArrayMemory::store(const expr &f32val,
  const expr &bid, const expr &idx) {
  const auto block = getMemBlock(bid);
  arrayMaps = z3::store(arrayMaps, bid, z3::store(block.array, idx, f32val));
  return z3::ult(idx, block.numelem) && block.writable;
}

std::pair<expr, expr> SingleArrayMemory::load(
  const expr &bid, const expr &idx) const {
  const auto block = getMemBlock(bid);
  return {z3::select(block.array, idx), z3::ult(idx, block.numelem)};
}


MultipleArrayMemory::MultipleArrayMemory(
    unsigned int numGlobalBlocks, unsigned int maxLocalBlocks):
  Memory(numGlobalBlocks, maxLocalBlocks, ulog2(numGlobalBlocks + maxLocalBlocks)) {
  for (unsigned i = 0; i < getNumBlocks(); ++i) {
    auto suffix = [&](const string &s) {
      return s + to_string(i);
    };
    arrays.push_back(ctx.constant(suffix("array").c_str(),
        ctx.array_sort(Index::sort(), Float::sort())));
    writables.push_back(ctx.bool_const(suffix("writable").c_str()));
    numelems.push_back(ctx.bv_const(suffix("numelems").c_str(), Index::BITS));
  }
}

expr MultipleArrayMemory::itebid(
    const expr &bid, function<expr(unsigned)> fn) const {
  assert(getNumBlocks() > 0);
  assert(bid.get_sort().is_bv() && bid.get_sort().bv_size() == getBIDBits());

  uint64_t const_bid;
  if (bid.is_numeral_u64(const_bid))
    return fn(const_bid);

  const unsigned bits = bid.get_sort().bv_size();

  expr expr = fn(0);
  for (unsigned i = 1; i < getNumBlocks(); i ++)
    expr = z3::ite(bid == ctx.bv_val(i, bits), fn(i), expr);

  return expr;
}

void MultipleArrayMemory::update(
    const expr &bid, function<expr*(unsigned)> getExprToUpdate,
    function<expr(unsigned)> getUpdatedValue) const {
  assert(getNumBlocks() > 0);
  assert(bid.get_sort().is_bv() && bid.get_sort().bv_size() == getBIDBits());

  uint64_t const_bid;
  if (bid.is_numeral_u64(const_bid)) {
    *getExprToUpdate(const_bid) = getUpdatedValue(const_bid);
    return;
  }

  const unsigned bits = getBIDBits();
  for (unsigned i = 0; i < getNumBlocks(); ++i) {
    expr *expr = getExprToUpdate(i);
    assert(expr);
    *expr = z3::ite(bid == ctx.bv_val(i, bits), getUpdatedValue(i), *expr);
  }
}

expr MultipleArrayMemory::addLocalBlock(const expr &numelem) {
  auto bid = numGlobalBlocks + numLocalBlocks;
  auto suffix = [&](const string &s) { return s + to_string(bid); };
  arrays.push_back(ctx.constant(suffix("array").c_str(),
        ctx.array_sort(Index::sort(), Float::sort())));
  writables.push_back(ctx.bool_const(suffix("writable").c_str()));
  numelems.push_back(numelem);
  numLocalBlocks ++;

  return ctx.bv_val(bid, bidBits);
}

expr MultipleArrayMemory::getNumElementsOfMemBlock(
    const expr &bid) const {
  return itebid(bid, [&](auto ubid) { return numelems[ubid]; });
}

void MultipleArrayMemory::setWritable(const expr &bid, bool writable) {
  update(bid, [&](unsigned ubid) { return &writables[ubid]; },
      [&](auto) { return ctx.bool_val(writable); });
}

expr MultipleArrayMemory::getWritable(const expr &bid) const {
  return itebid(bid, [&](auto ubid) { return writables[ubid]; });
}

expr MultipleArrayMemory::store(const expr &f32val,
    const expr &bid, const expr &idx) {
  update(bid, [&](auto ubid) { return &arrays[ubid]; },
      [&](auto ubid) { return z3::store(arrays[ubid], idx, f32val); });

  return z3::ult(idx, getNumElementsOfMemBlock(bid)) && getWritable(bid);
}

std::pair<expr, expr> MultipleArrayMemory::load(
    unsigned ubid, const expr &idx) const {
  assert(ubid < getNumBlocks());

  expr success = z3::ult(idx, getNumElementsOfMemBlock(ubid));
  return {z3::select(arrays[ubid], idx), success};
}

std::pair<expr, expr> MultipleArrayMemory::load(
    const expr &bid, const expr &idx) const {
  expr value = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).first; });
  expr success = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).second; });
  return {value, success};
}

pair<expr, std::vector<expr>>
MultipleArrayMemory::refines(const Memory &other0) const {
  // NOTE: We cannot use dynamic_cast because we disabled -fno-rtti to link to
  // a plain LLVM.
  const MultipleArrayMemory &other =
      *static_cast<const MultipleArrayMemory *>(&other0);
  assert(other.numGlobalBlocks == numGlobalBlocks);

  auto bid = ctx.bv_const("bid", bidBits);
  auto offset = Index("offset", true);

  auto refines = [this, &other, &bid, &offset](unsigned ubid) {
    auto [srcValue, srcSuccess] = load(ubid, offset);
    auto srcWritable = getWritable(ubid);
    auto [tgtValue, tgtSuccess] = other.load(ubid, offset);
    auto tgtWritable = other.getWritable(ubid);

    auto wRefinement = z3::implies(srcWritable, tgtWritable);
    auto vRefinement = (tgtValue == srcValue);
    return z3::implies(tgtSuccess, srcSuccess && wRefinement && vRefinement);
  };

  expr refinement = refines(0);
  for (unsigned i = 1; i < numGlobalBlocks; i ++)
    refinement = z3::ite(bid == ctx.bv_val(i, bidBits), refines(i), refinement);

  return {refinement, {bid, offset}};
}
