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

Expr Memory::isGlobalBlock(const Expr &bid) const {
  return bid.ult(numGlobalBlocks);
}

Expr Memory::isLocalBlock(const Expr &bid) const {
  return !isGlobalBlock(bid);
}

pair<Expr, vector<Expr>>
SingleArrayMemory::refines(const Memory &other) const {
  // Create fresh, unbound variables
  auto bid = Expr::mkFreshVar(Sort::bvSort(bidBits), "bid");
  auto offset = Index::var("offset", VarType::FRESH);

  auto [srcValue, srcSuccess] = load(bid, offset);
  auto srcWritable = getWritable(bid);
  auto [tgtValue, tgtSuccess] = other.load(bid, offset);
  auto tgtWritable = other.getWritable(bid);
  // define memory refinement using writable refinement and value refinement
  auto wRefinement = srcWritable.implies(tgtWritable);
  auto vRefinement = (tgtValue == srcValue);
  auto refinement = tgtSuccess.implies(srcSuccess & wRefinement & vRefinement);
  return {isGlobalBlock(bid).implies(refinement), {bid, offset}};
}

SingleArrayMemory::SingleArrayMemory(
    unsigned int numGlobalBlocks, unsigned int maxLocalBlocks):
  MemoryCRTP(numGlobalBlocks, maxLocalBlocks,
    ulog2(numGlobalBlocks + maxLocalBlocks)),
  arrayMaps(Expr::mkVar(
      Sort::arraySort(Sort::bvSort(bidBits),
        Sort::arraySort(Index::sort(), Float::sortFloat32())),
      "arrayMaps")),
  writableMaps(Expr::mkVar(
      Sort::arraySort(Sort::bvSort(bidBits), Sort::boolSort()),
      "writableMaps")),
  numelemMaps(Expr::mkVar(
      Sort::arraySort(Sort::bvSort(bidBits), Index::sort()),
      "numelemMaps"))
  {}

MemBlock SingleArrayMemory::getMemBlock(const Expr &bid) const {
  Expr array = arrayMaps.select(bid);
  Expr writable = writableMaps.select(bid);
  Expr numelem = numelemMaps.select(bid);
  return MemBlock(array, writable, numelem);
}

Expr SingleArrayMemory::addLocalBlock(
    const Expr &numelem, const Expr &writable) {
  assert(numLocalBlocks < maxLocalBlocks);

  auto bid = Expr::mkBV(numGlobalBlocks + numLocalBlocks, bidBits);
  numelemMaps = numelemMaps.store(bid, numelem);
  writableMaps = writableMaps.store(bid, writable);
  numLocalBlocks ++;
  return bid;
}

void SingleArrayMemory::setWritable(const Expr &bid, bool writable) {
  writableMaps = writableMaps.store(bid, Expr::mkBool(writable));
}

Expr SingleArrayMemory::getWritable(const Expr &bid) const {
  return writableMaps.select(bid);
}

Expr SingleArrayMemory::store(
    const Expr &f32val, const Expr &bid, const Expr &idx) {
  const auto block = getMemBlock(bid);
  arrayMaps = arrayMaps.store(bid, block.array.store(idx, f32val));
  return idx.ult(block.numelem) & block.writable;
}

Expr SingleArrayMemory::storeArray(
    const Expr &arr, const Expr &bid, const Expr &offset, const Expr &size,
    bool ubIfReadonly) {
  auto low = offset;
  auto high = offset + size - 1;
  auto idx = Index::var("idx", VarType::BOUND);
  auto arrayVal = arr.select((Expr)idx - low);

  auto block = getMemBlock(bid);
  auto currentVal = block.array.select(idx);
  auto cond = low.ule(idx) & ((Expr)idx).ule(high);
  auto stored = Expr::mkLambda(idx, Expr::mkIte(cond, arrayVal, currentVal));
  arrayMaps = arrayMaps.store(bid, stored);

  return Expr::mkAddNoOverflow(offset, size - 1, false) & // to prevent overflow
      high.ult(block.numelem) & // high < block.numelem
      (block.writable | !ubIfReadonly);
}

pair<Expr, Expr> SingleArrayMemory::load(
  const Expr &bid, const Expr &idx) const {
  const auto block = getMemBlock(bid);
  return {block.array.select(idx), idx.ult(block.numelem)};
}


MultipleArrayMemory::MultipleArrayMemory(
    unsigned int numGlobalBlocks, unsigned int maxLocalBlocks):
  MemoryCRTP(numGlobalBlocks, maxLocalBlocks,
    ulog2(numGlobalBlocks + maxLocalBlocks)) {
  for (unsigned i = 0; i < getNumBlocks(); ++i) {
    auto suffix = [&](const string &s) {
      return s + to_string(i);
    };
    arrays.push_back(Expr::mkVar(
        Sort::arraySort(Index::sort(), Float::sortFloat32()),
          suffix("array").c_str()));
    writables.push_back(
        Expr::mkVar(Sort::boolSort(),suffix("writable").c_str()));
    numelems.push_back(
        Expr::mkVar(Index::sort(), suffix("numelems").c_str()));
  }
}

Expr MultipleArrayMemory::itebid(
    const Expr &bid, function<Expr(unsigned)> fn) const {
  assert(getNumBlocks() > 0);
  assert(bid.sort().isBV() && bid.sort().bitwidth() == getBIDBits());

  uint64_t const_bid;
  if (bid.isUInt(const_bid))
    return fn(const_bid);

  const unsigned bits = bid.sort().bitwidth();

  Expr expr = fn(0);
  for (unsigned i = 1; i < getNumBlocks(); i ++)
    expr = Expr::mkIte(bid == Expr::mkBV(i, bits), fn(i), expr);

  return expr;
}

void MultipleArrayMemory::update(
    const Expr &bid, function<Expr*(unsigned)> getExprToUpdate,
    function<Expr(unsigned)> getUpdatedValue) const {
  assert(getNumBlocks() > 0);
  assert(bid.sort().isBV() && bid.sort().bitwidth() == getBIDBits());

  uint64_t const_bid;
  if (bid.isUInt(const_bid)) {
    *getExprToUpdate(const_bid) = getUpdatedValue(const_bid);
    return;
  }

  const unsigned bits = getBIDBits();
  for (unsigned i = 0; i < getNumBlocks(); ++i) {
    Expr *expr = getExprToUpdate(i);
    assert(expr);
    *expr = Expr::mkIte(bid == Expr::mkBV(i, bits), getUpdatedValue(i), *expr);
  }
}

Expr MultipleArrayMemory::addLocalBlock(
    const Expr &numelem, const Expr &writable) {
  assert(numLocalBlocks < maxLocalBlocks);

  auto bid = numGlobalBlocks + numLocalBlocks;
  auto suffix = [&](const string &s) {
    return s + to_string(bid) + (isSrc ? "_src" : "_tgt");
  };
  arrays.push_back(Expr::mkVar(
      Sort::arraySort(Index::sort(), Float::sortFloat32()),
                      suffix("array").c_str()));
  writables.push_back(writable);
  numelems.push_back(numelem);
  numLocalBlocks ++;
  return Expr::mkBV(bid, bidBits);
}

Expr MultipleArrayMemory::getNumElementsOfMemBlock(
    const Expr &bid) const {
  return itebid(bid, [&](auto ubid) { return numelems[ubid]; });
}

void MultipleArrayMemory::setWritable(const Expr &bid, bool writable) {
  update(bid, [&](unsigned ubid) { return &writables[ubid]; },
      [&](auto) { return Expr::mkBool(writable); });
}

Expr MultipleArrayMemory::getWritable(const Expr &bid) const {
  return itebid(bid, [&](auto ubid) { return writables[ubid]; });
}

Expr MultipleArrayMemory::store(const Expr &f32val,
    const Expr &bid, const Expr &idx) {
  update(bid, [&](auto ubid) { return &arrays[ubid]; },
      [&](auto ubid) { return arrays[ubid].store(idx, f32val); });

  return idx.ult(getNumElementsOfMemBlock(bid)) & getWritable(bid);
}

Expr MultipleArrayMemory::storeArray(
    const Expr &arr, const Expr &bid, const Expr &offset, const Expr &size,
    bool ubIfReadonly) {
  auto low = offset;
  auto high = offset + size - 1;
  auto idx = Index::var("idx", VarType::BOUND);
  auto arrayVal = arr.select((Expr)idx - low);

  update(bid, [&](auto ubid) { return &arrays[ubid]; },
    [&](auto ubid) {
      auto currentVal = arrays[ubid].select(idx);
      Expr cond = low.ule(idx) & ((Expr)idx).ule(high);
      return Expr::mkLambda(idx, Expr::mkIte(cond, arrayVal, currentVal));
    });

  return Expr::mkAddNoOverflow(offset, size - 1, false) & // to prevent overflow
      high.ult(getNumElementsOfMemBlock(bid)) & // high < block.numelem
      (getWritable(bid) | !ubIfReadonly);
}

pair<Expr, Expr> MultipleArrayMemory::load(
    unsigned ubid, const Expr &idx) const {
  assert(ubid < getNumBlocks());

  Expr success = idx.ult(getNumElementsOfMemBlock(ubid));
  return {arrays[ubid].select(idx), success};
}

pair<Expr, Expr> MultipleArrayMemory::load(
    const Expr &bid, const Expr &idx) const {
  Expr value = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).first; });
  Expr success = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).second; });
  return {value, success};
}

pair<Expr, vector<Expr>>
MultipleArrayMemory::refines(const Memory &other0) const {
  // NOTE: We cannot use dynamic_cast because we disabled -fno-rtti to link to
  // a plain LLVM.
  const MultipleArrayMemory &other =
      *static_cast<const MultipleArrayMemory *>(&other0);
  assert(other.numGlobalBlocks == numGlobalBlocks);

  // Create fresh, unbound variables
  auto bid = Expr::mkFreshVar(Sort::bvSort(bidBits), "bid");
  auto offset = Index::var("offset", VarType::FRESH);

  auto refines = [this, &other, &offset](unsigned ubid) {
    auto [srcValue, srcSuccess] = load(ubid, offset);
    auto srcWritable = getWritable(ubid);
    auto [tgtValue, tgtSuccess] = other.load(ubid, offset);
    auto tgtWritable = other.getWritable(ubid);

    auto wRefinement = srcWritable.implies(tgtWritable);
    auto vRefinement = (tgtValue == srcValue);
    return tgtSuccess.implies(srcSuccess & wRefinement & vRefinement);
  };

  Expr refinement = refines(0);
  for (unsigned i = 1; i < numGlobalBlocks; i ++)
    refinement = Expr::mkIte(
        bid == Expr::mkBV(i, bidBits), refines(i), refinement);

  return {refinement, {bid, offset}};
}
