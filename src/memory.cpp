#include "memory.h"
#include "smt.h"
#include "utils.h"
#include "value.h"
#include <string>

using namespace smt;
using namespace std;

static unsigned int ulog2(unsigned int numBlocks) {
  if (numBlocks == 0)
    return 0;
  return (unsigned int) ceil(log2(std::max(numBlocks, (unsigned int) 2)));
}

Expr Memory::isGlobalBlock(const Expr &bid) const {
  return bid.ult(numGlobalBlocks);
}

Expr Memory::isLocalBlock(const Expr &bid) const {
  return !isGlobalBlock(bid);
}


static Expr isSafeToWrite(
    const Expr &offset, const Expr &size, const Expr &block_numelem,
    const Expr &block_writable, bool ubIfReadonly) {
  return size.isZero() | // If size = 0, it does not touch the block
      // 1. If size != 0, no offset overflow
      (Expr::mkAddNoOverflow(offset, size - 1, false) &
      // 2. high < block.numelem
       (offset + size - 1).ult(block_numelem) &
      // 3. Can write
      (block_writable | !ubIfReadonly));
}


Memory::Memory(unsigned int numGlobalBlocks, unsigned int maxLocalBlocks):
    numGlobalBlocks(numGlobalBlocks),
    maxLocalBlocks(maxLocalBlocks),
    bidBits(ulog2(numGlobalBlocks + maxLocalBlocks)),
    numLocalBlocks(0),
    isSrc(true) {
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

Expr Memory::itebid(
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

void Memory::update(
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

Expr Memory::addLocalBlock(
    const Expr &numelem, const Expr &writable) {
  if (numLocalBlocks >= maxLocalBlocks)
    throw UnsupportedException("Too many local blocks");

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

Expr Memory::getNumElementsOfMemBlock(
    const Expr &bid) const {
  return itebid(bid, [&](auto ubid) { return numelems[ubid]; });
}

void Memory::setWritable(const Expr &bid, bool writable) {
  update(bid, [&](unsigned ubid) { return &writables[ubid]; },
      [&](auto) { return Expr::mkBool(writable); });
}

Expr Memory::getWritable(const Expr &bid) const {
  return itebid(bid, [&](auto ubid) { return writables[ubid]; });
}

Expr Memory::store(const Expr &f32val,
    const Expr &bid, const Expr &idx) {
  update(bid, [&](auto ubid) { return &arrays[ubid]; },
      [&](auto ubid) { return arrays[ubid].store(idx, f32val); });

  return idx.ult(getNumElementsOfMemBlock(bid)) & getWritable(bid);
}

Expr Memory::storeArray(
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

  return isSafeToWrite(offset, size, getNumElementsOfMemBlock(bid),
      getWritable(bid), ubIfReadonly);
}

pair<Expr, Expr> Memory::load(
    unsigned ubid, const Expr &idx) const {
  assert(ubid < getNumBlocks());

  Expr success = idx.ult(getNumElementsOfMemBlock(ubid));
  return {arrays[ubid].select(idx), success};
}

pair<Expr, Expr> Memory::load(
    const Expr &bid, const Expr &idx) const {
  Expr value = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).first; });
  Expr success = itebid(bid,
      [&](unsigned ubid) { return load(ubid, idx).second; });
  return {value, success};
}

pair<Expr, vector<Expr>>
Memory::refines(const Memory &other) const {
  assert(other.numGlobalBlocks == numGlobalBlocks);

  // Create fresh, unbound variables
  auto bid = Expr::mkFreshVar(Sort::bvSort(bidBits), "bid");
  auto offset = Index::var("offset", VarType::FRESH);

  auto refinesBlk = [this, &other, &offset](unsigned ubid) {
    auto [srcValue, srcSuccess] = other.load(ubid, offset);
    auto srcWritable = other.getWritable(ubid);
    auto [tgtValue, tgtSuccess] = load(ubid, offset);
    auto tgtWritable = getWritable(ubid);

    auto wRefinement = srcWritable.implies(tgtWritable);
    auto vRefinement = (tgtValue == srcValue);
    return tgtSuccess.implies(srcSuccess & wRefinement & vRefinement);
  };

  Expr refinement = refinesBlk(0);
  for (unsigned i = 1; i < numGlobalBlocks; i ++)
    refinement = Expr::mkIte(
        bid == Expr::mkBV(i, bidBits), refinesBlk(i), refinement);

  return {refinement, {bid, offset}};
}
