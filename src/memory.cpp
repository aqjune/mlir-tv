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

template<class T>
static size_t sumSizes(const TypeMap<vector<T>> &m) {
  size_t s = 0;
  for (auto &[k, v]: m)
    s += v.size();
  return s;
}

static size_t calcBidBW(
    const TypeMap<size_t> &numGlobalBlocksPerType,
    const TypeMap<size_t> &maxNumLocalBlocksPerType) {
  size_t bw = 0;
  for (auto &[ty, n]: numGlobalBlocksPerType) {
    size_t n2 = n;
    auto itr = maxNumLocalBlocksPerType.find(ty);
    if (itr != maxNumLocalBlocksPerType.end())
      n2 += itr->second;

    bw = max(bw, (size_t)ulog2(n2));
  }

  for (auto &[ty, n]: maxNumLocalBlocksPerType)
    bw = max(bw, (size_t)ulog2(n));
  return bw;
}


unsigned int Memory::getTotalNumBlocks() const {
  return sumSizes(arrays);
}

Expr Memory::isGlobalBlock(mlir::Type elemTy, const Expr &bid) const {
  auto itr = globalBlocksCnt.find(elemTy);
  if (itr == globalBlocksCnt.end())
    return Expr::mkBool(false);

  return bid.ult(itr->second);
}

Expr Memory::isLocalBlock(mlir::Type elemTy, const Expr &bid) const {
  return !isGlobalBlock(elemTy, bid);
}


// Liveness of the block must be checked by callers
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

Memory::Memory(TypeMap<size_t> numGlobalBlocksPerType,
               TypeMap<size_t> maxNumLocalBlocksPerType):
    bidBits(calcBidBW(numGlobalBlocksPerType, maxNumLocalBlocksPerType)),
    globalBlocksCnt(numGlobalBlocksPerType),
    maxLocalBlocksCnt(maxNumLocalBlocksPerType),
    isSrc(true) {

  for (auto &[elemTy, numBlks]: numGlobalBlocksPerType) {
    optional<Sort> elemSMTTy;

    if (elemTy.isa<mlir::FloatType>())
      elemSMTTy = Float::sort(elemTy);
    else if (elemTy.isIndex())
      elemSMTTy = Index::sort();
    else if (elemTy.isa<mlir::IntegerType>())
      elemSMTTy = Integer::sort(elemTy.getIntOrFloatBitWidth());

    if (!elemSMTTy)
      throw UnsupportedException(elemTy);

    vector<Expr> newArrs, newWrit, newNumElems, newLiveness;

    for (unsigned i = 0; i < numBlks; ++i) {
      auto suffix = [&](const string &s) {
        return s + "_" + to_string(i);
      };

      newArrs.push_back(Expr::mkVar(
          Sort::arraySort(Index::sort(), *elemSMTTy),
            suffix("array").c_str()));
      newWrit.push_back(
          Expr::mkVar(Sort::boolSort(),suffix("writable").c_str()));
      newNumElems.push_back(
          Expr::mkVar(Index::sort(), suffix("numelems").c_str()));
      newLiveness.push_back(
          Expr::mkVar(Sort::boolSort(), suffix("liveness").c_str()));
    }

    arrays.insert({elemTy, move(newArrs)});
    writables.insert({elemTy, move(newWrit)});
    numelems.insert({elemTy, move(newNumElems)});
    liveness.insert({elemTy, move(newLiveness)});
  }
}

Expr Memory::itebid(
    mlir::Type elemTy, const Expr &bid, function<Expr(unsigned)> fn) const {
  assert(getNumBlocks(elemTy) > 0);
  assert(bid.sort().isBV() && bid.sort().bitwidth() == getBIDBits());

  uint64_t const_bid;
  if (bid.isUInt(const_bid))
    return fn(const_bid);

  const unsigned bits = bid.sort().bitwidth();

  Expr expr = fn(0);
  for (unsigned i = 1; i < getNumBlocks(elemTy); i ++)
    expr = Expr::mkIte(bid == Expr::mkBV(i, bits), fn(i), expr);

  return expr;
}

void Memory::update(
    mlir::Type elemTy, const Expr &bid,
    function<Expr*(unsigned)> getExprToUpdate,
    function<Expr(unsigned)> getUpdatedValue) const {
  assert(getNumBlocks(elemTy) > 0);
  assert(bid.sort().isBV() && bid.sort().bitwidth() == getBIDBits());

  uint64_t const_bid;
  if (bid.isUInt(const_bid)) {
    *getExprToUpdate(const_bid) = getUpdatedValue(const_bid);
    return;
  }

  const unsigned bits = getBIDBits();
  for (unsigned i = 0; i < getNumBlocks(elemTy); ++i) {
    Expr *expr = getExprToUpdate(i);
    assert(expr);
    *expr = Expr::mkIte(bid == Expr::mkBV(i, bits), getUpdatedValue(i), *expr);
  }
}

Expr Memory::addLocalBlock(
    const Expr &numelem, mlir::Type elemTy, const Expr &writable) {
  auto bid = getNumBlocks(elemTy);
  if (bid >= getNumGlobalBlocks(elemTy) + getMaxNumLocalBlocks(elemTy))
    throw UnsupportedException("Too many local blocks");

  auto suffix = [&](const string &s) {
    return s + to_string(bid) + (isSrc ? "_src" : "_tgt");
  };

  arrays[elemTy].push_back(Expr::mkVar(
      Sort::arraySort(Index::sort(), *convertPrimitiveTypeToSort(elemTy)),
      suffix("array").c_str()));
  writables[elemTy].push_back(writable);
  numelems[elemTy].push_back(numelem);
  liveness[elemTy].push_back(Expr::mkBool(true));
  return Expr::mkBV(bid, bidBits);
}

Expr Memory::getNumElementsOfMemBlock(
    mlir::Type elemTy, const Expr &bid) const {
  return itebid(elemTy, bid, [&](auto ubid) {
    return numelems.find(elemTy)->second[ubid];
  });
}

void Memory::setWritable(mlir::Type elemTy, const Expr &bid, bool writable) {
  update(elemTy, bid, [&](unsigned ubid) {
        return &writables.find(elemTy)->second[ubid]; },
      [&](auto) { return Expr::mkBool(writable); });
}

Expr Memory::getWritable(mlir::Type elemTy, const Expr &bid) const {
  return itebid(elemTy, bid, [&](auto ubid) {
      return writables.find(elemTy)->second[ubid]; });
}

void Memory::setLivenessToFalse(mlir::Type elemTy, const Expr &bid) {
  update(elemTy, bid, [&](unsigned ubid) {
        return &liveness.find(elemTy)->second[ubid]; },
      [&](auto) { return Expr::mkBool(false); });
}

Expr Memory::getLiveness(mlir::Type elemTy, const Expr &bid) const {
  return itebid(elemTy, bid, [&](auto ubid) {
      return liveness.find(elemTy)->second[ubid]; });
}

Expr Memory::store(mlir::Type elemTy, const Expr &val,
    const Expr &bid, const Expr &idx) {
  update(elemTy, bid, [&](auto ubid) {
        return &arrays.find(elemTy)->second[ubid]; },
      [&](auto ubid) {
        return arrays.find(elemTy)->second[ubid].store(idx, val); });

  return idx.ult(getNumElementsOfMemBlock(elemTy, bid)) &
      getWritable(elemTy, bid) & getLiveness(elemTy, bid);
}

Expr Memory::storeArray(
    mlir::Type elemTy, const Expr &arr, const Expr &bid, const Expr &offset,
    const Expr &size, bool ubIfReadonly) {
  auto low = offset;
  auto high = offset + size - 1;
  auto idx = Index::var("idx", VarType::BOUND);
  auto arrayVal = arr.select((Expr)idx - low);

  update(elemTy, bid, [&](auto ubid) {
      return &arrays.find(elemTy)->second[ubid]; },
    [&](auto ubid) {
      auto currentVal = arrays.find(elemTy)->second[ubid].select(idx);
      Expr cond = low.ule(idx) & ((Expr)idx).ule(high);
      return Expr::mkLambda(idx, Expr::mkIte(cond, arrayVal, currentVal));
    });

  return isSafeToWrite(offset, size, getNumElementsOfMemBlock(elemTy, bid),
      getWritable(elemTy, bid), ubIfReadonly) & getLiveness(elemTy, bid);
}

pair<Expr, Expr> Memory::load(
    mlir::Type elemTy, unsigned ubid, const Expr &idx) const {
  assert(ubid < getNumBlocks(elemTy));

  Expr success = idx.ult(getNumElementsOfMemBlock(elemTy, ubid)) &
      getLiveness(elemTy, ubid);
  return {arrays.find(elemTy)->second[ubid].select(idx), success};
}

pair<Expr, Expr> Memory::load(
    mlir::Type elemTy, const Expr &bid, const Expr &idx) const {
  Expr value = itebid(elemTy, bid,
      [&](unsigned ubid) { return load(elemTy, ubid, idx).first; });
  Expr success = itebid(elemTy, bid,
      [&](unsigned ubid) { return load(elemTy, ubid, idx).second; });
  return {value, success};
}

TypeMap<pair<Expr, vector<Expr>>>
Memory::refines(const Memory &other) const {
  assert(globalBlocksCnt == other.globalBlocksCnt);

  // Create fresh, unbound variables
  auto refinesBlk = [this, &other](
      mlir::Type elemTy, unsigned ubid, Index offset) {
    auto [srcValue, srcSuccess] = other.load(elemTy, ubid, offset);
    auto srcWritable = other.getWritable(elemTy, ubid);
    auto [tgtValue, tgtSuccess] = load(elemTy, ubid, offset);
    auto tgtWritable = getWritable(elemTy, ubid);

    auto wRefinement = srcWritable.implies(tgtWritable);
    auto vRefinement = (tgtValue == srcValue);
    return tgtSuccess.implies(srcSuccess & wRefinement & vRefinement);
  };

  using ElemTy = pair<Expr, vector<Expr>>;
  TypeMap<ElemTy> tmap;

  for (auto &[ty, numblks]: globalBlocksCnt) {
    Expr refinement = Expr::mkBool(true);
    auto bid = Expr::mkFreshVar(Sort::bvSort(bidBits), "bid_" + to_string(ty));
    auto offset = Index::var("offset_" + to_string(ty), VarType::FRESH);

    for (unsigned i = 0; i < numblks; i ++)
      refinement = Expr::mkIte(
          bid == Expr::mkBV(i, bidBits), refinesBlk(ty, i, offset), refinement);

    vector<Expr> params{bid, offset};
    ElemTy elem = {move(refinement), move(params)};
    tmap.try_emplace(ty, move(elem));
  }

  return tmap;
}
