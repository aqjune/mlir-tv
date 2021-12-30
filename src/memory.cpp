#include "debug.h"
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


AccessInfo AccessInfo::mkIte(const Expr &cond,
    const AccessInfo &lhs, const AccessInfo &rhs) {
  return {
    .inbounds = Expr::mkIte(cond, lhs.inbounds, rhs.inbounds),
    .liveness = Expr::mkIte(cond, lhs.liveness, rhs.liveness),
    .writable = Expr::mkIte(cond, lhs.writable, rhs.writable),
    .initialized = Expr::mkIte(cond, lhs.initialized, rhs.initialized)
  };
}

Expr AccessInfo::checkRead() const {
  return inbounds & liveness & initialized;
}

Expr AccessInfo::checkWrite(bool ignoreWritable) const {
  return inbounds & liveness & (ignoreWritable ? Expr::mkBool(true) : writable);
}

Expr AccessInfo::checkReadWrite() const {
  return inbounds & liveness & initialized & writable;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const AccessInfo &ai) {
  os << "(inbounds: " << ai.inbounds << ", liveness: " << ai.liveness
     << ", writable: " << ai.writable << ")\n";
  return os;
}


unsigned int Memory::getTotalNumBlocks() const {
  return sumSizes(arrays);
}

vector<mlir::Type> Memory::getBlockTypes() const {
  vector<mlir::Type> v;
  for (auto &[t, _]: numelems)
    v.push_back(t);
  return v;
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


static Expr checkInBounds(
    const Expr &offset, const Expr &size, const Expr &block_numelem) {
  return size.isZero() | // If size = 0, it does not touch the block
      // 1. If size != 0, no offset overflow
      (Expr::mkAddNoOverflow(offset, size - 1, false) &
      // 2. high < block.numelem
      (offset + size - 1).ult(block_numelem));
}

Memory::Memory(const TypeMap<size_t> &numGlobalBlocksPerType,
               const TypeMap<size_t> &maxNumLocalBlocksPerType,
               const vector<mlir::memref::GlobalOp> &globals,
               bool blocksInitiallyAlive):
    bidBits(calcBidBW(numGlobalBlocksPerType, maxNumLocalBlocksPerType)),
    globalBlocksCnt(numGlobalBlocksPerType),
    maxLocalBlocksCnt(maxNumLocalBlocksPerType),
    isSrc(true) {

  unsigned addedGlobalVars = 0;

  for (auto &[elemTy, numBlks]: globalBlocksCnt) {
    optional<Sort> elemSMTTy;

    if (elemTy.isa<mlir::FloatType>())
      elemSMTTy = Float::sort(elemTy);
    else if (elemTy.isIndex())
      elemSMTTy = Index::sort();
    else if (elemTy.isa<mlir::IntegerType>())
      elemSMTTy = Integer::sort(elemTy.getIntOrFloatBitWidth());

    if (!elemSMTTy)
      throw UnsupportedException(elemTy);

    vector<Expr> newArrs, newInits, newWrit, newNumElems, newLiveness;
    vector<mlir::memref::GlobalOp> globalsForTy;

    for (auto glb: globals) {
      if (glb.type().getElementType() == elemTy)
        globalsForTy.push_back(glb);
    }

    assert(globalsForTy.size() <= numBlks && "Too many global vars!");
    addedGlobalVars += globalsForTy.size();

    auto arrSort = Sort::arraySort(Index::sort(), *elemSMTTy);
    auto initSort = Sort::arraySort(Index::sort(), Sort::boolSort());

    for (unsigned i = 0; i < globalsForTy.size(); ++i) {
      auto glb = globalsForTy[i];
      auto res = globalVarBids.try_emplace(glb.getName().str(), elemTy, i);
      assert(res.second && "Duplicated global var name");

      verbose("memory init") << "Assigning bid = " << i << " to global var "
          << glb.getName() << "...\n";

      if (glb.constant()) {
        auto tensorTy = mlir::RankedTensorType::get(glb.type().getShape(),
            glb.type().getElementType());
        Tensor t = Tensor::fromElemsAttr(tensorTy, *glb.initial_value());
        newArrs.push_back(t.asArray());
      } else {
        string name = "#" + glb.getName().str() + "_array";
        newArrs.push_back(Expr::mkFreshVar(arrSort, name));
      }
      newInits.push_back(Expr::mkSplatArray(Index::sort(), Expr::mkBool(true)));
      newWrit.push_back(Expr::mkBool(!glb.constant()));
      newNumElems.push_back(Index(glb.type().getNumElements()));
      newLiveness.push_back(Expr::mkBool(true));
    }

    for (unsigned i = globalsForTy.size(); i < numBlks; ++i) {
      auto suffix2 = [&](const string &s) {
        return "#nonlocal-" + to_string(i) + "_" + s;
      };

      auto boolSort = Sort::boolSort();
      auto idxSort = Index::sort();
      newInits.push_back(Expr::mkFreshVar(initSort, suffix2("initialized")));
      newArrs.push_back(Expr::mkFreshVar(arrSort, suffix2("array")));
      newWrit.push_back(Expr::mkFreshVar(boolSort, suffix2("writable")));
      newNumElems.push_back(Expr::mkFreshVar(idxSort, suffix2("numelems")));

      if (blocksInitiallyAlive)
        newLiveness.push_back(Expr::mkBool(true));
      else
        newLiveness.push_back(Expr::mkFreshVar(boolSort, suffix2("liveness")));
    }

    arrays.insert({elemTy, move(newArrs)});
    initialized.insert({elemTy, move(newInits)});
    writables.insert({elemTy, move(newWrit)});
    numelems.insert({elemTy, move(newNumElems)});
    liveness.insert({elemTy, move(newLiveness)});
  }

  assert(addedGlobalVars == globals.size());
}

template<class T>
T Memory::itebid(mlir::Type elemTy, const Expr &bid, function<T(unsigned)> fn)
    const {
  assert(getNumBlocks(elemTy) > 0);
  assert(bid.sort().isBV() && bid.sort().bitwidth() == getBIDBits());

  uint64_t const_bid;
  if (bid.isUInt(const_bid))
    return fn(const_bid);

  const unsigned bits = bid.sort().bitwidth();

  T expr = fn(0);
  for (unsigned i = 1; i < getNumBlocks(elemTy); i ++)
    expr = T::mkIte(bid == Expr::mkBV(i, bits), fn(i), expr);

  return expr;
}

template<>
pair<Expr, AccessInfo> Memory::itebid<pair<Expr, AccessInfo>>(
    mlir::Type elemTy, const Expr &bid,
    function<pair<Expr, AccessInfo>(unsigned)> fn) const {
  auto fn1 = [&](unsigned bid) { return fn(bid).first; };
  auto fn2 = [&](unsigned bid) { return fn(bid).second; };

  return {itebid<Expr>(elemTy, bid, fn1), itebid<AccessInfo>(elemTy, bid, fn2)};
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
    return s + "#local-" + to_string(bid) + (isSrc ? "_src" : "_tgt");
  };

  arrays[elemTy].push_back(Expr::mkVar(
      Sort::arraySort(Index::sort(), *convertPrimitiveTypeToSort(elemTy)),
      suffix("array").c_str()));
  initialized[elemTy].push_back(
      Expr::mkSplatArray(Index::sort(), Expr::mkBool(false)));
  writables[elemTy].push_back(writable);
  numelems[elemTy].push_back(numelem);
  liveness[elemTy].push_back(Expr::mkBool(true));
  return Expr::mkBV(bid, bidBits);
}

Expr Memory::getNumElementsOfMemBlock(
    mlir::Type elemTy, const Expr &bid) const {
  return itebid<Expr>(elemTy, bid, [&](auto ubid) {
    return numelems.find(elemTy)->second[ubid];
  });
}

// Return the block id for the global variable having name.
unsigned Memory::getBidForGlobalVar(const string &name) const {
  auto itr = globalVarBids.find(name);
  assert(itr != globalVarBids.end());
  return itr->second.second;
}

// Return the name of the global var name having the element type and bid.
optional<string>
Memory::getGlobalVarName(mlir::Type elemTy, unsigned bid) const {
  for (auto &[k, itm]: globalVarBids) {
    if (itm.first == elemTy && itm.second == bid)
      return k;
  }
  return nullopt;
}

void Memory::setWritable(mlir::Type elemTy, const Expr &bid, bool writable) {
  update(elemTy, bid, [&](unsigned ubid) {
        return &writables.find(elemTy)->second[ubid]; },
      [&](auto) { return Expr::mkBool(writable); });
}

Expr Memory::getWritable(mlir::Type elemTy, const Expr &bid) const {
  return itebid<Expr>(elemTy, bid, [&](auto ubid) {
      return writables.find(elemTy)->second[ubid]; });
}

void Memory::setLivenessToFalse(mlir::Type elemTy, const Expr &bid) {
  update(elemTy, bid, [&](unsigned ubid) {
        return &liveness.find(elemTy)->second[ubid]; },
      [&](auto) { return Expr::mkBool(false); });
}

Expr Memory::getLiveness(mlir::Type elemTy, const Expr &bid) const {
  return itebid<Expr>(elemTy, bid, [&](auto ubid) {
      return liveness.find(elemTy)->second[ubid]; });
}
Expr Memory::isInitialized(mlir::Type elemTy,
    const Expr &bid, const Expr &ofs) const {
  return itebid<Expr>(elemTy, bid, [&](auto ubid) {
      return initialized.find(elemTy)->second[ubid].select(ofs); });
}

AccessInfo Memory::getInfo(
    mlir::Type elemTy, const Expr &bid, const Expr &idx) const {
  return {
    .inbounds = idx.ult(getNumElementsOfMemBlock(elemTy, bid)),
    .liveness = getLiveness(elemTy, bid),
    .writable = getWritable(elemTy, bid),
    .initialized = isInitialized(elemTy, bid, idx)
  };
}

AccessInfo Memory::getInfo(
    mlir::Type elemTy, const Expr &bid, const Expr &idx, const Expr &size)
    const {
  auto numelems = getNumElementsOfMemBlock(elemTy, bid);
  Expr j = Index::var("j", VarType::BOUND);
  return {
    .inbounds = checkInBounds(idx, size, numelems),
    .liveness = getLiveness(elemTy, bid),
    .writable = getWritable(elemTy, bid),
    .initialized = Expr::mkForall({j}, j.ult(size).implies(
        isInitialized(elemTy, bid, j + idx)))
  };
}

AccessInfo Memory::store(mlir::Type elemTy, const Expr &val,
    const Expr &bid, const Expr &idx) {
  update(elemTy, bid, [&](auto ubid) {
        return &arrays.find(elemTy)->second[ubid]; },
      [&](auto ubid) {
        return arrays.find(elemTy)->second[ubid].store(idx, val); });
  update(elemTy, bid, [&](auto ubid) {
        return &initialized.find(elemTy)->second[ubid]; },
      [&](auto ubid) {
        return initialized.find(elemTy)->second[ubid]
          .store(idx, Expr::mkBool(true)); });

  return getInfo(elemTy, bid, idx);
}

AccessInfo Memory::storeArray(
    mlir::Type elemTy, const Expr &arr, const Expr &bid, const Expr &offset,
    const Expr &size) {
  auto low = offset;
  auto high = offset + size - 1;
  auto idx = Index::var("idx", VarType::BOUND);
  auto arrayVal = arr.select((Expr)idx - low);

  update(elemTy, bid, [&](auto ubid) {
      return &arrays.find(elemTy)->second[ubid]; },
    [&](auto ubid) {
      uint64_t offset_const;
      const Expr &size_bid = this->numelems.find(elemTy)->second[ubid];
      if (offset.isUInt(offset_const) && offset_const == 0 &&
          (size == size_bid).isTrue())
        // Simply replace the previous value
        return arr;

      auto currentVal = arrays.find(elemTy)->second[ubid].select(idx);
      Expr cond = low.ule(idx) & ((Expr)idx).ule(high);
      return Expr::mkLambda(idx, Expr::mkIte(cond, arrayVal, currentVal));
    });
  update(elemTy, bid, [&](auto ubid) {
      return &initialized.find(elemTy)->second[ubid]; },
    [&](auto ubid) {
      auto currentVal = initialized.find(elemTy)->second[ubid].select(idx);
      Expr cond = low.ule(idx) & ((Expr)idx).ule(high);
      Expr trueVal = Expr::mkBool(true);
      return Expr::mkLambda(idx, Expr::mkIte(cond, trueVal, currentVal));
    });

  return getInfo(elemTy, bid, offset, size);
}


pair<Expr, AccessInfo> Memory::load(
    mlir::Type elemTy, const Expr &bid, const Expr &idx) const {
  return itebid<pair<Expr, AccessInfo>>(elemTy, bid,
      [&](unsigned ubid) -> pair<Expr, AccessInfo> {
    return {arrays.find(elemTy)->second[ubid].select(idx),
      getInfo(elemTy, mkBID(ubid), idx)};
  });
}

pair<Expr, AccessInfo> Memory::loadArray(
    mlir::Type elemTy, const Expr &bid, const Expr &ofs, const Expr &size) {
  return itebid<pair<Expr, AccessInfo>>(elemTy, bid,
      [&](unsigned ubid) -> pair<Expr, AccessInfo>{
    Expr idx0 = Index::var("arridx", VarType::BOUND);
    Expr arr = arrays.find(elemTy)->second[ubid];
    auto l = Expr::mkLambda({idx0}, arr.select(idx0 + ofs));
    return {l, getInfo(elemTy, mkBID(ubid), ofs, size)};
  });
}

TypeMap<pair<Expr, vector<Expr>>>
Memory::refines(const Memory &other) const {
  assert(globalBlocksCnt == other.globalBlocksCnt);

  // Create fresh, unbound variables
  auto refinesBlk = [this, &other](
      mlir::Type elemTy, unsigned ubid, Index offset) {
    auto [srcValue, srcInfo] = other.load(elemTy, mkBID(ubid), offset);
    auto srcWritable = srcInfo.writable;
    auto [tgtValue, tgtInfo] = load(elemTy, mkBID(ubid), offset);
    auto tgtWritable = tgtInfo.writable;

    auto wRefinement = srcWritable.implies(tgtWritable);
    auto [vRefinement, vRefParam] = ::refines(
        *fromExpr(move(tgtValue), elemTy), *fromExpr(move(srcValue), elemTy));
    assert(vRefParam.empty() && "Values stored in memory must be simple");

    return (srcInfo.inbounds & srcInfo.liveness).implies(
        tgtInfo.inbounds & tgtInfo.liveness & wRefinement & vRefinement);
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
