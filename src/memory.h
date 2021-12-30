#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "smt.h"
#include "utils.h"

#include <algorithm>
#include <vector>

struct AccessInfo {
  smt::Expr inbounds; // is the index inbounds?
  smt::Expr liveness; // is block alive?
  smt::Expr writable; // Is the block writable?
  smt::Expr initialized; // Is the element initialized?

  static AccessInfo mkIte(const smt::Expr &cond,
      const AccessInfo &lhs, const AccessInfo &rhs);

  smt::Expr checkRead() const;
  smt::Expr checkWrite(bool ignoreWritable = false) const;
  smt::Expr checkReadWrite() const;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const AccessInfo &);

// A class that implements the memory model described in CAV'21 (An SMT
// Encoding of LLVM's Memory Model for Bounded Translation Validation)
// In addition, blocks of different types don't alias. This is for abstractly
// encoding floating points.
class Memory {
  const unsigned int bidBits;
  bool isSrc;

  TypeMap<size_t> globalBlocksCnt;
  TypeMap<size_t> maxLocalBlocksCnt;

  // element type -> vector<(Index::sort() -> The element's SMT type)>
  TypeMap<std::vector<smt::Expr>> arrays;
  // element type -> vector<(Index::sort() -> bool)>
  TypeMap<std::vector<smt::Expr>> initialized;
  // element type -> vector<Bool::sort()>
  TypeMap<std::vector<smt::Expr>> writables;
  // element type -> vector<Index::sort>
  TypeMap<std::vector<smt::Expr>> numelems;
  // element type -> vector<Bool::sort()>
  TypeMap<std::vector<smt::Expr>> liveness;

  // (Element type, bid) of global variables.
  std::map<std::string, std::pair<mlir::Type, unsigned>> globalVarBids;

public:
  Memory(const TypeMap<size_t> &numGlobalBlocksPerType,
         const TypeMap<size_t> &maxNumLocalBlocksPerType,
         const std::vector<mlir::memref::GlobalOp> &globals,
         bool blocksInitiallyAlive = false);

  void setIsSrc(bool flag) { isSrc = flag; }

  unsigned int getBIDBits() const { return bidBits; }
  smt::Expr mkBID(unsigned ubid) const
  { return smt::Expr::mkBV(ubid, bidBits); }

  unsigned int getTotalNumBlocks() const;
  unsigned int getNumBlocks(mlir::Type elemTy) const {
    auto itr = arrays.find(elemTy);
    assert(itr != arrays.end());
    return itr->second.size();
  }
  std::vector<mlir::Type> getBlockTypes() const;

  // Bids smaller than numGlobalBlocks are global (0 ~ numGlobalBlocks - 1)
  smt::Expr isGlobalBlock(mlir::Type elemType, const smt::Expr &bid) const;
  // Bids bigger than and equal to numGlobalBlocks are local blocks
  // (numGlobalBlocks ~ numGlobalBlocks + numGlobalBlocks)
  smt::Expr isLocalBlock(mlir::Type elemType, const smt::Expr &bid) const;

  // Returns: (newly created block id)
  smt::Expr addLocalBlock(const smt::Expr &numelem, mlir::Type elemTy,
      const smt::Expr &writable);

  smt::Expr getNumElementsOfMemBlock(mlir::Type elemTy, const smt::Expr &bid)
      const;

  // Return the block id for the global variable having name.
  unsigned getBidForGlobalVar(const std::string &name) const;
  // Return the name of the global var name having the element type and bid.
  std::optional<std::string> getGlobalVarName(mlir::Type elemTy, unsigned bid)
    const;

  // Mark memblock's writable flag to `writable`
  void setWritable(mlir::Type elemTy, const smt::Expr &bid, bool writable);
  // get memblocks' writable flag
  smt::Expr getWritable(mlir::Type elemTy, const smt::Expr &bid) const;

  // Mark memblock's liveness to false.
  void setLivenessToFalse(mlir::Type elemTy, const smt::Expr &bid);
  // get memblocks' writable flag
  smt::Expr getLiveness(mlir::Type elemTy, const smt::Expr &bid) const;

  smt::Expr isInitialized(mlir::Type elemTy,
      const smt::Expr &bid, const smt::Expr &ofs) const;

  AccessInfo store(
      mlir::Type elemTy, const smt::Expr &val, const smt::Expr &bid,
      const smt::Expr &idx);
  AccessInfo storeArray(
      mlir::Type elemTy, const smt::Expr &arr, const smt::Expr &bid,
      const smt::Expr &offset, const smt::Expr &size);

  // Returns: (loaded value, load successful?)
  std::pair<smt::Expr, AccessInfo> load(
      mlir::Type elemTy, const smt::Expr &bid, const smt::Expr &idx) const;
  std::pair<smt::Expr, AccessInfo> loadArray(
      mlir::Type elemTy, const smt::Expr &bid, const smt::Expr &idx,
      const smt::Expr &size);

  // Encode the refinement relation between src (other) and tgt (this) memory
  // for each element type.
  // Memory refinement is defined using global memory blocks only.
  TypeMap<std::pair<smt::Expr, std::vector<smt::Expr>>>
      refines(const Memory &other) const;

  Memory *clone() const { return new Memory(*this); }

private:
  template<class T>
  T itebid(
      mlir::Type elemTy, const smt::Expr &bid,
      std::function<T(unsigned)> fn) const;
  void update(
      mlir::Type elemTy, const smt::Expr &bid,
      std::function<smt::Expr*(unsigned)> exprToUpdate, // bid -> ptr to expr
      std::function<smt::Expr(unsigned)> updatedValue) const; // bid -> updated

  AccessInfo getInfo(mlir::Type elemTy, const smt::Expr &bid,
      const smt::Expr &ofs) const;
  AccessInfo getInfo(mlir::Type elemTy, const smt::Expr &bid,
      const smt::Expr &ofs, const smt::Expr &accessSize) const;

  size_t getMaxNumLocalBlocks(mlir::Type ty) const {
    auto itr = maxLocalBlocksCnt.find(ty);
    assert(itr != maxLocalBlocksCnt.end());
    return itr->second;
  }
  size_t getNumGlobalBlocks(mlir::Type ty) const {
    auto itr = globalBlocksCnt.find(ty);
    assert(itr != globalBlocksCnt.end());
    return itr->second;
  }
};
