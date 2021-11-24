#pragma once

#include "smt.h"

#include <algorithm>
#include <vector>

// A memory block containing f32 elements.
class MemBlock {
public:
  smt::Expr array;    // Index::sort() -> Float::sort()
  smt::Expr writable; // bool::sort()
  smt::Expr numelem;  // Index::sort()

  MemBlock(const smt::Expr &array, const smt::Expr &writable,
           const smt::Expr &numelem):
    array(array), writable(writable), numelem(numelem) {}
};

// A class that implements the memory model described in CAV'21 (An SMT
// Encoding of LLVM's Memory Model for Bounded Translation Validation)
class Memory {
  const unsigned int numGlobalBlocks;
  const unsigned int maxLocalBlocks;
  // Bid: we use lower half of the memory blocks as global MemBlock
  // and upper half of the memory blocks as local MemBlock.
  // Memory refinement is defined only using global MemBlocks.
  const unsigned int bidBits;
  unsigned int numLocalBlocks;
  bool isSrc;

  std::vector<smt::Expr> arrays;  // vector<(Index::sort() -> Float::sort())>
  std::vector<smt::Expr> writables; // vector<Bool::sort()>
  std::vector<smt::Expr> numelems;  // vector<Index::sort>

public:
  Memory(unsigned int globalBlocks, unsigned int localBlocks);

  void setIsSrc(bool flag) { isSrc = flag; }

  unsigned int getBIDBits() const { return bidBits; }
  unsigned int getNumBlocks() const { return numGlobalBlocks + numLocalBlocks; }

  // Bids smaller than numGlobalBlocks are global (0 ~ numGlobalBlocks - 1)
  smt::Expr isGlobalBlock(const smt::Expr &bid) const;
  // Bids bigger than and equal to numGlobalBlocks are local blocks
  // (numGlobalBlocks ~ numGlobalBlocks + numGlobalBlocks)
  smt::Expr isLocalBlock(const smt::Expr &bid) const;

  // Returns: (newly created block id)
  smt::Expr addLocalBlock(const smt::Expr &numelem,
      const smt::Expr &writable);

  smt::Expr getNumElementsOfMemBlock(const smt::Expr &bid) const;
  smt::Expr getNumElementsOfMemBlock(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return numelems[ubid]; }
  // Mark memblock's writable flag to `writable`
  void setWritable(const smt::Expr &bid, bool writable);
  // get memblocks' writable flag
  smt::Expr getWritable(const smt::Expr &bid) const;
  smt::Expr getWritable(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return writables[ubid]; }
  // Returns: store successful?
  smt::Expr store(
      const smt::Expr &f32val, const smt::Expr &bid,
      const smt::Expr &idx);
  // Returns: store successful?
  smt::Expr storeArray(
      const smt::Expr &arr, const smt::Expr &bid,
      const smt::Expr &offset, const smt::Expr &size,
      bool ubIfReadonly = true);
  // Returns: (loaded value, load successful?)
  std::pair<smt::Expr, smt::Expr> load(
      const smt::Expr &bid, const smt::Expr &idx) const;
  std::pair<smt::Expr, smt::Expr> load(unsigned bid, const smt::Expr &idx)
      const;

  // Encode the refinement relation between src (other) and tgt (this) memory
  std::pair<smt::Expr, std::vector<smt::Expr>>
      refines(const Memory &other) const;

  Memory *clone() const { return new Memory(*this); }

private:
  smt::Expr itebid(
      const smt::Expr &bid, std::function<smt::Expr(unsigned)> fn) const;
  void update(
      const smt::Expr &bid,
      std::function<smt::Expr*(unsigned)> exprToUpdate, // bid -> ptr to expr
      std::function<smt::Expr(unsigned)> updatedValue) const; // bid -> updated
};
