#pragma once

#include "smt.h"

#include <algorithm>
#include <vector>

enum MemEncoding {
  SINGLE_ARRAY, MULTIPLE_ARRAY
};

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

class Memory {
protected:
  const unsigned int numGlobalBlocks;
  const unsigned int maxLocalBlocks;
  const unsigned int bidBits;
  unsigned int numLocalBlocks;

public:
  static Memory * create(
      unsigned int numGlobalBlocks, unsigned int maxLocalBlocks,
      MemEncoding encoding);
  // Here we would like to use lower half of the memory blocks as global MemBlock
  // and upper half of the memory blocks as local MemBlock.
  // Memory refinement is defined only using global MemBlocks.
  Memory(unsigned int numGlobalBlocks,
      unsigned int maxLocalBlocks,
      unsigned int bidBits):
    numGlobalBlocks(numGlobalBlocks),
    maxLocalBlocks(maxLocalBlocks),
    bidBits(bidBits),
    numLocalBlocks(0) {}
  virtual ~Memory() {}

  unsigned int getBIDBits() const { return bidBits; }
  unsigned int getNumBlocks() const { return numGlobalBlocks + numLocalBlocks; }

  // Bids smaller than numGlobalBlocks are global (0 ~ numGlobalBlocks - 1)
  smt::Expr isGlobalBlock(const smt::Expr &bid) const;
  // Bids bigger than and equal to numGlobalBlocks are local blocks (numGlobalBlocks ~ numGlobalBlocks + numGlobalBlocks)
  smt::Expr isLocalBlock(const smt::Expr &bid) const;

  // Returns: (newly created block id)
  virtual smt::Expr addLocalBlock(const smt::Expr &numelem, const smt::Expr &writable) = 0;

  virtual smt::Expr getNumElementsOfMemBlock(const smt::Expr &bid) const = 0;
  // Mark memblock's writable flag to `writable`
  virtual void setWritable(const smt::Expr &bid, bool writable) = 0;
  // get memblocks' writable flag
  virtual smt::Expr getWritable(const smt::Expr &bid) const = 0;
  // Returns: store successful?
  virtual smt::Expr store(
      const smt::Expr &f32val, const smt::Expr &bid,
      const smt::Expr &idx) = 0;
  // Returns: store successful?
  virtual smt::Expr storeArray(
      const smt::Expr &arr, const smt::Expr &bid,
      const smt::Expr &offset, const smt::Expr &size) = 0;
  // Returns: (loaded value, load successful?)
  virtual std::pair<smt::Expr, smt::Expr> load(
      const smt::Expr &bid, const smt::Expr &idx) const = 0;

  // Encode the refinement relation between src (other) and tgt (this) memory
  virtual std::pair<smt::Expr, std::vector<smt::Expr>>
    refines(const Memory &other) const = 0;
};

class SingleArrayMemory: public Memory {
  smt::Expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  smt::Expr writableMaps; // bv(bits)::sort() -> bool::sort()
  smt::Expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const smt::Expr &bid) const;

public:
  SingleArrayMemory(unsigned int globalBlocks, unsigned int localBlocks);

  smt::Expr addLocalBlock(const smt::Expr &numelem, const smt::Expr &writable) override;

  smt::Expr getNumElementsOfMemBlock(const smt::Expr &bid) const override {
    return getMemBlock(bid).numelem;
  }

  void setWritable(const smt::Expr &bid, bool writable) override;
  smt::Expr getWritable(const smt::Expr &bid) const override;
  smt::Expr store(
      const smt::Expr &f32val, const smt::Expr &bid, const smt::Expr &idx)
      override;
  smt::Expr storeArray(
      const smt::Expr &arr, const smt::Expr &bid, const smt::Expr &offset, const smt::Expr &size)
      override;
  std::pair<smt::Expr, smt::Expr> load(
      const smt::Expr &bid, const smt::Expr &idx) const override;

  std::pair<smt::Expr, std::vector<smt::Expr>> refines(const Memory &other)
      const override;
};


// A class that implements the memory model described in CAV'21 (An SMT
// Encoding of LLVM's Memory Model for Bounded Translation Validation)
class MultipleArrayMemory: public Memory {
  std::vector<smt::Expr> arrays;  // vector<(Index::sort() -> Float::sort())>
  std::vector<smt::Expr> writables; // vector<Bool::sort()>
  std::vector<smt::Expr> numelems;  // vector<Index::sort>

public:
  MultipleArrayMemory(unsigned int globalBlocks, unsigned int localBlocks);

  smt::Expr addLocalBlock(const smt::Expr &numelem, const smt::Expr &writable) override;

  smt::Expr getNumElementsOfMemBlock(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return numelems[ubid]; }
  smt::Expr getNumElementsOfMemBlock(const smt::Expr &bid) const override;

  void setWritable(const smt::Expr &bid, bool writable) override;
  smt::Expr getWritable(const smt::Expr &bid) const override;
  smt::Expr getWritable(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return writables[ubid]; }

  smt::Expr store(
      const smt::Expr &f32val, const smt::Expr &bid, const smt::Expr &idx)
      override;
  smt::Expr storeArray(
      const smt::Expr &arr, const smt::Expr &bid, const smt::Expr &offset, const smt::Expr &size)
      override;
  std::pair<smt::Expr, smt::Expr> load(
      const smt::Expr &bid, const smt::Expr &idx) const override;
  std::pair<smt::Expr, smt::Expr> load(unsigned ubid, const smt::Expr &idx)
      const;

  std::pair<smt::Expr, std::vector<smt::Expr>> refines(
      const Memory &other) const override;

private:
  smt::Expr itebid(
      const smt::Expr &bid, std::function<smt::Expr(unsigned)> fn) const;
  void update(
      const smt::Expr &bid,
      std::function<smt::Expr*(unsigned)> exprToUpdate, // bid -> ptr to expr
      std::function<smt::Expr(unsigned)> updatedValue) const; // bid -> updated
};
