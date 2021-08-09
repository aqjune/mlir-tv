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
  smt::expr array;    // Index::sort() -> Float::sort()
  smt::expr writable; // bool::sort()
  smt::expr numelem;  // Index::sort()

  MemBlock(const smt::expr &array, const smt::expr &writable, const smt::expr &numelem):
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
  smt::expr isGlobalBlock(const smt::expr &bid) const;
  // Bids bigger than and equal to numGlobalBlocks are local blocks (numGlobalBlocks ~ numGlobalBlocks + numGlobalBlocks)
  smt::expr isLocalBlock(const smt::expr &bid) const;

  // Returns: (newly created block id)
  virtual smt::expr addLocalMemBlock(const smt::expr &numelem) = 0;

  virtual smt::expr getNumElementsOfMemBlock(const smt::expr &bid) const = 0;
  // Mark memblock's writable flag to `writable`
  virtual void setWritable(const smt::expr &bid, bool writable) = 0;
  // get memblocks' writable flag
  virtual smt::expr getWritable(const smt::expr &bid) const = 0;
  // Returns: store successful?
  virtual smt::expr store(
      const smt::expr &f32val, const smt::expr &bid,
      const smt::expr &idx) = 0;
  // Returns: (loaded value, load successful?)
  virtual std::pair<smt::expr, smt::expr> load(
      const smt::expr &bid, const smt::expr &idx) const = 0;

  // Encode the refinement relation between src (other) and tgt (this) memory
  virtual std::pair<smt::expr, std::vector<smt::expr>>
    refines(const Memory &other) const = 0;
};

class SingleArrayMemory: public Memory {
  smt::expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  smt::expr writableMaps; // bv(bits)::sort() -> bool::sort()
  smt::expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const smt::expr &bid) const;

public:
  SingleArrayMemory(unsigned int globalBlocks, unsigned int localBlocks);

  smt::expr addLocalMemBlock(const smt::expr &numelem) override;

  smt::expr getNumElementsOfMemBlock(const smt::expr &bid) const override {
    return getMemBlock(bid).numelem;
  }

  void setWritable(const smt::expr &bid, bool writable) override;
  smt::expr getWritable(const smt::expr &bid) const override;
  smt::expr store(
      const smt::expr &f32val, const smt::expr &bid, const smt::expr &idx)
      override;
  std::pair<smt::expr, smt::expr> load(
      const smt::expr &bid, const smt::expr &idx) const override;

  std::pair<smt::expr, std::vector<smt::expr>> refines(const Memory &other)
      const override;
};


// A class that implements the memory model described in CAV'21 (An SMT
// Encoding of LLVM's Memory Model for Bounded Translation Validation)
class MultipleArrayMemory: public Memory {
  std::vector<smt::expr> arrays;  // vector<(Index::sort() -> Float::sort())>
  std::vector<smt::expr> writables; // vector<Bool::sort()>
  std::vector<smt::expr> numelems;  // vector<Index::sort>

public:
  MultipleArrayMemory(unsigned int globalBlocks, unsigned int localBlocks);

  smt::expr addLocalMemBlock(const smt::expr &numelem) override;

  smt::expr getNumElementsOfMemBlock(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return numelems[ubid]; }
  smt::expr getNumElementsOfMemBlock(const smt::expr &bid) const override;

  void setWritable(const smt::expr &bid, bool writable) override;
  smt::expr getWritable(const smt::expr &bid) const override;
  smt::expr getWritable(unsigned ubid) const
  { assert(ubid < getNumBlocks()); return writables[ubid]; }

  smt::expr store(
      const smt::expr &f32val, const smt::expr &bid, const smt::expr &idx)
      override;
  std::pair<smt::expr, smt::expr> load(
      const smt::expr &bid, const smt::expr &idx) const override;
  std::pair<smt::expr, smt::expr> load(unsigned ubid, const smt::expr &idx)
      const;

  std::pair<smt::expr, std::vector<smt::expr>> refines(
      const Memory &other) const override;

private:
  smt::expr itebid(
      const smt::expr &bid, std::function<smt::expr(unsigned)> fn) const;
  void update(
      const smt::expr &bid,
      std::function<smt::expr*(unsigned)> exprToUpdate, // bid -> ptr to expr
      std::function<smt::expr(unsigned)> updatedValue) const; // bid -> updated
};
