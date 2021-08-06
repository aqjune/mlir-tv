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
  const unsigned int globalBlocks;
  const unsigned int localBlocks;
  const unsigned int bidBits;

public:
  static Memory * create(
      unsigned int globalBlocks, unsigned int localBlocks,
      MemEncoding encoding);
  // Here we would like to use lower half of the memory blocks as global MemBlock
  // and upper half of the memory blocks as local MemBlock.
  // Memory refinement is defined only using global MemBlocks.
  Memory(unsigned int globalBlocks, unsigned int localBlocks, unsigned int bidBits):
      globalBlocks(globalBlocks), localBlocks(localBlocks), bidBits(bidBits) {}
  virtual ~Memory() {}

  // Encode the refinement relation between src (other) and tgt (this) memory
  virtual std::pair<smt::expr, std::vector<smt::expr>>
    refines(const Memory &other) const = 0;

  unsigned int getBIDBits() const { return bidBits; }
  unsigned int getNumBlocks() const { return globalBlocks + localBlocks; }

  smt::expr isLocalBlock(smt::expr &bid) const;
  smt::expr isGlobalBlock(smt::expr &bid) const;

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
};

class SingleArrayMemory: public Memory {
  smt::expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  smt::expr writableMaps; // bv(bits)::sort() -> bool::sort()
  smt::expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const smt::expr &bid) const;

public:
  SingleArrayMemory(unsigned int globalBlocks, unsigned int localBlocks);

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
