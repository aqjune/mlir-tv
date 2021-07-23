#pragma once

#include "z3++.h"

#include <algorithm>
#include <vector>

enum MemEncoding {
  SINGLE_ARRAY, MULTIPLE_ARRAY
};

// A memory block containing f32 elements.
class MemBlock {
public:
  z3::expr array;    // Index::sort() -> Float::sort()
  z3::expr writable; // bool::sort()
  z3::expr numelem;  // Index::sort()

  MemBlock(z3::expr &array, z3::expr &writable, z3::expr &numelem):
    array(array), writable(writable), numelem(numelem) {}
};

class Memory {
protected:
  const unsigned int bidBits;
  const unsigned int numBlocks;

public:
  static Memory * create(unsigned int numBlocks, MemEncoding encoding);
  Memory(unsigned int bidBits, unsigned int numBlocks):
      bidBits(bidBits), numBlocks(numBlocks) {}
  virtual ~Memory() {}

  // Encode the refinement relation between src (other) and tgt (this) memory
  virtual std::pair<z3::expr, std::vector<z3::expr>>
    refines(const Memory &other) const = 0;

  unsigned int getBIDBits() const { return bidBits; }

  virtual z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const = 0;
  // Mark memblock's writable flag to `writable`
  virtual void setWritable(const z3::expr &bid, bool writable) = 0;
  // get memblocks' writable flag
  virtual z3::expr getWritable(const z3::expr &bid) const = 0;
  // Returns: store successful?
  virtual z3::expr store(
      const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx) = 0;
  // Returns: (loaded value, load successful?)
  virtual std::pair<z3::expr, z3::expr> load(
      const z3::expr &bid, const z3::expr &idx) const = 0;
};

class SingleArrayMemory: public Memory {
  z3::expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  z3::expr writableMaps; // bv(bits)::sort() -> bool::sort()
  z3::expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  SingleArrayMemory(unsigned int numBlocks);

  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const override {
    return getMemBlock(bid).numelem;
  }

  void setWritable(const z3::expr &bid, bool writable) override;
  z3::expr getWritable(const z3::expr &bid) const override;
  z3::expr store(
      const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx)
      override;
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx)
      const override;

  std::pair<z3::expr, std::vector<z3::expr>> refines(const Memory &other) const
      override;
};

class MultipleArrayMemory: public Memory {
  std::vector<z3::expr> arrays;  // vector<(Index::sort() -> Float::sort())>
  std::vector<z3::expr> writables; // vector<Bool::sort()>
  std::vector<z3::expr> numelems;  // vector<Index::sort>

public:
  MultipleArrayMemory(unsigned int numBlocks);

  z3::expr getNumElementsOfMemBlock(unsigned ubid) const
  { assert(ubid < numBlocks); return numelems[ubid]; }
  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const override;

  void setWritable(const z3::expr &bid, bool writable) override;
  z3::expr getWritable(const z3::expr &bid) const override;
  z3::expr getWritable(unsigned ubid) const
  { assert(ubid < numBlocks); return writables[ubid]; }

  z3::expr store(
      const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx)
      override;
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx)
      const override;
  std::pair<z3::expr, z3::expr> load(unsigned ubid, const z3::expr &idx)
      const;

  std::pair<z3::expr, std::vector<z3::expr>> refines(const Memory &other) const
      override;

private:
  z3::expr itebid(
      const z3::expr &bid, std::function<z3::expr(unsigned)> fn) const;
  void update(
      const z3::expr &bid,
      std::function<z3::expr*(unsigned)> exprToUpdate, // bid -> ptr to expr
      std::function<z3::expr(unsigned)> updatedValue) const; // bid -> updated e
};
