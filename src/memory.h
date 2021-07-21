#pragma once

#include "z3++.h"

#include <algorithm>

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
  const unsigned int bits;
  const unsigned int numBlocks;

  z3::expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  z3::expr writableMaps; // bv(bits)::sort() -> bool::sort()
  z3::expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  Memory(unsigned int numblocks);

  unsigned int getBIDBits() const {
    return bits;
  }
  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const {
    return getMemBlock(bid).numelem;
  }
  // Mark memblock's writable flag to `writable`
  void setWritable(const z3::expr &bid, bool writable);
  // Returns: store successful?
  z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx);
  // Returns: (loaded value, load successful?)
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const;
};
