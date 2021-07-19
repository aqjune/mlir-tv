#pragma once

#include "z3++.h"

#include <algorithm>

// A memory block containing f32 elements.
class MemBlock {
public:
  z3::expr array;    // Index::sort() -> Float::sort()
  z3::expr writable; // bool::sort()
  z3::expr numelem;  // Index::sort()

  MemBlock(z3::expr &array, z3::expr &writable, z3::expr &numelem);

  // Returns: store successful?
  z3::expr store(const z3::expr &f32val, const z3::expr &idx);
  // Returns: (loaded value, load successful?)
  std::pair<z3::expr, z3::expr> load(const z3::expr &idx) const;
};

class Memory {
  z3::expr arrayMap; // bv(2)::sort() -> (Index::sort() -> Float::sort())
  z3::expr writableMap; // bv(2)::sort() -> bool::sort()
  z3::expr numelemMap; // bv(2)::sort() -> Index::sort()

public:
  static const unsigned BID_BITS = 1;
  Memory();

  MemBlock getMemBlock(const z3::expr &bid) const;

  void updateMemBlock(const z3::expr &bid, bool writable);
};
