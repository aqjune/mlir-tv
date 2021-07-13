#pragma once

#include "z3++.h"
#include <algorithm>

// A memory block containing f32 elements.
class MemBlock {
public:
  const unsigned bid;
  z3::expr array;    // Index::sort() -> Float::sort()
  z3::expr numelem;  // Index::sort()
  z3::expr isConstant; // bool

  MemBlock(unsigned bid);

  // Returns: store successful?
  z3::expr store(const z3::expr &f32val, const z3::expr &idx);
  // Returns: (loaded value, load successful?)
  std::pair<z3::expr, z3::expr> load(const z3::expr &idx) const;
};

class Memory {
public:
  MemBlock mb0;
  Memory(): mb0(0) {}

  // Currently we support only one memblock. We relax this constraints afterward.
  MemBlock getMemBlock(unsigned bid) const { return mb0; }
};
