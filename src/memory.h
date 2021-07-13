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
  MemBlock mb0, mb1;

public:
  Memory(): mb0(0), mb1(1) {}

  MemBlock getMemBlock(const z3::expr &bid) const;
};
