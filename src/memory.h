#pragma once

#include "z3++.h"

#include <algorithm>
#include <vector>

enum MemType {
  SINGLE, MULTIPLE
};

class Memory {
public:
  static Memory * create(unsigned int NUM_BLOCKS, MemType type);

  virtual unsigned int getBIDBits() const = 0;

  virtual z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const = 0;

  virtual void updateMemBlock(const z3::expr &bid, bool writable) = 0;
  // Returns: store successful?
  virtual z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx) = 0;
  // Returns: (loaded value, load successful?)
  virtual std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const = 0;
};

class SingleArrayMemory: public Memory {
  unsigned int BID_BITS;
  unsigned int NUM_BLOCKS;
  z3::expr arrayMaps; // bv(2)::sort() -> (Index::sort() -> Float::sort())
  z3::expr writableMaps; // bv(2)::sort() -> bool::sort()
  z3::expr numelemMaps; // bv(2)::sort() -> Index::sort()

  // A memory block containing f32 elements.
  class MemBlock {
  public:
    z3::expr array;    // Index::sort() -> Float::sort()
    z3::expr writable; // bool::sort()
    z3::expr numelem;  // Index::sort()

    MemBlock(z3::expr &array, z3::expr &writable, z3::expr &numelem):
      array(array), writable(writable), numelem(numelem) {}
  };

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  SingleArrayMemory(unsigned int NUM_BLOCKS);

  unsigned int getBIDBits() const;

  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const;

  void updateMemBlock(const z3::expr &bid, bool writable);

  // Returns: store successful?
  z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx);
  // Returns: (loaded value, load successful?)
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const;
};

class MultipleArrayMemory: public Memory {
  unsigned int BID_BITS;
  unsigned int NUM_BLOCKS;
  std::vector<z3::expr> arrayMaps; // bv(2) -> (Index::sort() -> Float::sort()) 
  z3::expr writableMaps; // bv(2) -> Bool::sort()
  z3::expr numelemMaps; // bv(2) -> Index::sort

  // A memory block containing f32 elements.
  class MemBlock {
  public:
    z3::expr array;    // Index::sort() -> Float::sort()
    z3::expr writable; // bool::sort()
    z3::expr numelem;  // Index::sort()

    MemBlock(z3::expr &array, z3::expr &writable, z3::expr &numelem):
      array(array), writable(writable), numelem(numelem) {}
  };

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  MultipleArrayMemory(unsigned int NUM_BLOCKS);

  unsigned int getBIDBits() const;

  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const;

  void updateMemBlock(const z3::expr &bid, bool writable);

  // Returns: store successful?
  z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx);
  // Returns: (loaded value, load successful?)
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const;
};
