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
  const unsigned int bits;
  const unsigned int numBlocks;

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  static Memory * create(unsigned int numBlocks, MemEncoding encoding);
  Memory(unsigned int bits, unsigned int numBlocks): bits(bits), numBlocks(numBlocks) {}

  unsigned int getBIDBits() const { return bits; }

  virtual z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const = 0;
  // Mark memblock's writable flag to `writable`
  virtual void setWritable(const z3::expr &bid, bool writable) = 0;
  // Returns: store successful?
  virtual z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx) = 0;
  // Returns: (loaded value, load successful?)
  virtual std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const = 0;

  // Define refinement of memory
  virtual std::pair<z3::expr, std::vector<z3::expr>> refines(const Memory &other) const = 0;
};

class SingleArrayMemory: public Memory {
  z3::expr arrayMaps; // bv(bits)::sort() -> (Index::sort() -> Float::sort())
  z3::expr writableMaps; // bv(bits)::sort() -> bool::sort()
  z3::expr numelemMaps; // bv(bits)::sort() -> Index::sort()

private:
  MemBlock getMemBlock(const z3::expr &bid) const;

public:
  SingleArrayMemory(unsigned int numBlocks);

  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const {
    return getMemBlock(bid).numelem;
  }

  void setWritable(const z3::expr &bid, bool writable);
  z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx);
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const;

  std::pair<z3::expr, std::vector<z3::expr>> refines(const Memory &other) const;
};

class MultipleArrayMemory: public Memory {
  std::vector<z3::expr> arrayMaps; //  vector<(Index::sort() -> Float::sort())>
  z3::expr writableMaps; // bv(bits)::sort() -> Bool::sort()
  z3::expr numelemMaps; // bv(bits)::sort() -> Index::sort

public:
  MultipleArrayMemory(unsigned int numBlocks);

  z3::expr getNumElementsOfMemBlock(const z3::expr &bid) const {
    return z3::select(numelemMaps, bid);
  }

  void setWritable(const z3::expr &bid, bool writable);
  z3::expr store(const z3::expr &f32val, const z3::expr &bid, const z3::expr &idx);
  std::pair<z3::expr, z3::expr> load(const z3::expr &bid, const z3::expr &idx) const;

  std::pair<z3::expr, std::vector<z3::expr>> refines(const Memory &other) const;
};
