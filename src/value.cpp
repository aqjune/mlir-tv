#include "abstractops.h"
#include "value.h"
#include "smt.h"
#include "memory.h"

using namespace std;

static vector<z3::expr> getDims(
    const mlir::ShapedType &shapedTy, bool freshVarForUnknownSize = false) {
  vector<z3::expr> dims;
  //static int dim_var = 0;

  uint64_t rank = shapedTy.getRank();
  if (rank == 0) {
    // A single element tensor.
    return vector<z3::expr>{Index(1)};
  }

  dims.reserve(rank);
  for (auto i = 0; i < rank; ++i) {
    uint64_t sz = shapedTy.getDimSize(i);
    if (sz == (uint64_t)-1ull) {
      if (freshVarForUnknownSize) {
        dims.emplace_back(Index("dim", true));
      } else {
        // TODO: raise assert failure at some point.
        dims.push_back(Index(100));
      }
    } else
      dims.push_back(Index(sz));
  }

  return dims;
}

Index::Index(): e(ctx) {}

Index::Index(unsigned i): e(ctx.bv_val(i, BITS)) {}

Index::Index(const std::string &name, bool freshvar):
    e(ctx) {
  static int count = 0;
  string name0 = name;
  if (freshvar)
    name0 = name0 + "." + to_string(count);
  e = ctx.bv_const(name0.c_str(), BITS);
}

Index::Index(const z3::expr &e): e(e) {}

z3::sort Index::sort() {
  return ctx.bv_sort(BITS);
}

Index Index::one() { return Index(1); }
Index Index::zero() { return Index(0); }

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index &i) {
  os << or_omit((z3::expr)i);
  return os;
};

std::pair<z3::expr, vector<z3::expr>> Index::refines(const Index &other) const {
  return {(z3::expr) other == (z3::expr) *this, {}};
}

Index Index::eval(z3::model m) const {
  return Index(m.eval(e, true).simplify());
}

Float::Float(const std::string &name): e(ctx.bv_const(name.c_str(), BITS)) {}

static map<double, std::string> const_vars;

Float::Float(double f): e(ctx) {
  // We don't explicitly encode f
  auto res = const_vars.try_emplace(f,
      "#float_const" + to_string(const_vars.size()));
  e = ctx.bv_const(res.first->second.c_str(), BITS);
}

Float::Float(const llvm::APFloat &f): Float(f.convertToDouble()) {}

z3::sort Float::sort() {
  return ctx.bv_sort(BITS);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Float &f) {
  os << or_omit((z3::expr)f);
  return os;
};

std::pair<z3::expr, vector<z3::expr>> Float::refines(const Float &other) const {
  return {(z3::expr) other == (z3::expr) *this, {}};
}

Float Float::eval(z3::model m) const {
  return Float(m.eval(e, true).simplify());
}

Float Float::add(const Float &b) const {
  return Float(aop::fp_add(e, b.e));
}

Float Float::mul(const Float &b) const {
  return Float(aop::fp_mul(e, b.e));
}


Integer::Integer(const std::string &name, unsigned bw):
  e(ctx.bv_const(name.c_str(), bw)) {}

Integer::Integer(int64_t i, unsigned bw):
  e(ctx.bv_val(i, bw)) {}

z3::sort Integer::sort(unsigned sz) {
  return ctx.bv_sort(sz);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Integer &i) {
  os << or_omit((z3::expr)i);
  return os;
};

std::pair<z3::expr, vector<z3::expr>> Integer::refines(const Integer &other) const {
  return {(z3::expr) other == (z3::expr) *this, {}};
}

Integer Integer::eval(z3::model m) const {
  return Integer(m.eval(e, true).simplify());
}


Tensor::Tensor(): arr(ctx) {}

Tensor::Tensor(const z3::expr &splat_elem, const vector<z3::expr> &dimvec):
    arr(ctx), dims(dimvec) {
  arr = z3::const_array(Index::sort(), splat_elem);
}

Tensor::Tensor(const vector<z3::expr> &elems1d):
    arr(z3::const_array(Index::sort(), elems1d[0])),
    dims({ (z3::expr)Index(elems1d.size()) }) {
  for (unsigned i = 1; i < elems1d.size(); ++i)
    arr = z3::store(arr, i, elems1d[i]);
}

Tensor::Tensor(const string &name, const vector<z3::expr> &dimvec,
               const z3::sort &elemty):
  arr(ctx.constant(name.c_str(), ctx.array_sort(Index::sort(), elemty))),
  dims(dimvec) {}


z3::expr Tensor::get(const vector<z3::expr> &idxs) const {
  return z3::select(arr, to1DIdx(idxs, dims));
}

Index Tensor::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

Tensor Tensor::affine(
    const std::vector<z3::expr> &newidxvars,
    std::vector<z3::expr> srcidxs,
    const std::vector<z3::expr> &newsizes) const {
  auto idxvar = Index("idx");
  auto indices = from1DIdx(idxvar, newsizes);

  for (size_t i = 0; i < srcidxs.size(); ++i) {
    auto newv = srcidxs[i];
    for (size_t j = 0; j < newidxvars.size(); ++j) {
      newv = newv.substitute(
          toExprVector({ newidxvars[j] }), toExprVector({ indices[j] }));
    }
    srcidxs[i] = newv;
  }

  Tensor newm;
  newm.dims = newsizes;
  newm.arr = z3::lambda(
      idxvar,
      z3::ite(
        z3::ult(idxvar, ::get1DSize(newsizes)),
        get(srcidxs),
        aop::mkZeroElemFromArr(arr)
      ));
  return newm;
}

Tensor Tensor::rotateDimensions() const {
  vector<z3::expr> newdims;
  newdims.reserve(dims.size());
  newdims.push_back(dims.back());
  std::copy(dims.cbegin(), --dims.cend(), std::back_inserter(newdims));

  vector<z3::expr> vars, tgtvars;
  vars.reserve(dims.size());
  tgtvars.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = Index(string("i" + to_string(i)));
    vars.push_back(std::move(v));
  }
  std::copy(++vars.cbegin(), vars.cend(), std::back_inserter(tgtvars));
  tgtvars.push_back(vars.front());
  
  return affine(vars, tgtvars, newdims);
}

Tensor Tensor::conv(const Tensor &filter) const {
  vector<z3::expr> output_dims = {
    Index::one(), // support an input with batch size > 1
    dims[1] + 1 - filter.dims[0],
    dims[2] + 1 - filter.dims[1],
    filter.dims[3] // channel(dims[3] = filtr.dims[2]) disappears
  };
  std::vector<z3::expr> cube_size = {
    Index::one(),
    filter.dims[0], filter.dims[1], filter.dims[2]
  };

  // n, h, w, f 
  auto i = Index("i"), j = Index("j"), k = Index("k"), l = Index("l");
  auto input_subarr = to1DArrayWithOfs(
      // batch: 0, img size: (h, w), channel: 0~
      {Index::zero(), j, k, Index::zero()},
      cube_size);

  auto filter_arr = filter.rotateDimensions()
      .to1DArrayWithOfs({l, Index::zero(), Index::zero(), Index::zero()},
        cube_size);

  // TODO: switch dot <-> dot2 after determining the abstraction level
  auto res = aop::dot(input_subarr, filter_arr,
      cube_size[0] * cube_size[1] * cube_size[2] * cube_size[3]);

  return Tensor::mkLambda(move(output_dims), {i, j, k, l}, move(res));
}

Tensor Tensor::reshape(const vector<z3::expr> &newdims) const {
  // TODO: check whether size(newdims) == size(dims)
  Tensor t2;
  t2.dims = simplifyList(newdims);
  t2.arr = arr;
  return t2;
}

Tensor Tensor::matmul(const Tensor &b) const {
  assert(dims.size() == 2);
  assert(b.dims.size() == 2);

  auto bt = b.transpose();
  auto i = Index("i"), j = Index("j");
  auto a_row = to1DArrayWithOfs(
      {i, Index::zero()}, {Index::one(), dims[1]});
  auto bt_row = bt.to1DArrayWithOfs(
      {j, Index::zero()}, {Index::one(), bt.dims[1]});

  return mkLambda({dims[0], bt.dims[0]}, {i, j},
      aop::dot(a_row, bt_row, dims[1]));
}

z3::expr Tensor::dot(const Tensor &t2) const {
  return aop::dot(arr, t2.arr, get1DSize());
}

z3::expr Tensor::sum() const {
  return aop::sum(arr, get1DSize());
}

pair<z3::expr, vector<z3::expr>> Tensor::refines(const Tensor &other) const {
  assert(arr.get_sort().is_array());
  assert(other.arr.get_sort().is_array());

  // Size mismatch check.
  // If it does, don't return index var.
  size_t sz = getDims().size();
  if (other.getDims().size() != sz)
    return {ctx.bool_val(false), {}};

  z3::expr size_match = ctx.bool_val(true);
  for (size_t i = 0; i < sz; ++i)
    size_match = size_match && (z3::expr)other.getDim(i) == (z3::expr)getDim(i);
  size_match = size_match.simplify();
  if (size_match.is_false())
    return {size_match, {}};

  // Assume that src and tgt's shape equality is already checked
  z3::expr i = Index("i");
  vector<z3::expr> params = {i};
  return {size_match && z3::implies(
      z3::ult(i, ::get1DSize(dims)),
      z3::select(arr, i) == z3::select(other.arr, i)),
    params};
}

optional<pair<vector<z3::expr>, z3::sort>>
Tensor::getDimsAndElemTy(mlir::TensorType tensorTy) {
  auto ety = getElemTy(tensorTy);
  if (!ety)
    return {};
  return {{::getDims(tensorTy), *ety}};
}

optional<z3::sort> Tensor::getElemTy(mlir::TensorType tensorTy) {
  auto elemty = tensorTy.getElementType();
  z3::sort elemty2(ctx);

  if (auto ielemty = elemty.dyn_cast<mlir::IntegerType>()) {
    elemty2 = Integer::sort(ielemty.getWidth());
  } else if (auto felemty = elemty.dyn_cast<mlir::Float32Type>()) {
    elemty2 = Float::sort();
  } else if (auto felemty = elemty.dyn_cast<mlir::Float64Type>()) {
    // In the abstract world, f32 and f64 are all unknown values
    elemty2 = Float::sort();
  } else {
    return {};
  }

  return elemty2;
}


llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Tensor &t) {
  assert(t.dims.size() > 0);
  os << "(dim :" << or_omit(t.dims[0]);
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << or_omit(t.dims[i]);
  os << ") " << or_omit(t.arr);
  return os;
};

Tensor Tensor::eval(z3::model m) const {
  Tensor t2;
  t2.dims.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = m.eval(dims[i], true).simplify();
    t2.dims.push_back(std::move(v));
  }
  t2.arr = m.eval(arr, true).simplify();
  return t2;
}

Tensor Tensor::transpose() const {
  assert(dims.size() == 2);
  auto i = Index("i"), j = Index("j");
  return Tensor::mkLambda({dims[1], dims[0]}, {j, i}, get({i, j}));
}

Tensor Tensor::mkLambda(
    std::vector<z3::expr> &&newdims, std::vector<z3::expr> &&indexvars,
    z3::expr body) {
  if (indexvars.size() == 0) {
    int64_t i;
    // If indexvars is empty, let's assume that the tensor has only one
    // element.
    if (newdims.size() == 0) {
      newdims.push_back(Index(1));
    } else
      assert(newdims.size() == 1 && newdims[0].is_numeral_i64(i) && i == 1);
  } else
    assert(newdims.size() == indexvars.size());

  auto idx = Index("idx");
  auto idxexprs = from1DIdx(idx, newdims);

  if (!indexvars.empty()) {
    // If indexvars is empty, body represents the unique element.
    body = body.substitute(toExprVector(indexvars), toExprVector(idxexprs));
  }

  Tensor t2;
  t2.dims = move(newdims);
  t2.arr = z3::lambda({(z3::expr)idx}, body);
  return t2;
}

z3::expr Tensor::to1DArrayWithOfs(
      const vector<z3::expr> &offbegins,
      const vector<z3::expr> &sizes) const {
  assert(offbegins.size() == sizes.size());

  auto idxvar = Index("idx");
  auto relidxs = from1DIdx(idxvar, sizes);
  vector<z3::expr> absidxs;
  absidxs.reserve(relidxs.size());
  for (size_t i = 0; i < relidxs.size(); ++i) {
    auto absidx = relidxs[i] + offbegins[i];
    absidxs.push_back(std::move(absidx));
  }

  return z3::lambda(
      idxvar,
      z3::ite(
        z3::ult(idxvar, ::get1DSize(sizes)),
        get(absidxs),
        aop::mkZeroElemFromArr(arr)));
}

MemRef::MemRef(Memory *m): m(m), bid(ctx), offset(ctx) {}

MemRef::MemRef(Memory *m,
  const std::string &name,
  const std::vector<z3::expr> &dims,
  const z3::sort &elemty):
    m(m),
    bid(ctx.bv_const((name + "_bid").c_str(), m->getBIDBits())),
    offset(Index((name + "_offset").c_str())),
    dims(dims) {}

optional<pair<vector<z3::expr>, z3::sort>>
MemRef::getDimsAndElemTy(mlir::MemRefType memRefTy) {
  // Step1. check element type
  auto elemty = memRefTy.getElementType();
  z3::sort elemty2(ctx);

  if (auto felemty = elemty.dyn_cast<mlir::Float32Type>()) {
    elemty2 = Float::sort();
  } else {
    // Currently we only support f32 element type.
    return {};
  }

  // Step2. check affine map
  auto all_maps_are_identity = [](llvm::ArrayRef<mlir::AffineMap> maps) {
    return llvm::all_of(maps,
                        [](mlir::AffineMap map) { return map.isIdentity(); });
  };
  auto affine = memRefTy.getAffineMaps();
  if (all_maps_are_identity(affine)) {
    return {{::getDims(memRefTy), elemty2}};
  } else {
    // Currently we only support identity affine map memref.
    return {};
  }
}

pair<z3::expr, z3::expr> MemRef::load(const vector<z3::expr> &indices) const {
  z3::expr idx = to1DIdx(indices, dims);
  return m->load(bid, offset + idx);
}

z3::expr MemRef::store(const z3::expr &value, const std::vector<z3::expr> &indices) const {
  z3::expr idx = to1DIdx(indices, dims);
  return m->store(value, bid, offset + idx);
}

z3::expr MemRef::isInBounds() const {
  auto numelem = m->getNumElementsOfMemBlock(bid);
  auto memrefSize = get1DSize(dims);
  return z3::uge(numelem, memrefSize) && z3::ult(offset, numelem - memrefSize);
}

Index MemRef::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const MemRef &m) {
  assert(m.dims.size() > 0);
  os << "(bid: " << or_omit(m.bid)
    << ", offset: " << or_omit(m.offset)
    << ", dim: " << or_omit(m.dims[0]);
  for (size_t i = 1; i < m.dims.size(); ++i)
    os << ", " << or_omit(m.dims[i]);
  os << ")";
  return os;
};

std::pair<z3::expr, vector<z3::expr>> MemRef::refines(const MemRef &other) const {
  return {(z3::expr) other == (z3::expr) *this, {}};
}

MemRef MemRef::eval(z3::model m) const {
  MemRef m2(this->m);
  m2.dims.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = m.eval(dims[i], true).simplify();
    m2.dims.push_back(std::move(v));
  }
  m2.bid = m.eval(bid, true).simplify();
  m2.offset = m.eval(offset, true).simplify();
  return m2;
}
