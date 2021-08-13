#include "abstractops.h"
#include "value.h"
#include "smt.h"
#include "memory.h"

using namespace smt;
using namespace std;

static vector<expr> getDims(
    const mlir::ShapedType &shapedTy, bool freshVarForUnknownSize = false) {
  vector<expr> dims;
  //static int dim_var = 0;

  uint64_t rank = shapedTy.getRank();
  if (rank == 0) {
    // A single element tensor.
    return vector<expr>{Index(1)};
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

static expr getConstOrVal(int64_t val, const std::string &name) {
  return (val == mlir::ShapedType::kDynamicStrideOrOffset) ? Index(name, true) : Index(val);
}

static MemRef::Layout
getLayout(const mlir::MemRefType &memRefTy, const vector<expr> &dims) {
  auto affineMaps = memRefTy.getAffineMaps();

  if (affineMaps.empty()) {
    expr layout = Index::zero();
    expr stride = Index::one();
    vector<expr> indVars;

    for (int i = 0; i < dims.size(); i ++)
      indVars.push_back(Index("idx" + to_string(i)));

    for (int i = dims.size() - 1; i >= 0; i --) {
      layout = layout + stride * indVars[i];
      stride = stride * dims[i];
    }

    return MemRef::Layout(indVars, layout);
  } else {
    int64_t offset;
    llvm::SmallVector<int64_t, 4> strides;
    auto success = mlir::getStridesAndOffset(memRefTy, strides, offset);
    assert(succeeded(success) && "unexpected non-strided memref");
    expr layout = getConstOrVal(offset, "offset");
    vector<expr> indVars;
    for (int i = 0; i < strides.size(); i ++) {
      indVars.push_back(Index("idx" + to_string(i)));
      layout = layout + getConstOrVal(strides[i], "strides") * indVars[i];
    }

    return MemRef::Layout(indVars, layout);
  }
}

static string freshName(string prefix) {
  static int count = 0;
  return prefix + to_string(count ++);
}

Index::Index(): e(ctx) {}

Index::Index(unsigned i): e(ctx.bv_val(i, BITS)) {}

Index::Index(const std::string &name, bool freshvar):
    e(ctx) {
  static int count = 0;
  string name0 = name;
  if (freshvar)
    name0 = name0 + "." + to_string(count++);
  e = ctx.bv_const(name0.c_str(), BITS);
}

Index::Index(const expr &e): e(e) {}

z3::sort Index::sort() {
  return ctx.bv_sort(BITS);
}

Index Index::one() { return Index(1); }
Index Index::zero() { return Index(0); }

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index &i) {
  os << or_omit((expr)i);
  return os;
};

std::pair<expr, vector<expr>> Index::refines(const Index &other) const {
  return {(expr) other == (expr) *this, {}};
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
  os << or_omit((expr)f);
  return os;
};

std::pair<expr, vector<expr>> Float::refines(const Float &other) const {
  return {(expr) other == (expr) *this, {}};
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
  os << or_omit((expr)i);
  return os;
};

std::pair<expr, vector<expr>> Integer::refines(const Integer &other) const {
  return {(expr) other == (expr) *this, {}};
}

Integer Integer::eval(z3::model m) const {
  return Integer(m.eval(e, true).simplify());
}


Tensor::Tensor(): arr(ctx) {}

Tensor::Tensor(const expr &splat_elem, const vector<expr> &dimvec):
    arr(ctx), dims(dimvec) {
  arr = z3::const_array(Index::sort(), splat_elem);
}

Tensor::Tensor(const vector<expr> &elems1d):
    arr(z3::const_array(Index::sort(), elems1d[0])),
    dims({ (expr)Index(elems1d.size()) }) {
  for (unsigned i = 1; i < elems1d.size(); ++i)
    arr = z3::store(arr, i, elems1d[i]);
}

Tensor::Tensor(const string &name, const vector<expr> &dimvec,
               const z3::sort &elemty):
  arr(ctx.constant(name.c_str(), ctx.array_sort(Index::sort(), elemty))),
  dims(dimvec) {}

// Sparse
Tensor::Tensor(const vector<uint64_t> &indexes, const vector<expr> &elems1d,
               const vector<expr> &dimvec, const expr &zeroExpr):
    arr(z3::const_array(Index::sort(), zeroExpr)), dims(dimvec) {
  for (unsigned i = 0; i < indexes.size(); ++i)
    arr = z3::store(arr, indexes[i], elems1d[i]);
}

expr Tensor::getWellDefined() const {
  expr size = get1DSize();
  if (size.is_numeral())
    return ctx.bool_val(true);
  auto expr = z3::ule(size, MAX_TENSOR_SIZE);
  for (auto dim: dims) {
    if (dim.is_numeral()) continue;
    expr = expr && z3::ule(dim, MAX_DIM_SIZE);
  }
  return expr.simplify();
}

expr Tensor::get(const vector<expr> &idxs) const {
  return z3::select(arr, to1DIdx(idxs, dims));
}

Index Tensor::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

Tensor Tensor::affine(
    const std::vector<expr> &newidxvars,
    std::vector<expr> srcidxs,
    const std::vector<expr> &newsizes) const {
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
  vector<expr> newdims;
  newdims.reserve(dims.size());
  newdims.push_back(dims.back());
  std::copy(dims.cbegin(), --dims.cend(), std::back_inserter(newdims));

  vector<expr> vars, tgtvars;
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
  vector<expr> output_dims = {
    Index::one(), // support an input with batch size > 1
    dims[1] + 1 - filter.dims[0],
    dims[2] + 1 - filter.dims[1],
    filter.dims[3] // channel(dims[3] = filtr.dims[2]) disappears
  };
  std::vector<expr> cube_size = {
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

Tensor Tensor::reshape(const vector<expr> &newdims) const {
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

expr Tensor::dot(const Tensor &t2) const {
  return aop::dot(arr, t2.arr, get1DSize());
}

expr Tensor::sum() const {
  return aop::sum(arr, get1DSize());
}

pair<expr, vector<expr>> Tensor::refines(const Tensor &other) const {
  assert(arr.get_sort().is_array());
  assert(other.arr.get_sort().is_array());

  // Size mismatch check.
  // If it does, don't return index var.
  size_t sz = getDims().size();
  if (other.getDims().size() != sz)
    return {ctx.bool_val(false), {}};

  expr size_match = ctx.bool_val(true);
  for (size_t i = 0; i < sz; ++i)
    size_match = size_match && (expr)other.getDim(i) == (expr)getDim(i);
  size_match = size_match.simplify();
  if (size_match.is_false())
    return {size_match, {}};

  // Assume that src and tgt's shape equality is already checked
  expr i = Index("i");
  vector<expr> params = {i};
  return {size_match && z3::implies(
      z3::ult(i, ::get1DSize(dims)),
      z3::select(arr, i) == z3::select(other.arr, i)),
    params};
}

optional<pair<vector<expr>, z3::sort>>
Tensor::getDimsAndElemTy(
    mlir::TensorType tensorTy, bool freshVarForUnknownSize) {
  auto ety = getElemTy(tensorTy);
  if (!ety)
    return {};
  return {{::getDims(tensorTy, freshVarForUnknownSize), *ety}};
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
    std::vector<expr> &&newdims, std::vector<expr> &&indexvars,
    expr body) {
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
  t2.arr = z3::lambda({(expr)idx}, body);
  return t2;
}

expr Tensor::to1DArrayWithOfs(
      const vector<expr> &offbegins,
      const vector<expr> &sizes) const {
  assert(offbegins.size() == sizes.size());

  auto idxvar = Index("idx");
  auto relidxs = from1DIdx(idxvar, sizes);
  vector<expr> absidxs;
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

MemRef::MemRef(Memory *m) : m(m), bid(ctx), offset(ctx), layout(Layout({}, ctx)) {}

MemRef::MemRef(Memory *m,
  const smt::expr &bid,
  const smt::expr &offset,
  const std::vector<smt::expr> &dims,
  const Layout &layout,
  const z3::sort &elemty) : m(m), bid(bid), offset(offset), dims(dims), layout(layout) {}

MemRef::MemRef(Memory *m,
  const std::string &name,
  const std::vector<expr> &dims,
  const Layout &layout,
  const z3::sort &elemty):
    m(m),
    bid(ctx.bv_const((name + "_bid").c_str(), m->getBIDBits())),
    offset(Index((name + "_offset").c_str())),
    dims(dims),
    layout(layout) {}

MemRef::MemRef(Memory *m,
    const std::vector<expr> &dims,
    const Layout &layout,
    const z3::sort &elemty) : MemRef(m, freshName("memref"), dims, layout, elemty) {}

expr MemRef::getWellDefined() const {
  expr size = get1DSize();
  if (size.is_numeral())
    return ctx.bool_val(true);
  auto expr = z3::ule(size, MAX_MEMREF_SIZE);
  for (auto dim: dims) {
    if (dim.is_numeral()) continue;
    expr = expr && z3::ule(dim, MAX_DIM_SIZE);
  }
  return expr.simplify();
}

optional<tuple<vector<expr>, MemRef::Layout, z3::sort>>
MemRef::getDimsAndLayoutAndElemTy(
    mlir::MemRefType memRefTy,
    optional<vector<expr>> predefinedDims,
    bool freshVarForUnknownSize) {
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
  if (mlir::isStrided(memRefTy)) {
    auto dims = predefinedDims.value_or(::getDims(memRefTy, freshVarForUnknownSize));
    auto layout = ::getLayout(memRefTy, dims);
    return {{dims, layout, elemty2}};
  } else {
    // Currently we only support strided Memref.
    return {};
  }
}

pair<expr, expr> MemRef::load(const vector<expr> &indices) {
  expr idx = to1DIdxWithLayout(indices);
  auto [expr, success] = m->load(bid, offset + idx);

  // check whether indices are inbound.
  for (int i = 0; i < indices.size(); i ++)
    success = success && z3::ult(indices[i], getDim(i));

  return {expr, success.simplify()};
}

expr MemRef::store(const expr &value, const std::vector<expr> &indices) {
  expr idx = to1DIdxWithLayout(indices);
  auto success = m->store(value, bid, offset + idx);

  // check whether indices are inbound.
  for (int i = 0; i < indices.size(); i ++)
    success = success && z3::ult(indices[i], getDim(i));

  return success.simplify();
}

expr MemRef::storeArray(const expr &array, const expr &startOffset, const expr &size) {
  return m->storeArray(array, bid, offset + startOffset, size);
}

expr MemRef::isInBounds() const {
  auto numelem = m->getNumElementsOfMemBlock(bid);
  auto memrefSize = get1DSize();
  return z3::uge(numelem, memrefSize) && z3::ult(offset, numelem - memrefSize);
}

expr MemRef::isGlobalBlock() const {
  return m->isGlobalBlock(bid);
}

expr MemRef::isLocalBlock() const {
  return m->isLocalBlock(bid);
}

Index MemRef::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

void MemRef::setWritable(bool writable) {
  m->setWritable(bid, writable);
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

std::pair<expr, vector<expr>> MemRef::refines(const MemRef &other) const {
  return {(expr) other == (expr) *this, {}};
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

expr MemRef::to1DIdxWithLayout(const vector<expr> &idxs) {
  return layout.expr.substitute(toExprVector(layout.indVars), toExprVector(idxs));
}
