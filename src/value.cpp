#include "abstractops.h"
#include "value.h"
#include "smt.h"
#include "smtmatchers.h"
#include "memory.h"

using namespace smt;
using namespace std;

static string freshName(string prefix) {
  static int count = 0;
  return prefix + to_string(count ++);
}

static vector<Expr> getDims(
    const mlir::ShapedType &shapedTy, bool freshVarForUnknownSize = false) {
  vector<Expr> dims;

  uint64_t rank = shapedTy.getRank();
  if (rank == 0) {
    // A single element tensor.
    return vector<Expr>{Index(1)};
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

static Expr getConstOrVal(int64_t val, std::string &&name) {
  return (val == mlir::ShapedType::kDynamicStrideOrOffset) ?
      Index(move(name), true) : Index(val);
}

static MemRef::Layout
getLayout(const mlir::MemRefType &memRefTy, const vector<Expr> &dims) {
  auto affineMaps = memRefTy.getAffineMaps();

  if (affineMaps.empty())
    return MemRef::Layout(dims);

  int64_t offset;
  llvm::SmallVector<int64_t, 4> strides;
  auto success = mlir::getStridesAndOffset(memRefTy, strides, offset);
  assert(succeeded(success) && "unexpected non-strided memref");
  Expr layout = getConstOrVal(offset, "offset");
  vector<Expr> indVars;
  Expr inbounds = Expr::mkBool(true);
  for (int i = 0; i < strides.size(); i ++) {
    indVars.push_back(Index("idx" + to_string(i)));
    layout = layout + getConstOrVal(strides[i], "strides") * indVars[i];
    inbounds = inbounds & indVars[i].ult(dims[i]);
  }

  return MemRef::Layout(indVars, layout, inbounds);
}

Index::Index(unsigned i): e(Expr::mkBV(i, BITS)) {}

Index::Index(std::string &&name, bool freshvar):
    e(freshvar ?
      Expr::mkFreshVar(Index::sort(), move(name)) :
      Expr::mkVar(Index::sort(), move(name))) {}

Index::Index(const Expr &e): e(e) {}

Sort Index::sort() {
  return Sort::bvSort(BITS);
}

Index Index::one() { return Index(1); }
Index Index::zero() { return Index(0); }

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index &i) {
  os << or_omit((Expr)i);
  return os;
};

std::pair<Expr, vector<Expr>> Index::refines(const Index &other) const {
  return {(Expr) other == (Expr) *this, {}};
}

Index Index::eval(Model m) const {
  return Index(m.eval(e, true).simplify());
}

Float::Float(std::string &&name): e(Expr::mkVar(Float::sort(), move(name))) {}

Float::Float(double f): e(aop::fpConst(f)) {}

Float::Float(const llvm::APFloat &f): Float(f.convertToDouble()) {}

Sort Float::sort() {
  return aop::fpSort();
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Float &f) {
  Expr e = f;
  auto vec = aop::fpPossibleConsts(e);
  if (!vec.empty()) {
    os << vec[0];
    for (unsigned i = 1; i < vec.size(); ++i)
      os << " or " << vec[i];
  } else {
    os << "unknown (" << or_omit((Expr)f) << ")";
  }
  return os;
};

std::pair<Expr, vector<Expr>> Float::refines(const Float &other) const {
  return {(Expr) other == (Expr) *this, {}};
}

Float Float::eval(Model m) const {
  return Float(m.eval(e, true).simplify());
}

Float Float::add(const Float &b) const {
  return Float(aop::fpAdd(e, b.e));
}

Float Float::mul(const Float &b) const {
  return Float(aop::fpMul(e, b.e));
}


Integer::Integer(std::string &&name, unsigned bw):
  e(Expr::mkVar(Sort::bvSort(bw), move(name))) {}

Integer::Integer(int64_t i, unsigned bw):
  e(Expr::mkBV(i, bw)) {}

Integer::Integer(const llvm::APInt &api):
  Integer(api.getSExtValue(), api.getBitWidth()) {}

Sort Integer::sort(unsigned sz) {
  return Sort::bvSort(sz);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Integer &i) {
  os << or_omit((Expr)i);
  return os;
};

std::pair<Expr, vector<Expr>> Integer::refines(const Integer &other) const {
  return {(Expr) other == (Expr) *this, {}};
}

Integer Integer::eval(Model m) const {
  return Integer(m.eval(e, true).simplify());
}


Tensor::Tensor(const Expr &splat_elem, const vector<Expr> &dimvec):
    arr(Expr::mkConstArray(Index::sort(), splat_elem)), dims(dimvec) {}

Tensor::Tensor(const vector<Expr> &elems1d):
    arr(Expr::mkConstArray(Index::sort(), elems1d[0])),
    dims({ (Expr)Index(elems1d.size()) }) {
  for (unsigned i = 1; i < elems1d.size(); ++i)
    arr = arr.store(i, elems1d[i]);
}

Tensor::Tensor(string &&name, const vector<Expr> &dimvec,
               const smt::Sort &elemty):
  arr(Expr::mkVar(Sort::arraySort(Index::sort(), elemty), move(name))),
  dims(dimvec) {}

Tensor::Tensor(
    const vector<vector<uint64_t>> &indices,
    const vector<Expr> &elems,
    const vector<uint64_t> &dims, const Expr &zero):
  arr(Expr::mkConstArray(Index::sort(), zero)) {

  assert(indices.size() == elems.size());

  for (auto d: dims)
    this->dims.push_back(Index(d));

  for (unsigned i = 0; i < indices.size(); ++i) {
    assert(indices[i].size() == dims.size());

    uint64_t ofs = indices[i][0];
    for (unsigned j = 1; j < dims.size(); ++j)
      ofs = ofs * dims[j] + indices[i][j];

    arr = arr.store(ofs, elems[i]);
  }
}

Expr Tensor::getWellDefined() const {
  Expr size = get1DSize();
  if (size.isNumeral())
    return Expr::mkBool(true);

  auto e = size.ule(MAX_TENSOR_SIZE);
  for (auto dim: dims) {
    if (dim.isNumeral()) continue;
    e = e & dim.ule(MAX_DIM_SIZE);
  }
  return e.simplify();
}

Expr Tensor::get(const vector<Expr> &idxs) const {
  return arr.select(to1DIdx(idxs, dims));
}

Index Tensor::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

Tensor Tensor::affine(
    const vector<Expr> &newidxvars,
    vector<Expr> srcidxs,
    vector<Expr> &&newsizes) const {
  auto idxvar = Index("idx");
  auto indices = from1DIdx(idxvar, newsizes);

  for (size_t i = 0; i < srcidxs.size(); ++i) {
    auto newv = srcidxs[i];
    for (size_t j = 0; j < newidxvars.size(); ++j) {
      newv = newv.substitute({ newidxvars[j] }, { indices[j] });
    }
    srcidxs[i] = newv;
  }

  return {
    move(newsizes),
    Expr::mkLambda(
      idxvar,
      Expr::mkIte(
        ((Expr)idxvar).ult(::get1DSize(newsizes)),
        get(srcidxs),
        aop::mkZeroElemFromArr(arr)
      ))
  };
}

Tensor Tensor::rotateDimensions() const {
  vector<Expr> newdims;
  newdims.reserve(dims.size());
  newdims.push_back(dims.back());
  std::copy(dims.cbegin(), --dims.cend(), std::back_inserter(newdims));

  vector<Expr> vars, tgtvars;
  vars.reserve(dims.size());
  tgtvars.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = Index(string("i" + to_string(i)));
    vars.push_back(std::move(v));
  }
  std::copy(++vars.cbegin(), vars.cend(), std::back_inserter(tgtvars));
  tgtvars.push_back(vars.front());
  
  return affine(vars, tgtvars, move(newdims));
}

Tensor Tensor::conv(const Tensor &filter) const {
  vector<Expr> output_dims = {
    Index::one(), // support an input with batch size > 1
    dims[1] + 1 - filter.dims[0],
    dims[2] + 1 - filter.dims[1],
    filter.dims[3] // channel(dims[3] = filtr.dims[2]) disappears
  };
  std::vector<Expr> cube_size = {
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

Tensor Tensor::reshape(const vector<Expr> &newdims) const {
  // TODO: check whether size(newdims) == size(dims)
  return { simplifyList(newdims), Expr(arr) };
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

Expr Tensor::dot(const Tensor &t2) const {
  return aop::dot(arr, t2.arr, get1DSize());
}

Expr Tensor::sum() const {
  return aop::sum(arr, get1DSize());
}

pair<Expr, vector<Expr>> Tensor::refines(const Tensor &other) const {
  assert(arr.sort().isArray());
  assert(other.arr.sort().isArray());

  // Size mismatch check.
  // If it does, don't return index var.
  size_t sz = getDims().size();
  if (other.getDims().size() != sz)
    return {Expr::mkBool(false), {}};

  Expr size_match = Expr::mkBool(true);
  for (size_t i = 0; i < sz; ++i)
    size_match = size_match & (Expr)other.getDim(i) == (Expr)getDim(i);
  size_match = size_match.simplify();
  if (size_match.isFalse())
    return {size_match, {}};

  // Assume that src and tgt's shape equality is already checked
  Expr i = Index("i");
  vector<Expr> params = {i};
  return {size_match &
      i.ult(::get1DSize(dims)).implies(arr.select(i) == other.arr.select(i)),
    params};
}

optional<pair<vector<Expr>, smt::Sort>>
Tensor::getDimsAndElemTy(
    mlir::TensorType tensorTy, bool freshVarForUnknownSize) {
  auto ety = getElemTy(tensorTy);
  if (!ety)
    return {};
  return {{::getDims(tensorTy, freshVarForUnknownSize), *ety}};
}

optional<smt::Sort> Tensor::getElemTy(mlir::TensorType tensorTy) {
  auto elemty = tensorTy.getElementType();

  if (auto ielemty = elemty.dyn_cast<mlir::IntegerType>()) {
    return Integer::sort(ielemty.getWidth());
  } else if (auto felemty = elemty.dyn_cast<mlir::Float32Type>()) {
    return Float::sort();
  } else if (auto felemty = elemty.dyn_cast<mlir::Float64Type>()) {
    // In the abstract world, f32 and f64 are all unknown values
    return Float::sort();
  } else if (elemty.isa<mlir::IndexType>()) {
    return Index::sort();
  }

  return {};
}


llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Tensor &t) {
  assert(t.dims.size() > 0);
  os << "(dim: " << or_omit(t.dims[0]);
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << or_omit(t.dims[i]);
  os << ") ";

  using namespace smt::matchers;
  Expr arr = t.arr;
  bool hasStore = false;

  while (true) {
    optional<Expr> arr2, idx, val;

    if (Store(Any(arr2), Any(idx), Any(val)).match(arr)) {
      auto idxnd = from1DIdx(*idx, t.dims);
      vector<int64_t> idxconsts;

      bool constIdxs = all_of(idxnd.begin(), idxnd.end(), [&](const Expr &e) {
        int64_t i;
        if (e.simplify().isInt(i)) {
          idxconsts.push_back(i);
          return true;
        }
        return false;
      });
      if (constIdxs) {
        os << "(" << idxconsts[0];
        for (size_t i = 1; i < idxconsts.size(); ++i)
          os << ", " << idxconsts[i];
        os << ")";
      } else
        os << or_omit(*idx);
      os << " -> " << or_omit(*val) << ", ";

      arr = move(*arr2);
      hasStore = true;

    } else if (ConstSplatArray(Any(val)).match(arr)) {
      if (hasStore)
        os << "else " << or_omit(*val);
      else
        os << "a splat tensor of " << or_omit(*val);
      break;

    } else {
      os << (hasStore ? "else " : "") << or_omit(arr);
      break;
    }
  }
  return os;
};

Tensor Tensor::eval(Model m) const {
  vector<Expr> dims_ev;
  dims_ev.reserve(dims.size());
  for (auto &d: dims)
    dims_ev.push_back(m.eval(d, true).simplify());

  return { move(dims_ev), m.eval(arr, true).simplify() };
}

Tensor Tensor::transpose() const {
  assert(dims.size() == 2);
  auto i = Index("i"), j = Index("j");
  return Tensor::mkLambda({dims[1], dims[0]}, {j, i}, get({i, j}));
}

Tensor Tensor::mkLambda(
    std::vector<Expr> &&newdims, std::vector<Expr> &&indexvars,
    Expr body) {
  if (indexvars.size() == 0) {
    int64_t i;
    // If indexvars is empty, let's assume that the tensor has only one
    // element.
    if (newdims.size() == 0) {
      newdims.push_back(Index(1));
    } else
      assert(newdims.size() == 1 && newdims[0].isInt(i) && i == 1);
  } else
    assert(newdims.size() == indexvars.size());

  auto idx = Index("idx");
  auto idxExprs = from1DIdx(idx, newdims);

  if (!indexvars.empty()) {
    // If indexvars is empty, body represents the unique element.
    body = body.substitute(indexvars, idxExprs);
  }

  return { move(newdims), Expr::mkLambda(idx, body) };
}

Expr Tensor::to1DArrayWithOfs(
      const vector<Expr> &offbegins,
      const vector<Expr> &sizes) const {
  assert(offbegins.size() == sizes.size());

  auto idxvar = Index("idx");
  auto relidxs = from1DIdx(idxvar, sizes);
  vector<Expr> absidxs;
  absidxs.reserve(relidxs.size());
  for (size_t i = 0; i < relidxs.size(); ++i) {
    auto absidx = relidxs[i] + offbegins[i];
    absidxs.push_back(std::move(absidx));
  }

  return Expr::mkLambda(
      idxvar,
      Expr::mkIte(
        ((Expr)idxvar).ult(::get1DSize(sizes)),
        get(absidxs),
        aop::mkZeroElemFromArr(arr)));
}

MemRef::Layout::Layout(const vector<Expr> &dims):
    inbounds(Expr::mkBool(true)),
    mapping(Expr::mkBool(true)), // Filled with a new lambda expr later
    precondition(Expr::mkBool(true)) {
  vector<Expr> indVars, inverseMappings;

  for (int i = 0; i < dims.size(); i ++) {
    indVars.push_back(Index("idx" + to_string(i)));
    inbounds = inbounds & indVars[i].ult(dims[i]);
  }

  Expr idx = Index("1DIdx");
  vector<Expr> inverseIdxs = from1DIdx(idx, dims);
  for (auto inverse : inverseIdxs)
    inverseMappings.push_back(Expr::mkLambda(idx, inverse));

  this->indVars = indVars;
  this->inbounds = Expr::mkLambda(indVars, inbounds);
  this->mapping = Expr::mkLambda(indVars, to1DIdx(indVars, dims));
  this->inverseMappings = inverseMappings;
}

MemRef::Layout::Layout(const std::vector<smt::Expr> &indVars,
    const smt::Expr &layout,
    const smt::Expr &inbounds,
    bool useUF):
    indVars(indVars),
    inbounds(Expr::mkBool(true)),  // Filled with a new lambda expr later
    mapping(Expr::mkBool(true)), // Filled with a new lambda expr later
    precondition(Expr::mkBool(true)) // Filled with a new lambda expr later
    {
  if (useUF) {
    vector<smt::Sort> domains(indVars.size(), Index::sort());
    FnDecl layoutFn(domains, Index::sort(), freshName("layoutFn"));
    auto layoutFnExpr = layoutFn.apply(indVars);
    Expr condition = (layoutFnExpr == layout);
    vector<Expr> inverseMappings;

    for (unsigned i = 0; i < indVars.size(); i ++) {
      auto inverseName = freshName("inverse" + to_string(i));
      FnDecl inverseFn(Index::sort(), Index::sort(), move(inverseName));
      auto inverse = Expr::mkLambda(indVars[i], inverseFn(indVars[i]));
      inverseMappings.push_back(inverse);

      condition = condition & inverse.select(layoutFnExpr) == indVars[i];
    }
    this->inbounds = Expr::mkLambda(indVars, inbounds);
    this->mapping = Expr::mkLambda(indVars, layoutFnExpr);
    this->inverseMappings = inverseMappings;
    this->precondition = Expr::mkForall(
        indVars, inbounds.implies(condition));
  } else {
    Expr condition = Expr::mkBool(true);
    vector<Expr> inverseMappings;
    for (unsigned i = 0; i < indVars.size(); i ++) {
      auto inverseName = freshName("inverse" + to_string(i));
      FnDecl inverseFn(Index::sort(), Index::sort(), move(inverseName));
      auto inverse = Expr::mkLambda(indVars[i], inverseFn(indVars[i]));
      inverseMappings.push_back(inverse);

      condition = condition & inverse.select(layout) == indVars[i];
    }
    this->inbounds = Expr::mkLambda(indVars, inbounds);
    this->mapping = Expr::mkLambda(indVars, layout);
    this->inverseMappings = inverseMappings;
    this->precondition = Expr::mkForall(indVars, inbounds.implies(condition));
  }
}

MemRef::MemRef(Memory *m,
  const smt::Expr &bid,
  const smt::Expr &offset,
  const std::vector<smt::Expr> &dims,
  const Layout &layout,
  const smt::Sort &elemty) : m(m), bid(bid), offset(offset), dims(dims), layout(layout) {}

MemRef::MemRef(Memory *m,
  const std::string &name,
  const std::vector<Expr> &dims,
  const Layout &layout,
  const smt::Sort &elemty):
    m(m),
    bid(Expr::mkVar(Sort::bvSort(m->getBIDBits()), (name + "_bid").c_str())),
    offset(Index((name + "_offset").c_str())),
    dims(dims),
    layout(layout) {}

MemRef::MemRef(Memory *m,
    const std::vector<Expr> &dims,
    const Layout &layout,
    const smt::Sort &elemty) : MemRef(m, freshName("memref"), dims, layout, elemty) {}

Expr MemRef::getPrecondition() const {
  return layout.precondition;
}

Expr MemRef::getWellDefined() const {
  Expr size = get1DSize();
  if (size.isNumeral())
    return Expr::mkBool(true);

  auto e = size.ule(MAX_MEMREF_SIZE);
  for (auto dim: dims) {
    if (dim.isNumeral()) continue;
    e = e & dim.ule(MAX_DIM_SIZE);
  }
  return e.simplify();
}

optional<tuple<vector<Expr>, MemRef::Layout, smt::Sort>>
MemRef::getDimsAndLayoutAndElemTy(
    mlir::MemRefType memRefTy,
    optional<vector<Expr>> predefinedDims,
    bool freshVarForUnknownSize) {
  // Step1. check element type
  auto elemty = memRefTy.getElementType();
  if (!elemty.isa<mlir::Float32Type>())
    // Currently we only support f32 element type.
    return {};

  auto elemty2 = Float::sort();

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

pair<Expr, Expr> MemRef::load(const vector<Expr> &indices) {
  auto [idx, inbounds] = to1DIdxWithLayout(indices);
  auto [loaded, success] = m->load(bid, (Expr)offset + idx);

  return {loaded, (success & inbounds).simplify()};
}

Expr MemRef::store(const Expr &value, const std::vector<Expr> &indices) {
  auto [idx, inbounds] = to1DIdxWithLayout(indices);
  auto success = m->store(value, bid, (Expr)offset + idx);

  return (success & inbounds).simplify();
}

Expr MemRef::storeArray(
    const Expr &array, const Expr &startOffset, const Expr &size) {
  return m->storeArray(array, bid, (Expr)offset + startOffset, size);
}

Expr MemRef::isInBounds() const {
  auto numelem = m->getNumElementsOfMemBlock(bid);
  auto memrefSize = get1DSize();
  return numelem.uge(memrefSize) & ((Expr)offset).ult(numelem - memrefSize);
}

Expr MemRef::isGlobalBlock() const {
  return m->isGlobalBlock(bid);
}

Expr MemRef::isLocalBlock() const {
  return m->isLocalBlock(bid);
}

Index MemRef::getDim(uint64_t idx) const {
  return Index(dims[idx]);
}

void MemRef::setWritable(bool writable) {
  m->setWritable(bid, writable);
}

MemRef MemRef::subview(const vector<Expr> &offsets,
    const vector<Expr> &sizes,
    const vector<Expr> &strides,
    int rankDiff) {
  if (rankDiff > 0) {
    vector<Expr> indVars, reducedSizes;
    for (unsigned i = 0; i < sizes.size(); i++) {
      uint64_t size_cst;
      if (rankDiff > 0 && sizes[i].isUInt(size_cst) && size_cst == 1) {
        //statically known to be 1
        indVars.push_back(Index::zero());
        rankDiff --;
      } else {
        indVars.push_back(layout.indVars[i]);
        reducedSizes.push_back(sizes[i]);
      }
    }
    auto subviewLayout = createSubViewLayout(indVars, offsets, strides);
    return MemRef(m, bid, offset, reducedSizes, subviewLayout, Float::sort());
  } else {
    auto subviewLayout = createSubViewLayout(layout.indVars, offsets, strides);
    return MemRef(m, bid, offset, sizes, subviewLayout, Float::sort());
  }
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

std::pair<Expr, vector<Expr>> MemRef::refines(const MemRef &other) const {
  return {(Expr) other == (Expr) *this, {}};
}

MemRef MemRef::eval(Model mdl) const {
  MemRef m2 = *this;
  for (size_t i = 0; i < m2.dims.size(); ++i)
    m2.dims[i] = mdl.eval(m2.dims[i], true).simplify();

  m2.bid = mdl.eval(m2.bid, true).simplify();
  m2.offset = mdl.eval(m2.offset, true).simplify();

  return m2;
}

pair<Expr, Expr> MemRef::to1DIdxWithLayout(const vector<Expr> &idxs) {
  auto Expr = layout.mapping.select(idxs);
  auto inbounds = layout.inbounds.select(idxs);
  return {Expr, inbounds};
}

MemRef::Layout MemRef::createSubViewLayout(
    const vector<Expr> &indVars,
    const vector<Expr> &offsets,
    const vector<Expr> &strides) {
  // Before : <(d0, d1) -> (d0 * s0 + d1)>,
  // After: <(d0, d1) -> ((indVars[0] + offsets[0]) * strides[0] * s0 + (indVars[1] + offsets[1]) * strides[1])>
  // indVars[i] can be Index::zero() if reducing the dimension.
  assert(layout.indVars.size() == indVars.size());
  assert(layout.indVars.size() == offsets.size());
  assert(layout.indVars.size() == strides.size());

  vector<Expr> idxs, transformedIndVars;
  for (unsigned i = 0; i < layout.indVars.size(); i ++) {
    idxs.push_back((indVars[i] + offsets[i]) * strides[i]);
    if (!indVars[i].isNumeral()) transformedIndVars.push_back(indVars[i]);
  }
  auto transformedLayout = layout.mapping.select(idxs);
  auto transformedInbounds = layout.inbounds.select(idxs);
  return Layout(transformedIndVars, transformedLayout, transformedInbounds);
}
