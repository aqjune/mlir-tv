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

vector<Expr> ShapedValue::getDims(
    const mlir::ShapedType &shapedTy, bool freshVarForUnknownSize,
    optional<vector<Expr>> &&valsForUnknownSz) {
  vector<Expr> dims;

  uint64_t rank = shapedTy.getRank();
  if (rank == 0) {
    // A single element tensor.
    return vector<Expr>{Index(1)};
  }

  dims.reserve(rank);
  unsigned unknownVarIdx = 0;
  for (auto i = 0; i < rank; ++i) {
    uint64_t sz = shapedTy.getDimSize(i);
    if (sz == (uint64_t)-1ull) {
      if (freshVarForUnknownSize) {
        dims.emplace_back(Index::var("dim", VarType::FRESH));
      } else if (valsForUnknownSz) {
        dims.emplace_back(move((*valsForUnknownSz)[unknownVarIdx++]));
      } else {
        llvm_unreachable("Don't know what to do with a dimension of "
                         "an unknown size");
      }
    } else
      dims.push_back(Index(sz));
  }

  return dims;
}

static Expr getConstOrFreshVar(int64_t val, std::string &&name) {
  return (val == mlir::ShapedType::kDynamicStrideOrOffset) ?
      Index::var(move(name), VarType::FRESH) : Index(val);
}

Index::Index(unsigned i): e(Expr::mkBV(i, BITS)) {}

Sort Index::sort() {
  return Sort::bvSort(BITS);
}

Index Index::one() { return Index(1); }
Index Index::zero() { return Index(0); }
Index Index::var(std::string &&name, VarType varty) {
  switch(varty) {
  case VarType::BOUND:
  case VarType::UNBOUND:
    return {Expr::mkVar(Index::sort(), move(name), varty == VarType::BOUND)};
  case VarType::FRESH:
    return {Expr::mkFreshVar(Index::sort(), move(name))};
  }
  assert("Unknown case");
}


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

Float::Float(double f): e(aop::fpConst(f)) {}

Float::Float(const llvm::APFloat &f): Float(f.convertToDouble()) {}

Sort Float::sort() {
  return aop::fpSort();
}

Float Float::var(std::string &&name, VarType varty) {
  switch(varty) {
  case VarType::BOUND:
  case VarType::UNBOUND:
    return {Expr::mkVar(Float::sort(), move(name), varty == VarType::BOUND)};
  case VarType::FRESH:
    return {Expr::mkFreshVar(Float::sort(), move(name))};
  }
  assert("Unknown case");
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


Integer::Integer(int64_t i, unsigned bw):
  e(Expr::mkBV(i, bw)) {}

Integer::Integer(const llvm::APInt &api):
  Integer(api.getSExtValue(), api.getBitWidth()) {}

Sort Integer::sort(unsigned sz) {
  return Sort::bvSort(sz);
}

Integer Integer::var(std::string &&name, unsigned bw, VarType varty) {
  switch(varty) {
  case VarType::BOUND:
  case VarType::UNBOUND:
    return {Expr::mkVar(Sort::bvSort(bw), move(name), varty == VarType::BOUND)};
  case VarType::FRESH:
    return {Expr::mkFreshVar(Sort::bvSort(bw), move(name))};
  }
  assert("Unknown case");
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

std::pair<std::vector<smt::Expr>, smt::Expr> ShapedValue::conv(const ShapedValue &filter,
    const vector<Expr> &strides,
    const vector<Expr> &dilations) const {
  // input: Batch_size x Dim_0 x Dim_1 .. x Dim_{n-1} x Input_channel
  // filter: Dim_0 x Dim_1 .. x Dim_{n-1} x Input_channel x Output_channel
  // output: Batch_size x Dim_0 x Dim_1 .. x Dim_{n-1} x Output_channel
  assert(getDims().size() == filter.getDims().size());
  assert(getDims().size() > 2);
  auto dim = getDims().size() - 2;

  // outputIdxs: Batch, Dim_0, Dim_1, ... Dim_{n-1}, Output_channel
  vector<Expr> outputIdxs;
  for (unsigned i = 0; i < getDims().size(); i ++)
    outputIdxs.push_back(Index::var("i" + to_string(i), VarType::BOUND));

  // cubeSize = Dim_0 x Dim_1 .. x Dim_{n-1} x Input_channel
  vector<Expr> cubeSize;
  for (unsigned i = 0; i < dim; i ++)
    cubeSize.push_back(filter.getDim(i));
  cubeSize.push_back(filter.getDim(dim));
  auto cubeIdx = Index::var("cubeIdx", VarType::BOUND);
  auto cubeIdxs = from1DIdx(cubeIdx, cubeSize);

  vector<Expr> filterIdxs;
  // filterIdxs: Dim_0, Dim_1, ... Dim_{n-1}, Input_channel, Output_channel
  for (auto idx: cubeIdxs) filterIdxs.push_back(idx);
  filterIdxs.push_back(outputIdxs.back());

  vector<Expr> inputIdxs;
  // inputIdxs: Batch, Dim_0, Dim_1, ... Dim_{n-1}, Input_channel
  inputIdxs.push_back(outputIdxs.front());
  for (unsigned i = 0; i < dim; i ++)
    inputIdxs.push_back(outputIdxs[i + 1] * strides[i] + cubeIdxs[i] * dilations[i]);
  inputIdxs.push_back(cubeIdxs.back()); // Input_channel

  Expr inputExpr = Expr::mkLambda(cubeIdx, get(inputIdxs).first);
  Expr filterExpr = Expr::mkLambda(cubeIdx, filter.get(filterIdxs).first);
  Expr outputExpr = aop::dot(inputExpr, filterExpr, ::get1DSize(cubeSize));

  return {move(outputIdxs), move(outputExpr)};
}

Tensor::Tensor(Expr &&splat_elem, vector<Expr> &&dimvec):
    arr(Expr::mkSplatArray(Index::sort(), move(splat_elem))),
    dims(move(dimvec)) {}

Tensor::Tensor(const vector<Expr> &elems1d):
    arr(Expr::mkFreshVar(Sort::arraySort(Index::sort(), elems1d[0].sort()),
        "tensor_val")),
    dims({ (Expr)Index(elems1d.size()) }) {
  for (unsigned i = 0; i < elems1d.size(); ++i)
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
  arr(Expr::mkSplatArray(Index::sort(), zero)) {

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

pair<Expr, Expr> Tensor::get(const vector<Expr> &indices) const {
  auto elem = arr.select(to1DIdx(indices, dims));
  auto inbounds = Expr::mkBool(true);
  for (unsigned i = 0; i < indices.size(); ++i)
    inbounds = inbounds & indices[i].ult(dims[i]);

  return {elem, inbounds.simplify()};
}

pair<Tensor, Expr> Tensor::insert(const smt::Expr &value,
    const std::vector<smt::Expr> &indices) const {
  auto inbounds = Expr::mkBool(true);
  for (unsigned i = 0; i < indices.size(); ++i)
    inbounds = inbounds & indices[i].ult(dims[i]);

  auto idxvar = Index::var("idx", VarType::BOUND);
  auto cond = (Expr)idxvar == to1DIdx(indices, dims);
  auto originValue = get(from1DIdx(idxvar, dims)).first;

  auto newdims = dims;
  auto lambda = Expr::mkLambda(idxvar, Expr::mkIte(cond, value, originValue));
  return {{move(newdims), move(lambda)}, inbounds};
}

Tensor Tensor::affine(
    const vector<Expr> &newidxvars,
    vector<Expr> srcidxs,
    vector<Expr> &&newsizes) const {
  auto idxvar = Index::var("idx", VarType::BOUND);
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
        get(srcidxs).first,
        aop::mkZeroElemFromArr(arr)
      ))
  };
}

Tensor Tensor::conv(const Tensor &filter,
    const vector<Expr> strides,
    const vector<Expr> dilations) const {
  // output[b, x[0], ..., x[N-1], k] =
  //     sum_{z[0], ..., z[N-1], q}
  //         filter[z[0], ..., z[N-1], q, k] *
  //         input[b,
  //                      x[0]*strides[0] + dilation_rate[0]*z[0],
  //                      ...,
  //                      x[N-1]*strides[N-1] + dilation_rate[N-1]*z[N-1],
  //                      q]
  // So we can calculate output dims bounds as follow. (Assuming zero based index)
  // x[0]*strides[0] + dilation_rate[0]*z[0] < Original_Dim
  // x[0]*strides[0] + dilation_rate[0] * Filter_Dim - 1 < Original_Dim
  // x[0] < (Original_Dim + 1 - diltaion_rate[0] * Filter_Dim) / strides[0]
  // x[0] < ceil((Original_Dim + 1 - diltaion_rate[0] * Filter_Dim) / strides[0])
  // x[0] < (Original_Dim + 1 - diltaion_rate[0] * Filter_Dim + (strides[0] - 1)).udiv(strides[0])
  // x[0] < (Original_Dim - diltaion_rate[0] * Filter_Dim + strides[0]).udiv(strides[0])
  // Output Dim = (Original_Dim - diltaion_rate[0] * Filter_Dim + strides[0]).udiv(strides[0])
  vector<Expr> outputDims;
  outputDims.push_back(getDim(0)); // Input Batch Size
  for (unsigned i = 0; i < getDims().size() - 2; i ++) {
    Expr originalSize = getDim(i + 1);
    Expr filterSize = dilations[i] * filter.getDim(i);
    Expr expr = (originalSize - filterSize + strides[i]).udiv(strides[i]);
    outputDims.push_back(expr);
  }
  outputDims.push_back(filter.getDims().back()); // Output Channel

  auto [indices, res] = ShapedValue::conv(filter, strides, dilations);
  return Tensor::mkLambda(move(outputDims), move(indices), move(res));
}

Tensor Tensor::reshape(const vector<Expr> &newdims) const {
  // TODO: check whether size(newdims) == size(dims)
  return { simplifyList(newdims), Expr(arr) };
}

Tensor Tensor::matmul(const Tensor &b) const {
  assert(dims.size() == 2);
  assert(b.dims.size() == 2);

  auto bt = b.transpose();
  auto i = Index::var("i", VarType::BOUND);
  auto j = Index::var("j", VarType::BOUND);
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
  Expr i = Index::var("i", VarType::UNBOUND);
  vector<Expr> params = {i};
  return {size_match &
      i.ult(::get1DSize(dims)).implies(arr.select(i) == other.arr.select(i)),
    params};
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
  vector<Expr> dims_ev = smt::simplifyList(m.eval(dims));
  return { move(dims_ev), m.eval(arr, true).simplify() };
}

Tensor Tensor::transpose() const {
  assert(dims.size() == 2);
  auto i = Index::var("i", VarType::BOUND);
  auto j = Index::var("j", VarType::BOUND);
  return Tensor::mkLambda({dims[1], dims[0]}, {j, i}, get({i, j}).first);
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

  auto idx = Index::var("idx", VarType::BOUND);
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

  auto idxvar = Index::var("idx", VarType::BOUND);
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
        get(absidxs).first,
        aop::mkZeroElemFromArr(arr)));
}

MemRef::Layout::Layout(const vector<Expr> &dims):
    inbounds(Expr::mkBool(true)),
    mapping(Expr::mkBool(true)), // Filled with a new lambda expr later
    precondition(Expr::mkBool(true)) {
  vector<Expr> indVars, inverseMappings;

  for (int i = 0; i < dims.size(); i ++) {
    indVars.push_back(Index::var("idx" + to_string(i), VarType::BOUND));
    inbounds = inbounds & indVars[i].ult(dims[i]);
  }

  Expr idx = Index::var("1DIdx", VarType::BOUND);
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
    offset(Index::var(name + "_offset", VarType::UNBOUND)),
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

optional<smt::Sort> MemRef::getElemTy(mlir::MemRefType memRefTy) {
  auto elemty = memRefTy.getElementType();
  if (!elemty.isa<mlir::Float32Type>())
    // Currently we only support f32 element type.
    return {};

  return Float::sort();
}

optional<MemRef::Layout> MemRef::getLayout(
    mlir::MemRefType memRefTy, const vector<Expr> &dims) {
  if (!mlir::isStrided(memRefTy)) {
    // Currently we only support strided Memref.
    return {};
  }

  auto affineMaps = memRefTy.getAffineMaps();

  if (affineMaps.empty())
    return MemRef::Layout(dims);

  int64_t offset;
  llvm::SmallVector<int64_t, 4> strides;
  auto success = mlir::getStridesAndOffset(memRefTy, strides, offset);
  assert(succeeded(success) && "unexpected non-strided memref");
  Expr layout = getConstOrFreshVar(offset, "offset");
  vector<Expr> indVars;

  Expr inbounds = Expr::mkBool(true);
  for (int i = 0; i < strides.size(); i ++) {
    indVars.push_back(Index::var("idx" + to_string(i), VarType::BOUND));
    layout = layout + getConstOrFreshVar(strides[i], "strides") * indVars[i];
    inbounds = inbounds & indVars[i].ult(dims[i]);
  }

  return MemRef::Layout(indVars, layout, inbounds);
}

pair<Expr, Expr> MemRef::get(const vector<Expr> &indices) const {
  auto [idx, inbounds] = to1DIdxWithLayout(indices);
  auto [loaded, success] = m->load(bid, (Expr)offset + idx);

  return {loaded, (success & inbounds).simplify()};
}

Expr MemRef::store(const Expr &value, const std::vector<Expr> &indices) const {
  auto [idx, inbounds] = to1DIdxWithLayout(indices);
  auto success = m->store(value, bid, (Expr)offset + idx);

  return (success & inbounds).simplify();
}

Expr MemRef::storeArray(
    const Expr &array, const Expr &startOffset, const Expr &size,
    bool ubIfReadonly) const {
  return m->storeArray(array, bid, (Expr)offset + startOffset, size,
      ubIfReadonly);
}

Expr MemRef::isInBounds() const {
  auto numelem = m->getNumElementsOfMemBlock(bid);
  auto memrefSize = get1DSize();
  return numelem.uge(memrefSize) & ((Expr)offset).ule(numelem - memrefSize);
}

Expr MemRef::isGlobalBlock() const {
  return m->isGlobalBlock(bid);
}

Expr MemRef::isLocalBlock() const {
  return m->isLocalBlock(bid);
}

smt::Expr MemRef::noalias(const MemRef &other) const {
  if (!isIdentityMap() || !other.isIdentityMap())
    assert("Noalias check with arbitrary layout memref is not supported yet");

  auto l1 = (Expr) offset;
  auto r1 = (Expr) offset + get1DSize();
  auto l2 = (Expr) other.offset;
  auto r2 = (Expr) other.offset + other.get1DSize();

  // Case 1. bid != other.bid
  // Case 2. bid == other.bid && (r2 <= l1 || r1 <= l2)
  return !(bid == other.bid) | (bid == other.bid & (r2.ule(l1) | r1.ule(l2)));
}

void MemRef::setWritable(bool writable) {
  m->setWritable(bid, writable);
}

bool MemRef::isIdentityMap() const {
  return layout.precondition.isTrue();
}

MemRef MemRef::subview(const vector<Expr> &offsets,
    const vector<Expr> &sizes,
    const vector<Expr> &strides,
    const llvm::SmallDenseSet<unsigned> &unusedDims,
    int rankDiff) {
  if (rankDiff > 0) {
    vector<Expr> indVars, reducedSizes;
    for (unsigned i = 0; i < sizes.size(); i++) {
      uint64_t size_cst;
      if (rankDiff > 0 && unusedDims.contains(i)) {
        //statically known to be 1
        indVars.push_back(Index::zero());
        rankDiff --;
      } else {
        indVars.push_back(layout.indVars[i]);
        reducedSizes.push_back(sizes[i]);
      }
    }

    auto subviewLayout = createSubViewLayout(indVars, offsets, strides, sizes);
    return MemRef(m, bid, offset, reducedSizes, subviewLayout, Float::sort());
  } else {
    auto subviewLayout = createSubViewLayout(layout.indVars, offsets, strides, sizes);
    return MemRef(m, bid, offset, sizes, subviewLayout, Float::sort());
  }
}

Expr MemRef::conv(const MemRef &input,
    const MemRef &filter,
    const std::vector<smt::Expr> strides,
    const std::vector<smt::Expr> dilations) {
  auto [indices, expr] = input.ShapedValue::conv(filter, strides, dilations);

  // we splat results into 1D memory layout
  auto idx = Index::var("outputIdx", VarType::BOUND);
  auto outputIndices = layout.getInverseIndices(idx);
  auto outputExpr = expr.substitute(indices, outputIndices);
  auto outputArray = Expr::mkLambda(idx, outputExpr);

  // store output memref
  auto success = isInBounds() & input.isInBounds() & filter.isInBounds() &
    noalias(input) & noalias(filter) & storeArray(outputArray, Index::zero(), get1DSize());

  return success;
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

pair<Expr, Expr> MemRef::to1DIdxWithLayout(const vector<Expr> &idxs) const {
  auto Expr = layout.mapping.select(idxs);
  auto inbounds = layout.inbounds.select(idxs);
  return {Expr, inbounds};
}

MemRef::Layout MemRef::createSubViewLayout(
    const vector<Expr> &indVars,
    const vector<Expr> &offsets,
    const vector<Expr> &strides,
    const vector<Expr> &sizes) {
  // Before : <(d0, d1) -> (d0 * s0 + d1)>,
  // After: <(d0, d1) -> ((indVars[0] * strides[0] + offsets[0]) * s0 + indVars[1] * strides[1] + offsets[1])>
  // indVars[i] can be Index::zero() if reducing the dimension.
  assert(layout.indVars.size() == indVars.size());
  assert(layout.indVars.size() == offsets.size());
  assert(layout.indVars.size() == strides.size());
  assert(layout.indVars.size() == sizes.size());

  vector<Expr> idxs, transformedIndVars;
  Expr inbounds = Expr::mkBool(true);
  for (unsigned i = 0; i < layout.indVars.size(); i ++) {
    idxs.push_back(indVars[i] * strides[i] + offsets[i]);
    inbounds = inbounds & indVars[i].ult(sizes[i]);

    if (!indVars[i].isNumeral()) transformedIndVars.push_back(indVars[i]);
  }
  auto transformedLayout = layout.mapping.select(idxs);
  auto transformedInbounds = layout.inbounds.select(idxs) & inbounds;
  return Layout(transformedIndVars, transformedLayout, transformedInbounds);
}

vector<Expr> MemRef::Layout::getInverseIndices(const Expr &idx) const {
  vector<Expr> indices;
  for (auto inverse: inverseMappings)
    indices.push_back(inverse.select(idx));

  return indices;
}
