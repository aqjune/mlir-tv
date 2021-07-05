#include "tensor.h"
#include "smt.h"

using namespace std;

static z3::expr to1DIdx(
    const vector<z3::expr> &idxs,
    const vector<z3::expr> &dims) {
  assert(idxs.size() == dims.size());
  auto idx = idxs[0];

  for (size_t i = 1; i < idxs.size(); ++i) {
    // TODO: migrate constant foldings
    idx = idx * dims[i] + idxs[i];
  }
  return idx;
}

static vector<z3::expr> from1DIdx(
    z3::expr idx1d,
    const vector<z3::expr> &dims) {
  assert(dims.size() > 0);
  vector<z3::expr> idxs;

  for (size_t ii = dims.size(); ii > 0; --ii) {
    size_t i = ii - 1;
    // TODO: migrate constant foldings & simplifications
    auto a = z3::urem(idx1d, dims[i]), b = z3::udiv(idx1d, dims[i]);
    idxs.emplace_back(a);
    idx1d = b;
  }

  reverse(idxs.begin(), idxs.end());
  return idxs;
}

static z3::expr get1DSize(const vector<z3::expr> &dims) {
  z3::expr szaccml = Index::one();
  for (auto &d: dims)
    szaccml = szaccml * d;
  szaccml = szaccml.simplify();
  return szaccml;
}

static z3::expr fitsInDims(
    const vector<z3::expr> &idxs,
    const vector<z3::expr> &sizes) {
  assert(idxs.size() == sizes.size());

  z3::expr cond = ctx.bool_val(true);
  for (size_t i = 0; i < idxs.size(); ++i)
    cond = cond && (z3::ult(idxs[i], sizes[i]));
  return cond;
}

static z3::expr_vector toExprVector(const vector<z3::expr> &vec) {
  z3::expr_vector ev(ctx);
  for (auto &e: vec)
    ev.push_back(e);
  return ev;
}

static z3::expr mkZeroElemFromArr(const z3::expr &arr) {
  unsigned bvsz = z3::select(arr, Index::zero()).get_sort().bv_size();
  return ctx.bv_val(0, bvsz);
}

static z3::expr
abstractDot(const z3::expr &a, const z3::expr &b, const z3::expr &n) {
  // TODO: check that a.get_sort() == b.get_sort()
  auto i = Index("idx");

  z3::sort_vector domain(ctx);
  domain.push_back(a.get_sort());
  domain.push_back(b.get_sort());
  auto dotfn = ctx.function("smt_dot", domain, Float::sort());

  z3::expr_vector args(ctx);
  z3::expr ai = z3::select(a, i), bi = z3::select(b, i);
  z3::expr zero = mkZeroElemFromArr(a);
  args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), ai, zero)));
  args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), bi, zero)));
  return dotfn(args);
}

static vector<z3::expr> simplifyList(const vector<z3::expr> &exprs) {
  vector<z3::expr> v;
  for (auto &e: exprs)
    v.emplace_back(e.simplify());
  return v;
}


Index::Index(): e(ctx) {}

Index::Index(unsigned i): e(ctx.bv_val(i, BITS)) {}

Index::Index(const std::string &name): e(ctx.bv_const(name.c_str(), BITS)) {}

Index::Index(const z3::expr &e): e(e) {}

z3::sort Index::sort() {
  return ctx.bv_sort(BITS);
}

Index Index::one() { return Index(1); }
Index Index::zero() { return Index(0); }

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index &i) {
  os << (z3::expr)i;
  return os;
};

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
  os << (z3::expr)f;
  return os;
};

Float Float::eval(z3::model m) const {
  return Float(m.eval(e, true).simplify());
}

Float Float::add(const Float &b) const {
  auto fty = Float::sort();

  z3::sort_vector domain(ctx);
  domain.push_back(fty);
  domain.push_back(fty);
  auto addfn = ctx.function("smt_add", domain, fty);

  z3::expr_vector args(ctx);
  args.push_back(e);
  args.push_back(b.e);
  return addfn(args);
}

Float Float::mul(const Float &b) const {
  auto fty = Float::sort();

  z3::sort_vector domain(ctx);
  domain.push_back(fty);
  domain.push_back(fty);
  auto mulfn = ctx.function("smt_mul", domain, fty);

  z3::expr_vector args(ctx);
  args.push_back(e);
  args.push_back(b.e);
  return mulfn(args);
}


Integer::Integer(const std::string &name, unsigned bw):
  e(ctx.bv_const(name.c_str(), bw)) {}

z3::sort Integer::sort(unsigned sz) {
  return ctx.bv_sort(sz);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Integer &i) {
  os << (z3::expr)i;
  return os;
};

Integer Integer::eval(z3::model m) const {
  return Integer(m.eval(e, true).simplify());
}


Tensor::Tensor(): arr(ctx) {}

Tensor::Tensor(const z3::expr &splat_elem, const vector<z3::expr> &dimvec):
    arr(ctx), dims(dimvec) {
  arr = z3::const_array(Index::sort(), splat_elem);
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
        z3::ult(idxvar, get1DSize(newsizes)),
        get(srcidxs),
        mkZeroElemFromArr(arr)
      ));
  return newm;
}

Tensor Tensor::rotateDimensions() const {
  vector<z3::expr> newdims;
  newdims.emplace_back(dims.back());
  for (size_t i = 0; i < dims.size() - 1; ++i)
    newdims.emplace_back(dims[i]);

  vector<z3::expr> vars, tgtvars;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = Index(string("i" + to_string(i)));
    vars.emplace_back(v);
    if (i != 0)
      tgtvars.emplace_back(v);
  }
  tgtvars.emplace_back(vars[0]);

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

  auto res = abstractDot(input_subarr, filter_arr,
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
      abstractDot(a_row, bt_row, dims[1]));
}

pair<z3::expr, z3::expr> Tensor::refines(const Tensor &src) const {
  assert(arr.get_sort().is_array());
  assert(src.arr.get_sort().is_array());

  // Assume that src and tgt's shape equality is already checked
  auto i = Index("i");
  return {z3::implies(
      z3::ult(i, get1DSize(dims)),
      z3::select(arr, i) == z3::select(src.arr, i)),
    i};
}


vector<z3::expr> Tensor::getDims(mlir::TensorType tensorTy) {
  vector<z3::expr> dims;
  //static int dim_var = 0;

  uint64_t rank = tensorTy.getRank();
  if (rank == 0) {
    // A single element tensor.
    dims.emplace_back(Index(1));
    return dims;
  }

  for (auto i = 0; i < rank; ++i) {
    uint64_t sz = tensorTy.getDimSize(i);
    if (sz == (uint64_t)-1ull)
      // TODO: this requires encoding of well-formedness of input tensors.
      // dims.emplace_back(Index("dim" + to_string(dim_var++)));
      dims.emplace_back(Index(100));
    else
      dims.emplace_back(Index(sz));
  }

  return dims;
}

optional<pair<vector<z3::expr>, z3::sort>>
Tensor::getDimsAndElemTy(mlir::TensorType tensorTy) {
  auto elemty = tensorTy.getElementType();
  z3::sort elemty2(ctx);

  if (auto ielemty = elemty.dyn_cast<mlir::IntegerType>()) {
    elemty2 = Integer::sort(ielemty.getWidth());
  } else if (auto felemty = elemty.dyn_cast<mlir::Float32Type>()) {
    elemty2 = Float::sort();
  } else {
    return {};
  }

  return {{getDims(tensorTy), elemty2}};
}


llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Tensor &t) {
  assert(t.dims.size() > 0);
  os << t.arr << "(dim :" << t.dims[0];
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << t.dims[i];
  os << ")";
  return os;
};

Tensor Tensor::eval(z3::model m) const {
  Tensor t2;
  for (size_t i = 0; i < dims.size(); ++i)
    t2.dims.emplace_back(m.eval(dims[i], true).simplify());
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
  for (size_t i = 0; i < relidxs.size(); ++i)
    absidxs.emplace_back(relidxs[i] + offbegins[i]);

  return z3::lambda(
      idxvar,
      z3::ite(
        z3::ult(idxvar, get1DSize(sizes)),
        get(absidxs),
        mkZeroElemFromArr(arr)));
}
