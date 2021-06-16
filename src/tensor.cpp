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
  z3::expr szaccml = ctx.bv_val(1, Tensor::BITS_INDEX);
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

static z3::expr dot(const z3::expr &a, const z3::expr &b, const z3::expr &n) {
  auto ity = ctx.bv_sort(Tensor::BITS_INDEX),
       fty = ctx.bv_sort(Tensor::BITS_FLOAT);
  auto aty = ctx.array_sort(ity, fty);
  auto i = ctx.bv_const("idx", Tensor::BITS_INDEX);

  z3::sort_vector domain(ctx);
  domain.push_back(aty);
  domain.push_back(aty);
  auto dotfn = ctx.function("smt_dot", domain, fty);

  z3::expr_vector args(ctx);
  z3::expr zero = ctx.bv_val(0, Tensor::BITS_FLOAT);
  args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), z3::select(a, i), zero)));
  args.push_back(z3::lambda(i, z3::ite(z3::ult(i, n), z3::select(b, i), zero)));
  return dotfn(args);
}

static vector<z3::expr> simplifyList(const vector<z3::expr> &exprs) {
  vector<z3::expr> v;
  for (auto &e: exprs)
    v.emplace_back(e.simplify());
  return v;
}


Tensor::Tensor(): arr(ctx) {}

z3::expr Tensor::get(const vector<z3::expr> &idxs) const {
  return z3::select(arr, to1DIdx(idxs, dims));
}

Tensor Tensor::affine(
    const std::vector<z3::expr> &newidxvars,
    std::vector<z3::expr> srcidxs,
    const std::vector<z3::expr> &newsizes) const {
  auto idxvar = newIdxVar("idx");
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
        ctx.bv_val(0, BITS_FLOAT)
      ));
  return newm;
}

Tensor Tensor::rotateDimensions() const {
  vector<z3::expr> newdims;
  for (size_t i = 1; i < dims.size(); ++i)
    newdims.emplace_back(dims[i]);
  newdims.emplace_back(dims[0]);

  vector<z3::expr> vars, tgtvars;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto v = newIdxVar(string("i" + to_string(i)));
    vars.emplace_back(v);
    if (i != 0)
      tgtvars.emplace_back(v);
  }
  tgtvars.emplace_back(vars[0]);

  return affine(vars, tgtvars, newdims);
}

Tensor Tensor::conv(const Tensor &filter) const {
  vector<z3::expr> output_dims = {
    Tensor::newIdxConst(1), // support an input with batch size > 1
    dims[1] + 1 - filter.dims[0],
    dims[2] + 1 - filter.dims[1],
    filter.dims[3] // channel(dims[3] = filtr.dims[2]) disappears
  };
  std::vector<z3::expr> cube_size = {
    Tensor::newIdxConst(1),
    filter.dims[0], filter.dims[1], filter.dims[2]
  };

  auto vars = Tensor::newIdxVars({"i", "j", "k", "l"}); // n, h, w, f
  auto i = vars[0], j = vars[1], k = vars[2], l = vars[3];
  auto zero = Tensor::newIdxConst(0);
  auto input_subarr = to1DArrayWithOfs(
      {zero, j, k, zero}, // batch: 0, img size: (h, w), channel: 0~
      cube_size);

  auto filter_arr = filter.rotateDimensions()
      .to1DArrayWithOfs({l, zero, zero, zero}, cube_size);

  auto res = dot(input_subarr, filter_arr,
      cube_size[0] * cube_size[1] * cube_size[2] * cube_size[3]);

  Tensor res_t;
  res_t.dims = move(output_dims);
  res_t.arr = move(res);
  return res_t;
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
  auto i = newIdxVar("i"), j = newIdxVar("j"), zero = newIdxConst(0),
       one = newIdxConst(1);
  auto a_row = to1DArrayWithOfs({i, zero}, {one, dims[1]});
  auto b_row = bt.to1DArrayWithOfs({j, zero}, {one, bt.dims[1]});

  return mkLambda({dims[0], bt.dims[0]}, {i, j}, dot(a_row, b_row, dims[1]));
}

vector<z3::expr> Tensor::getDims(mlir::TensorType tensorTy) {
  vector<z3::expr> dims;

  uint64_t rank = tensorTy.getRank();
  for (auto i = 0; i < rank; ++i) {
    dims.emplace_back(ctx.bv_val(tensorTy.getDimSize(i), BITS_INDEX));
  }

  return dims;
}

Tensor Tensor::newVar(mlir::TensorType tensorTy, const string &name) {
  Tensor t;
  t.dims = getDims(tensorTy);
  t.arr = ctx.constant(name.c_str(),
        ctx.array_sort(ctx.bv_sort(BITS_INDEX), ctx.bv_sort(BITS_FLOAT)));

  return t;
}

z3::expr Tensor::newIdxConst(uint64_t idx) {
  return ctx.bv_val(idx, BITS_INDEX);
}

z3::expr Tensor::newIdxVar(const string &name) {
  return ctx.bv_const(name.c_str(), BITS_INDEX);
}

vector<z3::expr>
Tensor::newIdxVars(const vector<string> &names) {
  vector<z3::expr> v;
  for (auto &n: names)
    v.emplace_back(newIdxVar(n));
  return v;
}


llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Tensor &t) {
  assert(t.dims.size() > 0);
  os << t.arr << "(dim :" << t.dims[0];
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << t.dims[i];
  os << ")";
  return os;
};

Tensor Tensor::transpose() const {
  assert(dims.size() == 2);
  auto i = newIdxVar("i"), j = newIdxVar("j");
  return Tensor::mkLambda({dims[1], dims[0]}, {j, i}, get({i, j}));
}

Tensor Tensor::mkLambda(
    std::vector<z3::expr> &&newdims, std::vector<z3::expr> &&indexvars,
    z3::expr body) {
  assert(newdims.size() == indexvars.size());

  auto idx = newIdxVar("idx");
  auto idxexprs = from1DIdx(idx, newdims);
  body = body.substitute(toExprVector(indexvars), toExprVector(idxexprs));

  Tensor t2;
  t2.dims = move(newdims);
  t2.arr = body;
  return t2;
}

z3::expr Tensor::to1DArrayWithOfs(
      const vector<z3::expr> &offbegins,
      const vector<z3::expr> &sizes) const {
  assert(offbegins.size() == sizes.size());

  auto idxvar = newIdxVar("idx");
  auto relidxs = from1DIdx(idxvar, sizes);
  vector<z3::expr> absidxs;
  for (size_t i = 0; i < relidxs.size(); ++i)
    absidxs.emplace_back(relidxs[i] + offbegins[i]);

  return z3::lambda(
      idxvar,
      z3::ite(
        z3::ult(idxvar, get1DSize(sizes)),
        get(absidxs),
        ctx.bv_val(0, BITS_FLOAT)));
}
