#include "smt.h"
#include "value.h"
#include <numeric>

using namespace std;

namespace {

// optional::map from
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0798r0.html
// Fn is simply declared because std::function with template arguments works
// poorly. :(
template<class T, class Fn>
optional<T> fmap(const optional<T> &x, Fn &&fn) {
  if (!x)
    return std::nullopt;
  return {fn(*x)};
}

template<class Fn>
std::optional<z3::expr> fmap(z3::context *ctx, Fn &&fn) {
  if (!ctx)
    return std::nullopt;
  return std::make_optional(fn(ctx));
}
}

namespace smt {
z3::context ctx;

vector<expr> from1DIdx(
    expr idx1d,
    const vector<expr> &dims) {
  assert(dims.size() > 0);
  vector<expr> idxs;

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

expr get1DSize(const vector<expr> &dims) {
  expr szaccml = Index::one();
  for (auto &d: dims)
    szaccml = szaccml * d;
  szaccml = szaccml.simplify();
  return szaccml;
}

vector<expr> simplifyList(const vector<expr> &exprs) {
  vector<expr> v;
  v.reserve(exprs.size());
  for (auto &e: exprs)
    v.push_back(std::move(e.simplify()));
  return v;
}

expr to1DIdx(
    const vector<expr> &idxs,
    const vector<expr> &dims) {
  assert(idxs.size() == dims.size());
  auto idx = idxs[0];

  for (size_t i = 1; i < idxs.size(); ++i) {
    // TODO: migrate constant foldings
    idx = idx * dims[i] + idxs[i];
  }
  return idx;
}

expr to1DIdxWithLayout(const vector<expr> &idxs, expr layout) {
  vector<expr> indices;
  for (unsigned i = 0; i < idxs.size(); i ++)
    indices.push_back(Index("idx" + to_string(i)));

  return layout.substitute(toExprVector(indices), toExprVector(idxs));
}

expr fitsInDims(
    const vector<expr> &idxs,
    const vector<expr> &sizes) {
  assert(idxs.size() == sizes.size());

  expr cond = ctx.bool_val(true);
  for (size_t i = 0; i < idxs.size(); ++i)
    cond = cond && (z3::ult(idxs[i], sizes[i]));
  return cond;
}

z3::expr_vector toExprVector(const vector<expr> &vec) {
  z3::expr_vector ev(ctx);
  for (auto &e: vec)
    ev.push_back(e);
  return ev;
}

string or_omit(const expr &e) {
  string s;
  llvm::raw_string_ostream rso(s);
  rso << e.simplify();
  rso.flush();

  if (s.size() > 500)
    return "(omitted)";
  return s;
}

ContextBuilder::ContextBuilder() {
  use_z3 = false;
}

ContextBuilder& ContextBuilder::useZ3() {
  use_z3 = true;
  return *this;
}

std::optional<Context> ContextBuilder::build() const {
  if (!use_z3) {
    return std::nullopt;
  }
  return std::make_optional(Context(use_z3));
}

Context::Context() {
  z3_ctx = nullptr;
}

Context::Context(bool use_z3) : Context() {
  if (use_z3) {
    z3_ctx = &ctx;
  }
}

Expr Context::bvVal(const uint32_t val, const size_t sz) {
  auto z3_expr = fmap(z3_ctx, [val, sz](auto ctx){ return ctx->bv_val(val, sz); });

  return Expr(this, std::move(z3_expr));
}

Expr Context::bvConst(char* const name, const size_t sz) {
  auto z3_expr = fmap(z3_ctx, [name, sz](auto ctx){ return ctx->bv_const(name, sz); });

  return Expr(this, std::move(z3_expr));
}

Expr Context::boolVal(const bool val) {
  auto z3_expr = fmap(z3_ctx, [val](auto ctx){ return ctx->bool_val(val); });

  return Expr(this, std::move(z3_expr));
}

Expr::Expr(Context* const ctx, std::optional<z3::expr> &&z3_expr) : Expr(ctx) {
  this->z3_expr = std::move(z3_expr);
}

Expr::Expr(Expr&& from) {
  this->ctx = from.ctx;
  this->z3_expr = std::move(from.z3_expr);
}

Expr Expr::clone() const {
  auto cloned_z3_expr = this->z3_expr;
  return Expr(this->ctx, std::move(cloned_z3_expr));
}

Expr Expr::simplify() const {
  auto z3_expr = fmap(this->z3_expr, [](auto e) { return e.simplify(); });

  return Expr(this->ctx, std::move(z3_expr));
}

ExprVec Expr::toNDIndices(const ExprVec &dims) const {
  assert(dims.exprs.size() > 0);

  auto idx_1d = this->clone();
  auto expanded_exprs = ExprVec::withCapacity(this->ctx, dims.exprs.size());
  std::for_each(dims.exprs.crbegin(), dims.exprs.crend(), 
    [&idx_1d, &expanded_exprs](const Expr& dim) {
      expanded_exprs.exprs.push_back(idx_1d.urem(dim));
      idx_1d = idx_1d.udiv(dim);
    });
  std::reverse(expanded_exprs.exprs.begin(), expanded_exprs.exprs.end());
  return expanded_exprs;
}

Expr Expr::urem(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return z3::urem(e, *rhs.z3_expr); });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::udiv(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return z3::udiv(e, *rhs.z3_expr); });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::add(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e + *rhs.z3_expr; });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::sub(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e - *rhs.z3_expr; });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::mul(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e * *rhs.z3_expr; });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::ult(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return z3::ult(e, *rhs.z3_expr); });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::ugt(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return z3::ugt(e, *rhs.z3_expr); });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::boolAnd(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e && *rhs.z3_expr; });
  
  return Expr(this->ctx, std::move(z3_expr));
}

Expr Expr::boolOr(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e || *rhs.z3_expr; });
  
  return Expr(this->ctx, std::move(z3_expr));
}

ExprVec::ExprVec(Context* const ctx, std::vector<Expr>&& exprs) : ExprVec(ctx) {
  this->exprs = std::move(exprs);
}

ExprVec::ExprVec(ExprVec&& from) : ExprVec(from.ctx) {
  this->exprs = std::move(from.exprs);
}

ExprVec ExprVec::withCapacity(Context* const ctx, size_t size) {
  std::vector<Expr> exprs;
  exprs.reserve(size);
  return ExprVec(ctx, std::move(exprs));
}

ExprVec ExprVec::clone() const {
  auto cloned_exprs = ExprVec::withCapacity(this->ctx, this->exprs.size());
  std::transform(this->exprs.cbegin(), this->exprs.cend(), 
    std::back_inserter(cloned_exprs.exprs),
    [](const Expr &expr){ return expr.clone(); });
  return cloned_exprs;
}

ExprVec ExprVec::simplify() const {
  auto simplified_exprs = ExprVec::withCapacity(this->ctx, this->exprs.size());

  std::transform(this->exprs.cbegin(), this->exprs.cend(), 
    std::back_inserter(simplified_exprs.exprs), 
    [](const Expr &expr) { return expr.simplify(); });
  
  return simplified_exprs;
}

Expr ExprVec::to1DIndices(const ExprVec &dims) const {
  assert(this->exprs.size() == dims.exprs.size());
  
  auto idx = this->exprs[0].clone();
  for (size_t i = 1; i < this->exprs.size(); i++) {
    idx = idx.mul(dims.exprs[i]).add(this->exprs[i]);
  }
  return idx;
}

Expr ExprVec::fitsInDims(const ExprVec &sizes) const {
  assert(this->exprs.size() == sizes.exprs.size());

  Expr cond = this->ctx->boolVal(true);
  for (size_t i = 0; i < this->exprs.size(); i++) {
    cond = cond.boolAnd(this->exprs[0].ult(sizes.exprs[0]));
  }
  return cond;
}
} // namespace smt


void test() {
  auto ct = smt::ContextBuilder().useZ3().build().value();
  auto e1 = ct.bvVal(32, 32);
  auto e2 = ct.bvVal(32, 32);
  e1.add(e2);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e) {
  std::stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}


llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<smt::expr> &es) {
  os << "(";
  if (es.size() != 0) {
    os << es[0];
    for (size_t i = 1; i < es.size(); ++i) {
      os << ", " << es[i];
    }
  }
  os << ")";
  return os;
}
