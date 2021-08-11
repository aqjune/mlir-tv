#include "smt.h"
#include "value.h"
#include <numeric>

using namespace std;

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

Context::Context() {
  Context::z3_ctx = nullptr;
}

void Context::useZ3() {
  Context::z3_ctx = &ctx;
}

Expr Context::bvVal(const uint32_t val, const size_t sz) {
  auto z3_expr = applyZ3Op(
    [](z3::context* ctx, const uint32_t val, const size_t sz) { return ctx->bv_val(val, sz); },
    this->z3_ctx, val, sz);
  
  return Expr(std::move(z3_expr));
}

Expr Context::bvConst(char* const name, const size_t sz) {
  auto z3_expr = applyZ3Op(
    [](z3::context* ctx, char* const name, const size_t sz) { return ctx->bv_const(name, sz); },
    this->z3_ctx, name, sz);
  
  return Expr(std::move(z3_expr));
}

Expr::Expr(std::optional<z3::expr>&& z3_expr) {
  this->z3_expr = z3_expr;
}

std::optional<z3::expr> Expr::replaceExpr(z3::expr&& z3_expr) {
  auto prev_z3_expr = std::move(this->z3_expr);
  this->z3_expr = z3_expr;
  return prev_z3_expr;
}

std::vector<Expr> Expr::toElements(const std::vector<Expr>& dims) const {
  assert(dims.size() > 0);

  std::vector<Expr> exprs;
  exprs.reserve(dims.size());

  auto expanded_exprs = std::accumulate(dims.crbegin(), dims.crend(), 
    std::make_pair(Expr(*this), std::move(exprs)),
    [](std::pair<Expr, std::vector<Expr>>& acc, const Expr& dim) {
      auto [idx_1d, expanded_exprs] = std::move(acc);
      expanded_exprs.push_back(idx_1d.urem(dim));
      idx_1d = idx_1d.udiv(dim);
      return std::make_pair(std::move(idx_1d), std::move(expanded_exprs));
    })
    .second;
  std::reverse(expanded_exprs.begin(), expanded_exprs.end());
  return expanded_exprs;
}

Expr Expr::urem(const Expr& rhs) const {
  Expr e;
  e.applyZ3Op(
    [](const z3::expr &lhs, const z3::expr &rhs) { return z3::urem(lhs, rhs); },
    *this, rhs);

  return e;
}

Expr Expr::udiv(const Expr& rhs) const {
  Expr e;
  e.applyZ3Op(
    [](const z3::expr &lhs, const z3::expr &rhs) { return z3::udiv(lhs, rhs); },
    *this, rhs);

  return e;
}

Expr Expr::simplify() const {
  Expr e;
  e.applyZ3Op(
    [](const z3::expr &e) { return e.simplify(); }, *this);

  return e;
}

ExprVec::ExprVec(std::vector<Expr>&& exprs) {
  this->exprs = std::move(exprs);
}

ExprVec::ExprVec(ExprVec&& from) {
  this->exprs = std::move(from.exprs);
}

size_t ExprVec::size() const {
  return this->exprs.size();
}

ExprVec ExprVec::simplify() const {
  std::vector<Expr> simplified_exprs;
  simplified_exprs.reserve(this->exprs.size());

  std::transform(this->exprs.cbegin(), this->exprs.cend(), simplified_exprs.begin(), 
    [](const Expr &expr) { return expr.simplify(); });
  
  return ExprVec(std::move(simplified_exprs));
}

std::vector<Expr>::const_iterator ExprVec::cbegin() const {
  return this->exprs.cbegin();
}
std::vector<Expr>::const_iterator ExprVec::cend() const {
  return this->exprs.cend();
}
std::vector<Expr>::const_reverse_iterator ExprVec::crbegin() const {
  return this->exprs.crbegin();
}
std::vector<Expr>::const_reverse_iterator ExprVec::crend() const {
  return this->exprs.crend();
}
} // namespace smt

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
