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

Expr::Expr() {
  this->z3_expr = {};
}

Expr::Expr(const Expr& from) {
  this->z3_expr = from.z3_expr;
}

Expr::Expr(Expr&& from) {
  this->z3_expr = std::move(from.z3_expr);
  from.z3_expr = {}; // moving optional do not set 'moved' optional to none
}

void Expr::applyZ3Operation(std::function<z3::expr(z3::expr const&)>&& op, const Expr& arg0) {
  if (arg0.z3_expr.has_value()) {
    this->z3_expr = op(arg0.z3_expr.value());
  }
}

void Expr::applyZ3Operation(std::function<z3::expr(z3::expr const&, z3::expr const&)>&& op, const Expr& arg0, const Expr& arg1) {
  if (arg0.z3_expr.has_value() && arg1.z3_expr.has_value()) {
    this->z3_expr = op(arg0.z3_expr.value(), arg1.z3_expr.value());
  }
}

Expr Expr::clone() const {
  return Expr(*this);
}

Expr& Expr::operator=(Expr&& from) {
  this->z3_expr = std::move(from.z3_expr);
  from.z3_expr = {}; // moving optional do not set 'moved' optional to none
  return *this;
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

  auto acc = std::accumulate(dims.crbegin(), dims.crend(), 
    std::make_pair(this->clone(), std::move(exprs)),
    [](std::pair<Expr, std::vector<Expr>>& acc, const Expr& dim) {
      auto [idx_1d, expanded_exprs] = std::move(acc);
      expanded_exprs.push_back(urem(idx_1d, dim));
      idx_1d = udiv(idx_1d, dim);
      return std::make_pair(std::move(idx_1d), std::move(expanded_exprs));
    });
  
  auto expanded_exprs = std::move(acc.second);
  std::reverse(expanded_exprs.begin(), expanded_exprs.end());
  return expanded_exprs;
}

Expr urem(const Expr& lhs, const Expr& rhs) {
  Expr e;
  e.applyZ3Operation(
    [](const z3::expr &lhs, const z3::expr &rhs) { return z3::urem(lhs, rhs); },
    lhs, rhs);

  return e;
}

Expr udiv(const Expr& lhs, const Expr& rhs) {
  Expr e;
  e.applyZ3Operation(
    [](const z3::expr &lhs, const z3::expr &rhs) { return z3::udiv(lhs, rhs); },
    lhs, rhs);

  return e;
}

Expr Expr::simplify() const {
  Expr e;
  e.applyZ3Operation(
    [](const z3::expr &e) { return e.simplify(); }, *this);

  return e;
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
