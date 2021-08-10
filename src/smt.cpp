#include "smt.h"
#include "value.h"

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
  return *this;
}

std::optional<z3::expr> Expr::replaceExpr(z3::expr&& z3_expr) {
  auto prev_z3_expr = std::move(this->z3_expr);
  this->z3_expr = z3_expr;
  return prev_z3_expr;
}

std::vector<Expr> Expr::toElements(const std::vector<Expr>& dims) const {
  assert(dims.size() > 0);

  std::vector<Expr> expanded_exprs;
  expanded_exprs.reserve(dims.size());

  Expr idx_1d = this->clone();
  std::transform(dims.crbegin(), dims.crend(), expanded_exprs.begin(), 
    [idx_1d](const Expr &dim) mutable { 
      auto e = urem(idx_1d, dim);
      idx_1d = udiv(idx_1d, dim);
      return e;
    });
  
  std::reverse(expanded_exprs.begin(), expanded_exprs.end());
  return expanded_exprs;
}

Expr urem(const Expr& lhs, const Expr& rhs) {
  Expr urem_expr;

  auto z3_urem = static_cast<z3::expr(*)(const z3::expr&, const z3::expr&)>(&z3::urem);
  urem_expr.applyZ3Operation(z3_urem, lhs, rhs);

  return urem_expr;
}

Expr udiv(const Expr& lhs, const Expr& rhs) {
  Expr udiv_expr;

  auto z3_udiv = static_cast<z3::expr(*)(const z3::expr&, const z3::expr&)>(&z3::udiv);
  udiv_expr.applyZ3Operation(z3_udiv, lhs, rhs);

  return udiv_expr;
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
