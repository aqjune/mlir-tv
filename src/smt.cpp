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
optional<T> fmap(const optional<T> &x, Fn fn) {
  if (!x)
    return std::nullopt;
  return {fn(*x)};
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


Expr Expr::urem(const Expr& rhs) const {
  return {fmap(z3_expr, [&](auto e) { return z3::urem(e, *rhs.z3_expr); })};
}

Expr Expr::udiv(const Expr& rhs) const {
  return {fmap(z3_expr, [&](auto e) { return z3::udiv(e, *rhs.z3_expr); })};
}

Expr Expr::simplify() const {
  return {fmap(z3_expr, [](auto e) { return e.simplify(); })};
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
