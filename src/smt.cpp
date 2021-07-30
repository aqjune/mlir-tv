#include "smt.h"
#include "value.h"

using namespace std;

z3::context ctx;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const z3::expr &e) {
  std::stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}


llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const std::vector<z3::expr> &es) {
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


vector<z3::expr> from1DIdx(
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

z3::expr get1DSize(const vector<z3::expr> &dims) {
  z3::expr szaccml = Index::one();
  for (auto &d: dims)
    szaccml = szaccml * d;
  szaccml = szaccml.simplify();
  return szaccml;
}

vector<z3::expr> simplifyList(const vector<z3::expr> &exprs) {
  vector<z3::expr> v;
  v.reserve(exprs.size());
  for (auto &e: exprs)
    v.push_back(std::move(e.simplify()));
  return v;
}

z3::expr to1DIdx(
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

z3::expr fitsInDims(
    const vector<z3::expr> &idxs,
    const vector<z3::expr> &sizes) {
  assert(idxs.size() == sizes.size());

  z3::expr cond = ctx.bool_val(true);
  for (size_t i = 0; i < idxs.size(); ++i)
    cond = cond && (z3::ult(idxs[i], sizes[i]));
  return cond;
}

z3::expr_vector toExprVector(const vector<z3::expr> &vec) {
  z3::expr_vector ev(ctx);
  for (auto &e: vec)
    ev.push_back(e);
  return ev;
}

string or_omit(const z3::expr &e) {
  string s;
  llvm::raw_string_ostream rso(s);
  rso << e.simplify();
  rso.flush();

  if (s.size() > 500)
    return "(omitted)";
  return s;
}
