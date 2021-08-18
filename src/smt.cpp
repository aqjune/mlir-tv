#include "value.h"
#include "smt.h"
#include "utils.h"

using namespace std;

namespace {
z3::expr_vector toExprVector(const vector<smt::expr> &vec) {
  z3::expr_vector ev(smt::ctx);
  for (auto &e: vec)
    ev.push_back(e);
  return ev;
}
}

namespace smt {
class Context {
public:
  optional<z3::context> z3_ctx;

  Context() {
    this->z3_ctx = nullopt;
  }

  Context(bool use_z3) {
    if (use_z3) {
      this->z3_ctx.emplace();
    }
  }
};

z3::context ctx;
static Context sctx;

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
    v.push_back(move(e.simplify()));
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

  expr cond = mkBool(true);
  for (size_t i = 0; i < idxs.size(); ++i)
    cond = cond && (z3::ult(idxs[i], sizes[i]));
  return cond;
}

expr mkFreshVar(const sort &s, string &&prefix) {
  Z3_ast ast = Z3_mk_fresh_const(ctx, prefix.c_str(), s);
  return z3::expr(ctx, ast);
}

expr mkVar(const sort &s, string &&name) {
  return ctx.constant(name.c_str(), s);
}

expr mkBV(uint64_t i, unsigned bw) {
  return ctx.bv_val(i, bw);
}

expr mkBool(bool b) {
  return ctx.bool_val(b);
}

func_decl mkUF(const sort &domain, const sort &range, string &&name) {
  return ctx.function(move(name).c_str(), domain, range);
}

func_decl mkUF(
    const vector<sort> &domain,
    const sort &range,
    string &&name) {
  z3::sort_vector v(ctx);
  for (const auto &s: domain)
    v.push_back(s);
  return ctx.function(move(name).c_str(), v, range);
}

bool structurallyEq(const expr &e1, const expr &e2) {
  return (Z3_ast)e1 == (Z3_ast)e2;
}

expr substitute(
    expr e,
    const vector<expr> &vars,
    const vector<expr> &values) {
  return e.substitute(toExprVector(vars), toExprVector(values));
}

expr forall(const vector<expr> &vars, const expr &e) {
  return z3::forall(toExprVector(vars), e);
}

sort bvSort(unsigned bw) {
  return ctx.bv_sort(bw);
}

sort boolSort() {
  return ctx.bool_sort();
}

sort arraySort(const sort &domain, const sort &range) {
  return ctx.array_sort(domain, range);
}

string or_omit(const expr &e) {
  string s;
  llvm::raw_string_ostream rso(s);
  expr e2 = e.simplify();

  int64_t i;
  if (e2.is_numeral_i64(i))
    return to_string(i);
  rso << e2;
  rso.flush();

  if (s.size() > 500)
    return "(omitted)";
  return s;
}

string or_omit(const vector<expr> &evec) {
  string s;
  llvm::raw_string_ostream rso(s);
  rso << "(";

  if (evec.size() != 0) {
    rso << or_omit(evec[0]);
    for (size_t i = 1; i < evec.size(); ++i)
      rso << ", " << or_omit(evec[i]);
  }
  rso << ")";
  rso.flush();

  return s;
}

Expr::Expr(optional<z3::expr> &&z3_expr) {
  this->z3_expr = move(z3_expr);
}

Expr Expr::simplify() const {
  auto z3_expr = fmap(this->z3_expr, [](auto e) { return e.simplify(); });

  return Expr(move(z3_expr));
}

vector<Expr> Expr::toNDIndices(const vector<Expr> &dims) const {
  assert(dims.size() > 0);

  auto idx_1d = *this;
  vector<Expr> expanded_exprs;
  expanded_exprs.reserve(dims.size());
  for_each(dims.crbegin(), dims.crend(), 
    [&idx_1d, &expanded_exprs](const Expr& dim) {
      expanded_exprs.push_back(idx_1d.urem(dim));
      idx_1d = idx_1d.udiv(dim);
    });
  reverse(expanded_exprs.begin(), expanded_exprs.end());
  return expanded_exprs;
}

Expr Expr::urem(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::urem(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::udiv(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::udiv(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::ult(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::ult(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::ugt(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::ugt(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator+(const Expr &rhs) {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e + *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator-(const Expr &rhs) {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e - *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator*(const Expr &rhs) {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e * *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator&(const Expr &rhs) {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    if (e.is_bool()) 
      return e && *rhs.z3_expr;
    else
      return e & *rhs.z3_expr;
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator|(const Expr &rhs) {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    if (e.is_bool()) 
      return e || *rhs.z3_expr;
    else
      return e | *rhs.z3_expr;
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::mkFreshVar(const Sort &s, std::string_view prefix) {
  auto z3_expr = fupdate(sctx.z3_ctx, [s, prefix](auto &ctx){ 
    auto ast = Z3_mk_fresh_const(ctx, prefix.data(), *s.z3_sort);
    return z3::expr(ctx, ast);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkVar(const Sort &s, std::string_view name) {
  auto z3_expr = fupdate(sctx.z3_ctx, [s, name](auto &ctx){ 
    return ctx.constant(name.data(), *s.z3_sort);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkBV(const uint64_t val, const size_t sz) {
  auto z3_expr = fupdate(sctx.z3_ctx, [val, sz](auto &ctx){ 
    return ctx.bv_val(val, sz); 
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkBool(const bool val) {
  auto z3_expr = fupdate(sctx.z3_ctx, [val](auto &ctx){ 
    return ctx.bool_val(val); 
  });

  return Expr(move(z3_expr));
}

Sort::Sort(std::optional<z3::sort> &&z3_sort) {
  this->z3_sort = std::move(z3_sort);
}

Sort Sort::bvSort(size_t bw) {
  auto z3_sort = fupdate(sctx.z3_ctx, [bw](auto &ctx){ return ctx.bv_sort(bw); });

  return Sort(move(z3_sort));
}

Sort Sort::boolSort() {
  auto z3_sort = fupdate(sctx.z3_ctx, [](auto &ctx){ return ctx.bool_sort(); });

  return Sort(move(z3_sort));
}

Sort Sort::arraySort(const Sort &domain, const Sort &range) {
  auto z3_sort = fupdate(sctx.z3_ctx, [domain, range](auto &ctx){ 
    return ctx.array_sort(*domain.z3_sort, *range.z3_sort); 
  });

  return Sort(move(z3_sort));
}
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::expr &e) {
  stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}


llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const vector<smt::expr> &es) {
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
