#include "value.h"
#include "smt.h"
#include "utils.h"

#include <unordered_map>
#include <numeric>

#ifdef SOLVER_Z3
#define TRY_SET_Z3_EXPR(fn) e.setZ3Expr(fn)
#define TRY_SET_Z3_SORT(fn) s.setZ3Sort(fn)
#else
#define TRY_SET_Z3_EXPR(fn)
#define TRY_SET_Z3_SORT(fn)
#endif

#ifdef SOLVER_CVC5
#define TRY_SET_CVC5_EXPR(fn) e.setCVC5Expr(fn)
#define TRY_SET_CVC5_SORT(fn) s.setCVC5Sort(fn)
#else
#define TRY_SET_CVC5_EXPR(fn)
#define TRY_SET_CVC5_SORT(fn)
#endif

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
private:
  unordered_map<string, uint64_t> fresh_var_map;

public:
  IF_Z3_ENABLED(optional<z3::context> z3_ctx);
  IF_CVC5_ENABLED(optional<cvc5::api::Solver> cvc5_ctx);

  Context() {
    IF_Z3_ENABLED(this->z3_ctx.reset());
    IF_CVC5_ENABLED(this->cvc5_ctx.reset());
  }

  IF_Z3_ENABLED(void useZ3() { this->z3_ctx.emplace(); })
  IF_CVC5_ENABLED(void useCVC5() { this->cvc5_ctx.emplace(); })

  string getFreshName(string prefix) {
    this->fresh_var_map.insert({prefix, 0});
    uint64_t suffix = fresh_var_map.at(prefix)++;
    return prefix.append("_" + to_string(suffix));
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

expr fapply(const func_decl &func, const vector<expr> &vars) {
  return func(toExprVector(vars));
}

bool structurallyEq(const expr &e1, const expr &e2) {
  return (Z3_ast)e1 == (Z3_ast)e2;
}

expr init() { return ctx; }

expr substitute(
    expr e,
    const vector<expr> &vars,
    const vector<expr> &values) {
  return e.substitute(toExprVector(vars), toExprVector(values));
}

expr implies(const expr &a, const expr &b) {
  return z3::implies(a, b);
}

expr forall(const vector<expr> &vars, const expr &e) {
  return z3::forall(toExprVector(vars), e);
}

expr lambda(const expr &var, const expr &e) {
  return z3::lambda(var, e);
}

expr lambda(const vector<expr> &vars, const expr &e) {
  return z3::lambda(toExprVector(vars), e);
}

expr select(const expr &arr, const expr &idx) {
  return z3::select(arr, idx);
}

expr select(const expr &arr, const vector<expr> &idxs) {
  return z3::select(arr, toExprVector(idxs));
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

Expr::Expr() {
  IF_Z3_ENABLED(this->z3_expr.reset());
  IF_CVC5_ENABLED(this->cvc5_expr.reset());
}

#ifdef SOLVER_Z3
void Expr::setZ3Expr(optional<z3::expr> &&z3_expr) {
  this->z3_expr = move(z3_expr);
}
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
void Expr::setCVC5Expr(optional<cvc5::api::Term> &&cvc5_expr) {
  this->cvc5_expr = move(cvc5_expr);
}
#endif // SOLVER_CVC5

Expr Expr::simplify() const {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [](auto e) {
    return e.simplify();
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this](auto &ctx) {
    return ctx.simplify(*this->cvc5_expr);
  }));

  return e;
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
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::urem(e, *rhs.z3_expr); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_UREM,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::udiv(const Expr& rhs) const {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { 
    return z3::udiv(e, *rhs.z3_expr); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_UDIV,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::ult(const Expr& rhs) const {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { 
    return z3::ult(e, *rhs.z3_expr); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_ULT,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::ugt(const Expr& rhs) const {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { 
    return z3::ugt(e, *rhs.z3_expr); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_UGT,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::operator+(const Expr &rhs) {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { return e + *rhs.z3_expr; }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_ADD,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::operator-(const Expr &rhs) {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { return e - *rhs.z3_expr; }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_SUB,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::operator*(const Expr &rhs) {
  Expr e;
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { return e * *rhs.z3_expr; }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) { 
    return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_MULT,
      *this->cvc5_expr, *rhs.cvc5_expr);
  }));
  
  return e;
}

Expr Expr::operator&(const Expr &rhs) {
  Expr e;
  // z3::expr::operator& automatically disambiguates bool and bv
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { return e & *rhs.z3_expr; }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) {
    if (this->cvc5_expr->isBooleanValue()) {
      return ctx.mkTerm(cvc5::api::Kind::AND,
        *this->cvc5_expr, *rhs.cvc5_expr);
    } else {
      return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_AND,
        *this->cvc5_expr, *rhs.cvc5_expr);
    }
  }));
  
  return e;
}

Expr Expr::operator|(const Expr &rhs) {
  Expr e;
  // z3::expr::operator| automatically disambiguates bool and bv
  TRY_SET_Z3_EXPR(fmap(this->z3_expr, [rhs](auto e) { return e | *rhs.z3_expr; }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [this, rhs](auto &ctx) {
    if (this->cvc5_expr->isBooleanValue()) {
      return ctx.mkTerm(cvc5::api::Kind::OR,
        *this->cvc5_expr, *rhs.cvc5_expr);
    } else {
      return ctx.mkTerm(cvc5::api::Kind::BITVECTOR_OR,
        *this->cvc5_expr, *rhs.cvc5_expr);
    }
  }));
  
  return e;
}

Expr Expr::mkFreshVar(const Sort &s, const string &prefix) {
  string fresh_name = sctx.getFreshName(prefix);

  Expr e;
  TRY_SET_Z3_EXPR(fupdate(sctx.z3_ctx, [s, fresh_name](auto &ctx) { 
    return ctx.constant(fresh_name.c_str(), *s.z3_sort);
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [s, fresh_name](auto &ctx) { 
    return ctx.mkConst(*s.cvc5_sort, fresh_name);
  }));

  return e;
}

Expr Expr::mkVar(const Sort &s, const string &name) {
  Expr e;
  TRY_SET_Z3_EXPR(fupdate(sctx.z3_ctx, [s, name](auto &ctx){ 
    return ctx.constant(name.c_str(), *s.z3_sort);
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [s, name](auto &ctx){ 
    return ctx.mkConst(*s.cvc5_sort, name);
  }));

  return e;
}

Expr Expr::mkBV(const uint64_t val, const size_t sz) {
  Expr e;
  TRY_SET_Z3_EXPR(fupdate(sctx.z3_ctx, [val, sz](auto &ctx){ 
    return ctx.bv_val(val, sz); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [val, sz](auto &ctx){ 
    return ctx.mkBitVector(sz, val); 
  }));

  return e;
}

Expr Expr::mkBool(const bool val) {
  Expr e;
  TRY_SET_Z3_EXPR(fupdate(sctx.z3_ctx, [val](auto &ctx){ 
    return ctx.bool_val(val); 
  }));
  TRY_SET_CVC5_EXPR(fupdate(sctx.cvc5_ctx, [val](auto &ctx){ 
    return ctx.mkBoolean(val); 
  }));

  return e;
}

Sort::Sort() {
  IF_Z3_ENABLED(this->z3_sort.reset());
  IF_CVC5_ENABLED(this->cvc5_sort.reset());
}

#ifdef SOLVER_Z3
void Sort::setZ3Sort(optional<z3::sort> &&z3_sort) {
  this->z3_sort = move(z3_sort);
}
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
void Sort::setCVC5Sort(optional<cvc5::api::Sort> &&cvc5_sort) {
  this->cvc5_sort = move(cvc5_sort);
}
#endif // SOLVER_CVC5

Sort Sort::bvSort(size_t bw) {
  Sort s;
  TRY_SET_Z3_SORT(fupdate(sctx.z3_ctx, [bw](auto &ctx){ return ctx.bv_sort(bw); }));
  TRY_SET_CVC5_SORT(fupdate(sctx.cvc5_ctx, [bw](auto &ctx){
    return ctx.mkBitVectorSort(bw);
  }));

  return s;
}

Sort Sort::boolSort() {
  Sort s;
  TRY_SET_Z3_SORT(fupdate(sctx.z3_ctx, [](auto &ctx){ return ctx.bool_sort(); }));
  TRY_SET_CVC5_SORT(fupdate(sctx.cvc5_ctx, [](auto &ctx){
    return ctx.getBooleanSort();
  }));

  return s;
}

Sort Sort::arraySort(const Sort &domain, const Sort &range) {
  Sort s;
  TRY_SET_Z3_SORT(fupdate(sctx.z3_ctx, [domain, range](auto &ctx){ 
    return ctx.array_sort(*domain.z3_sort, *range.z3_sort); 
  }));
  TRY_SET_CVC5_SORT(fupdate(sctx.cvc5_ctx, [domain, range](auto &ctx){
    return ctx.mkArraySort(*range.cvc5_sort, *domain.cvc5_sort);
  }));

  return s;
}

#ifdef SOLVER_Z3
Result::Result(const optional<z3::check_result> &z3_result) {
  auto unwrapped_z3_result = z3_result.value_or(z3::check_result::unknown);
  if (unwrapped_z3_result == z3::check_result::sat) {
    this->result = Result::SAT;
  } else if (unwrapped_z3_result == z3::check_result::unsat) {
    this->result = Result::UNSAT;
  } else {
    this->result = Result::UNKNOWN;
  }
}
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
Result::Result(const optional<cvc5::api::Result> &cvc5_result) {
  this->result = Result::UNKNOWN; // None also becomes unknown
  if (cvc5_result.has_value()) {
    if (cvc5_result->isSat()) {
      this->result = Result::SAT;
    } else if (cvc5_result->isUnsat()) {
      this->result = Result::UNSAT;
    }
  }
}
#endif // SOLVER_CVC5

const bool Result::operator==(const Result &rhs) {
  return this->result == rhs.result;
}

Result Result::evaluateResults(const vector<Result> &results) {
  return accumulate(results.cbegin(), results.cend(), Result(),
    [](auto acc, const auto result) {
      acc.result = max(acc.result, result.result); return acc;
    }
  );
}

Solver::Solver() {
#ifdef SOLVER_Z3
  this->z3_solver = fupdate(sctx.z3_ctx, [](auto &ctx) {
    return z3::solver(ctx);
  });
#endif // SOLVER_Z3

  IF_CVC5_ENABLED(this->cvc5_solver.emplace());
}

void Solver::add(const Expr &e) {
#ifdef SOLVER_Z3
  fupdate(this->z3_solver, [e](auto &solver) {
    solver.add(*e.z3_expr); return 0;
  });
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  fupdate(this->cvc5_solver, [e](auto &solver) {
    solver.assertFormula(*e.cvc5_expr); return 0;
  });
#endif // SOLVER_CVC5
}

void Solver::reset() {
#ifdef SOLVER_Z3
  fupdate(this->z3_solver, [](auto &solver) {
    solver.reset(); return 0;
  });
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  fupdate(this->cvc5_solver, [](auto &solver) {
    solver.resetAssertions(); return 0;
  });
#endif // SOLVER_CVC5
}

Result Solver::check() {
  vector<Result> solver_results;

#ifdef SOLVER_Z3
  solver_results.push_back(
    Result(fupdate(this->z3_solver, [](auto &solver) {
      return solver.check();
    }))
  );
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  solver_results.push_back(
    Result(fupdate(this->cvc5_solver, [](auto &solver) {
      return solver.checkSat();
    }))
  );
#endif // SOLVER_CVC5
  
  // TODO: concurrent run with solvers and return the fastest one?
  return Result::evaluateResults(solver_results);
}

IF_Z3_ENABLED(void useZ3() { sctx.useZ3(); })
IF_CVC5_ENABLED(void useCVC5() { sctx.useCVC5(); })
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
