#include "value.h"
#include "smt.h"
#include "smtmatchers.h"
#include "utils.h"

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
z3::expr_vector toZ3ExprVector(const vector<z3::expr> &vec) {
  z3::expr_vector ev(*smt::sctx.z3);
  for (auto &e: vec)
    ev.push_back(e);
  return ev;
}

z3::expr_vector toZ3ExprVector(const vector<smt::Expr> &vec) {
  z3::expr_vector ev(*smt::sctx.z3);
  for (auto &e: vec)
    ev.push_back(e.getZ3Expr());
  return ev;
}
}

namespace smt {
class Context {
private:
  unordered_map<string, uint64_t> fresh_var_map;

public:
  IF_Z3_ENABLED(optional<z3::context> z3);
  IF_CVC5_ENABLED(optional<cvc5::api::Solver> cvc5);

  uint64_t timeout_ms;

  Context() {
    IF_Z3_ENABLED(this->z3.reset());
    IF_CVC5_ENABLED(this->cvc5.reset());
    timeout_ms = 20000;
  }

  IF_Z3_ENABLED(void useZ3() { this->z3.emplace(); })
  IF_CVC5_ENABLED(void useCVC5() { this->cvc5.emplace(); })

  string getFreshName(string prefix) {
    this->fresh_var_map.insert({prefix, 0});
    uint64_t suffix = fresh_var_map.at(prefix)++;
    return prefix.append("_" + to_string(suffix));
  }
};

Context sctx;

vector<Expr> from1DIdx(
    Expr idx1d,
    const vector<Expr> &dims) {
  assert(dims.size() > 0);
  vector<Expr> idxs;

  for (size_t ii = dims.size(); ii > 0; --ii) {
    size_t i = ii - 1;
    // TODO: migrate constant foldings & simplifications
    auto a = idx1d.urem(dims[i]), b = idx1d.udiv(dims[i]);
    idxs.emplace_back(a);
    idx1d = b;
  }

  reverse(idxs.begin(), idxs.end());
  return idxs;
}

Expr get1DSize(const vector<Expr> &dims) {
  Expr szaccml = Index::one();
  for (auto &d: dims) {
    assert(d.sort().isBV());
    szaccml = szaccml * d;
  }

  szaccml = szaccml.simplify();
  return szaccml;
}

vector<Expr> simplifyList(const vector<Expr> &exprs) {
  vector<Expr> v;
  v.reserve(exprs.size());
  for (auto &e: exprs)
    v.push_back(move(e.simplify()));
  return v;
}

Expr to1DIdx(
    const vector<Expr> &idxs,
    const vector<Expr> &dims) {
  assert(idxs.size() == dims.size());
  auto idx = idxs[0];

  for (size_t i = 1; i < idxs.size(); ++i) {
    // TODO: migrate constant foldings
    idx = idx * dims[i] + idxs[i];
  }
  return idx;
}

Expr fitsInDims(
    const vector<Expr> &idxs,
    const vector<Expr> &sizes) {
  assert(idxs.size() == sizes.size());

  Expr cond = Expr::mkBool(true);
  for (size_t i = 0; i < idxs.size(); ++i)
    cond = cond & (idxs[i].ult(sizes[i]));
  return cond;
}

FnDecl::FnDecl(const Sort &domain, const Sort &range, string &&name) {
  if (domain.z3_sort)
    z3_fdecl = sctx.z3->function(name.c_str(), *domain.z3_sort, *range.z3_sort);
}

FnDecl::FnDecl(
    const vector<Sort> &domain,
    const Sort &range,
    string &&name) {
  if (range.z3_sort) {
    z3::sort_vector v(*sctx.z3);
    for (const auto &s: domain)
      v.push_back(*s.z3_sort);
    z3_fdecl = sctx.z3->function(name.c_str(), v, *range.z3_sort);
  }
}

Expr FnDecl::apply(const std::vector<Expr> &args) const {
  // FIXME: z3 can be disabled
  return {(*z3_fdecl)(toZ3ExprVector(args))};
}

Expr FnDecl::apply(const Expr &arg) const {
  // FIXME: z3 can be disabled
  return {(*z3_fdecl)(*arg.z3_expr)};
}


string or_omit(const Expr &e) {
  string s;
  llvm::raw_string_ostream rso(s);
  Expr e2 = e.simplify();

  int64_t i;
  if (e2.isInt(i))
    return to_string(i);
  rso << e2;
  rso.flush();

  if (s.size() > 500)
    return "(omitted)";
  return s;
}

string or_omit(const vector<Expr> &evec) {
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

z3::expr Expr::getZ3Expr() const {
  return *z3_expr;
}

Expr Expr::simplify() const {
  auto z3_expr = fmap(this->z3_expr, [](auto e) { return e.simplify(); });

  return Expr(move(z3_expr));
}

Sort Expr::sort() const {
  return Sort(fmap(z3_expr, [](auto e) { return e.get_sort(); }));
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

bool Expr::isUInt(uint64_t &v) const {
  bool flag = false;
  if (this->z3_expr)
    flag = this->z3_expr->is_numeral_u64(v);
  return flag;
}

bool Expr::isInt(int64_t &v) const {
  bool flag = false;
  if (this->z3_expr)
    flag = this->z3_expr->is_numeral_i64(v);
  return flag;
}

bool Expr::isNumeral() const {
  // FIXME
  optional<bool> z3res;
  if (z3_expr)
    z3res = z3_expr->is_numeral();
  return *z3res;
}

bool Expr::isFalse() const {
  // FIXME
  optional<bool> z3res;
  if (z3_expr)
    z3res = z3_expr->is_false();
  return *z3res;
}

#define EXPR_BVOP_UINT64(NAME) \
Expr Expr:: NAME (uint64_t arg) const {\
  return NAME(mkBV(arg, sort().bitwidth())); \
}

EXPR_BVOP_UINT64(urem)
Expr Expr::urem(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::urem(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(udiv)
Expr Expr::udiv(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::udiv(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(mod)
Expr Expr::mod(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::mod(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(ult)
Expr Expr::ult(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::ult(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(ule)
Expr Expr::ule(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::ule(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(ugt)
Expr Expr::ugt(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::ugt(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(uge)
Expr Expr::uge(const Expr& rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) {
    return z3::uge(e, *rhs.z3_expr);
  });

  return Expr(move(z3_expr));
}

Expr Expr::select(const Expr &idx) const {
  auto z3_expr = fmap(this->z3_expr, [&idx](auto e) { 
    return z3::select(e, *idx.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::select(const vector<Expr> &idxs) const {
  auto z3_expr = fmap(this->z3_expr, [&idxs](auto e) { 
    return z3::select(e, toZ3ExprVector(idxs)); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::store(const Expr &idx, const Expr &val) const {
  auto z3_expr = fmap(this->z3_expr, [&idx, &val](auto e) { 
    return z3::store(e, *idx.z3_expr, *val.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::store(uint64_t idx, const Expr &val) const {
  return store(mkBV(idx, sort().getArrayDomain().bitwidth()), val);
}

Expr Expr::extract(unsigned hbit, unsigned lbit) const {
  auto z3_expr = fmap(this->z3_expr, [&hbit, &lbit](auto e) { 
    return e.extract(hbit, lbit); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::concat(const Expr &lowbits) const {
  auto z3_expr = fmap(this->z3_expr, [&](auto e) { 
    return z3::concat(e, *lowbits.z3_expr); 
  });

  return Expr(move(z3_expr));
}

Expr Expr::implies(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return z3::implies(e, *rhs.z3_expr); 
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::isNonZero() const {
  return !(*this == Expr::mkBV(0, sort().bitwidth()));
}

EXPR_BVOP_UINT64(operator+)
Expr Expr::operator+(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e + *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(operator-)
Expr Expr::operator-(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e - *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(operator*)
Expr Expr::operator*(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { return e * *rhs.z3_expr; });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator&(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    if (e.is_bool()) 
      return e && *rhs.z3_expr;
    else
      return e & *rhs.z3_expr;
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator|(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    if (e.is_bool()) 
      return e || *rhs.z3_expr;
    else
      return e | *rhs.z3_expr;
  });
  
  return Expr(move(z3_expr));
}

EXPR_BVOP_UINT64(operator==)
Expr Expr::operator==(const Expr &rhs) const {
  auto z3_expr = fmap(this->z3_expr, [&rhs](auto e) { 
    return e == *rhs.z3_expr;
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::operator!() const {
  auto z3_expr = fmap(this->z3_expr, [&](auto e) { 
    return !e;
  });
  
  return Expr(move(z3_expr));
}

Expr Expr::substitute(
    const std::vector<Expr> &vars,
    const std::vector<Expr> &values) const {
  auto z3_expr = fmap(this->z3_expr, [&vars, &values](auto e) { 
    vector<z3::expr> z3vars, z3values;
    for (auto &var: vars)
      z3vars.push_back(*var.z3_expr);
    for (auto &val: values)
      z3values.push_back(*val.z3_expr);
    return e.substitute(toZ3ExprVector(z3vars), toZ3ExprVector(z3values));
  });

  return Expr(move(z3_expr));
}

bool Expr::structurallyEq(const Expr &e2) const {
  // FIXME: z3 can be disabled
  return (Z3_ast)*z3_expr == (Z3_ast)*e2.z3_expr;
}


Expr Expr::mkFreshVar(const Sort &s, std::string_view prefix) {
  auto z3_expr = fupdate(sctx.z3, [s, prefix](auto &ctx){ 
    auto ast = Z3_mk_fresh_const(ctx, prefix.data(), *s.z3_sort);
    return z3::expr(ctx, ast);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkVar(const Sort &s, std::string_view name) {
  auto z3_expr = fupdate(sctx.z3, [s, name](auto &ctx){ 
    return ctx.constant(name.data(), *s.z3_sort);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkBV(const uint64_t val, const size_t sz) {
  auto z3_expr = fupdate(sctx.z3, [val, sz](auto &ctx){ 
    return ctx.bv_val(val, sz); 
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkBool(const bool val) {
  auto z3_expr = fupdate(sctx.z3, [val](auto &ctx){ 
    return ctx.bool_val(val); 
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkForall(const vector<Expr> &vars, const Expr &body) {
  auto z3_expr = fmap(body.z3_expr, [&](auto &z3body){ 
    return z3::forall(toZ3ExprVector(vars), z3body);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkLambda(const Expr &var, const Expr &body) {
  auto z3_expr = fmap(body.z3_expr, [&](auto &z3body){ 
    return z3::lambda(*var.z3_expr, z3body);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkLambda(const vector<Expr> &vars, const Expr &body) {
  auto z3_expr = fmap(body.z3_expr, [&](auto &z3body){ 
    return z3::lambda(toZ3ExprVector(vars), z3body);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkConstArray(const Sort &domain, const Expr &splatElem) {
  auto z3_expr = fmap(splatElem.z3_expr, [&](auto &e){ 
    return z3::const_array(*domain.z3_sort, e); 
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkIte(const Expr &cond, const Expr &then, const Expr &els) {
  auto z3_expr = fmap(cond.z3_expr, [&](auto &condz3){ 
    return z3::ite(condz3, *then.z3_expr, *els.z3_expr);
  });

  return Expr(move(z3_expr));
}

Expr Expr::mkAddNoOverflow(const Expr &a, const Expr &b, bool is_signed) {
  auto z3_expr = fmap(a.z3_expr, [&](auto &az3){ 
    return z3::bvadd_no_overflow(az3, *b.z3_expr, is_signed);
  });

  return Expr(move(z3_expr));
}

Sort::Sort(std::optional<z3::sort> &&z3_sort) {
  this->z3_sort = std::move(z3_sort);
}

Sort Sort::getArrayDomain() const {
  auto z3_sort_dom = fmap(z3_sort, [&](const z3::sort &sz3) {
    return sz3.array_domain();
  });

  return Sort(move(z3_sort_dom));
}

bool Sort::isBV() const {
  return z3_sort->is_bv();
}

unsigned Sort::bitwidth() const {
  if (z3_sort)
    // FIXME: Check whether its size is equivalent to cvc5's size
    return z3_sort->bv_size();
  assert(false && "unknown sort");
}

bool Sort::isArray() const {
  if (z3_sort)
    // FIXME: Check whether its size is equivalent to cvc5's size
    return z3_sort->is_array();
  assert(false && "unknown sort");
}

Sort Sort::bvSort(size_t bw) {
  auto z3_sort = fupdate(sctx.z3, [bw](auto &ctx){ return ctx.bv_sort(bw); });

  return Sort(move(z3_sort));
}

Sort Sort::boolSort() {
  auto z3_sort = fupdate(sctx.z3, [](auto &ctx){ return ctx.bool_sort(); });

  return Sort(move(z3_sort));
}

Sort Sort::arraySort(const Sort &domain, const Sort &range) {
  auto z3_sort = fupdate(sctx.z3, [domain, range](auto &ctx){ 
    return ctx.array_sort(*domain.z3_sort, *range.z3_sort); 
  });

  return Sort(move(z3_sort));
}

FnDecl::FnDecl(std::optional<z3::func_decl> &&z3_fdecl) {
  this->z3_fdecl = std::move(z3_fdecl);
}

CheckResult::CheckResult(const optional<z3::check_result> &z3_result) {
  auto unwrapped_z3_result = z3_result.value_or(z3::check_result::unknown);
  if (unwrapped_z3_result == z3::check_result::sat) {
    this->result = CheckResult::SAT;
  } else if (unwrapped_z3_result == z3::check_result::unsat) {
    this->result = CheckResult::UNSAT;
  } else {
    this->result = CheckResult::UNKNOWN;
  }
}

const bool CheckResult::operator==(const CheckResult &rhs) {
  return this->result == rhs.result;
}


Expr Model::eval(const Expr &e, bool modelCompletion) const {
  auto z3e = fmap(z3, [modelCompletion, &e](auto &z3model){
    return z3model.eval(e.getZ3Expr(), modelCompletion);
  });

  return Expr({move(z3e)});
}

Model Model::empty() {
  // FIXME
  return {*sctx.z3};
}


Solver::Solver(const char *logic) {
  z3 = fupdate(sctx.z3, [logic](auto &ctx){
    return z3::solver(ctx, logic);
  });
}

void Solver::add(const Expr &e) {
  fupdate(z3, [e](auto &solver) { solver.add(*e.z3_expr); return 0; });
}

void Solver::reset() {
  fupdate(z3, [](auto &solver) { solver.reset(); return 0; });
}

CheckResult Solver::check() {
  auto z3_result = fupdate(z3, [](auto &solver) { return solver.check(); });
  
  // TODO: compare with results from other solvers
  // TODO: concurrent run with solvers and return the fastest one?
  return CheckResult(z3_result);
}

Model Solver::getModel() const {
  auto z3_result = fmap(z3, [](auto &solver) { return solver.get_model(); });

  return Model(move(z3_result));
}

void useZ3() { IF_Z3_ENABLED(sctx.useZ3()); }
void useCVC5() { IF_CVC5_ENABLED(sctx.useCVC5()); }
void setTimeout(const uint64_t ms) { sctx.timeout_ms = ms; }

namespace matchers {

Expr Matcher::createExpr(optional<z3::expr> &&opt) const {
  return Expr(move(opt));
}

bool ConstSplatArray::operator()(const Expr &expr) const {
  // FIXME: cvc5
  auto e = expr.getZ3Expr();
  if (!e.is_app())
    return false;

  Z3_app a = e;
  Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
  if (Z3_get_decl_kind(*sctx.z3, decl) != Z3_OP_CONST_ARRAY)
    return false;

  z3::expr newe(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0));
  return subMatcher(createExpr(move(newe)));
}

bool Store::operator()(const Expr &expr) const {
  // FIXME: cvc5
  auto e = expr.getZ3Expr();
  if (!e.is_app())
    return false;

  Z3_app a = e;
  Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
  if (Z3_get_decl_kind(*sctx.z3, decl) != Z3_OP_STORE)
    return false;

  z3::expr arr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0));
  z3::expr idx(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 1));
  z3::expr val(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 2));

  return arrMatcher(createExpr(move(arr))) &&
      idxMatcher(createExpr(move(idx))) &&
      valMatcher(createExpr(move(val)));
}
}
} // namespace smt

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::Expr &e) {
  // FIXME
  stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const smt::Expr &e) {
  // FIXME
  os << e.getZ3Expr();
  return os;
}

llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const vector<smt::Expr> &es) {
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
