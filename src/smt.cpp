#include "value.h"
#include "smt.h"
#include "smtmatchers.h"
#include "utils.h"

#ifdef SOLVER_Z3
#define SET_Z3(e, v) (e).setZ3(v)
#else
#define SET_Z3(e, v)
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
#define SET_CVC5(e, v) (e).setCVC5(v)
#else
#define SET_CVC5(e, v)
#endif // SOLVER_CVC5

using namespace std;


namespace {
#ifdef SOLVER_Z3
z3::expr_vector toZ3ExprVector(const vector<smt::Expr> &vec);
z3::sort_vector toZ3SortVector(const vector<smt::Sort> &vec);
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
vector<cvc5::Term> toCVC5TermVector(const vector<smt::Expr> &vec);
vector<cvc5::Sort> toCVC5SortVector(const vector<smt::Sort> &vec);
#endif // SOLVER_CVC5


template<class T>
void writeOrCheck(optional<T> &org, T &&t) {
  if (org)
    smart_assert(*org == t, "org must be empty");
  else
    org.emplace(std::move(t));
}

uint64_t to_uint64(string &&str) {
  istringstream ss(str);
  uint64_t tmp;
  ss >> tmp;
  return tmp;
}

[[maybe_unused]]
int64_t to_int64(string &&str) {
  uint64_t i = to_uint64(std::move(str));
  // Don't do (int64_t)i; it may raise UB
  union {
    uint64_t x;
    int64_t y;
  } u;
  u.x = i;
  return u.y;
}

}

namespace smt {
class Context: public Object<T_Z3(z3::context), T_CVC5(cvc5::Solver)> {
private:
  uint64_t fresh_var_counter;
#ifdef SOLVER_CVC5
  map<string, cvc5::Term, less<>> cvc5_term_cache;
#endif // SOLVER_CVC5

public:
  uint64_t timeout_ms;

  Context() {
    fresh_var_counter = 0;
    timeout_ms = 10000;
  }

#ifdef SOLVER_Z3
  void useZ3() {
    this->z3.emplace();
    this->z3->set("timeout", (int)timeout_ms);
  }
#endif
#ifdef SOLVER_CVC5
  void useCVC5() {
    this->cvc5.emplace();
    // TODO: Conditionally use HO_AUFBV
    this->cvc5->setLogic("HO_ALL");
    this->cvc5->setOption("tlimit-per", to_string(timeout_ms));
    this->cvc5->setOption("produce-models", "true");
  }

  optional<cvc5::Term> getNamedTerm(string_view name) {
    auto term_iter = cvc5_term_cache.find(name);
    if (term_iter == cvc5_term_cache.end()) {
      return nullopt;
    }
    return term_iter->second;
  }

  void addNamedTerm(const string &name, cvc5::Term &&term) {
    cvc5_term_cache.insert({name, std::move(term)});
  }

  void clearCachedTerms() {
    cvc5_term_cache.clear();
  }
#endif // SOLVER_CVC5

  string getFreshName(string prefix) {
    return prefix.append("#" + to_string(fresh_var_counter++));
  }
};

Context sctx;

void releaseResources() {
#ifdef SOLVER_Z3
  sctx.z3.reset();
#endif
#ifdef SOLVER_CVC5
  sctx.cvc5.reset();
#endif
}

vector<Expr> from1DIdx(
    Expr idx1d,
    const vector<Expr> &dims) {
  assert(dims.size() > 0);
  vector<Expr> idxs;

  // Start from the lowest dimension
  for (size_t ii = dims.size(); ii > 0; --ii) {
    size_t i = ii - 1;
    idxs.emplace_back(i == 0 ? idx1d : idx1d.urem(dims[i]));
    idx1d = idx1d.udiv(dims[i]);
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
    v.push_back(e.simplify());
  return v;
}

vector<Expr> makeCube(Expr &&e, unsigned rank) {
  vector<Expr> vec(1, Index(1));
  for (unsigned i = 1; i < rank; ++i)
    vec.push_back(Index(1));
  return vec;
}

Expr to1DIdx(
    const vector<Expr> &idxs,
    const vector<Expr> &dims) {
  // to handle `tensor.extract %t[] : tensor<f32>` case.
  if (idxs.size() == 0)
    return Index::zero();

  assert(idxs.size() == dims.size());
  auto idx = idxs[0];

  for (size_t i = 1; i < idxs.size(); ++i) {
    idx = idx * dims[i] + idxs[i];
  }
  idx = idx.simplify();
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

Expr listsEqual(const vector<Expr> &v1, const vector<Expr> &v2) {
  assert(v1.size() == v2.size());

  Expr c = Expr::mkBool(true);
  for (unsigned i = 0;  i < v1.size(); ++i) {
    c &= v1[i] == v2[i];
  }
  return c;
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
  // FIXME: consider CVC5 as well
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


// ------- FnDecl -------

FnDecl::FnDecl(const Sort &domain, const Sort &range, string &&name):
  FnDecl(vector<Sort>({domain}), range, std::move(name)) {}

FnDecl::FnDecl(
    const vector<Sort> &domain,
    const Sort &range,
    string &&name): range(range) {
  IF_Z3_ENABLED(if (range.z3) {
    z3 = sctx.z3->function(name.c_str(), toZ3SortVector(domain), *range.z3);
  });
  IF_CVC5_ENABLED(if (range.cvc5) {
    cvc5 = sctx.cvc5->declareFun(name, toCVC5SortVector(domain), *range.cvc5);
  });
}

Expr FnDecl::apply(const std::vector<Expr> &args) const {
  Expr e;
  SET_Z3(e, fmap(z3, [&args](auto &s) { return s(toZ3ExprVector(args)); }));
  SET_CVC5(e, fupdate2(sctx.cvc5, cvc5, [&args](auto &solver, auto fdecl) {
    auto args_cvc5 = toCVC5TermVector(args);
    args_cvc5.insert(args_cvc5.begin(), fdecl);
    return solver.mkTerm(cvc5::Kind::APPLY_UF, args_cvc5);
  }));
  return e;
}

Expr FnDecl::apply(const Expr &arg) const {
  return apply(vector{arg});
}

Sort FnDecl::getRange() const {
  return range;
}


// ------- Expr -------

#ifdef SOLVER_Z3
z3::expr Expr::getZ3Expr() const {
  return *z3;
}

bool Expr::hasZ3Expr() const {
  return (bool)z3;
}
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
cvc5::Term Expr::getCVC5Term() const {
  return *cvc5;
}

bool Expr::hasCVC5Term() const {
  return (bool)cvc5;
}

bool Expr::isConstantCVC5Term() const {
  return cvc5 && (cvc5->isIntegerValue() || cvc5->isBitVectorValue()
      || cvc5->isBooleanValue() || cvc5->isFloatingPointValue());
}
#endif // SOLVER_CVC5

void Expr::lockOps() {
  isOpLocked = true;
}

void Expr::unlockOps() {
  isOpLocked = false;
}

Expr Expr::simplify() const {
  Expr e;
  SET_Z3(e, fmap(this->z3, [](auto e) { return e.simplify(); }));
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5, [](auto &ctx, auto e) {
    return ctx.simplify(e);
  }));
  return e;
}

Sort Expr::sort() const {
  Sort s;
  SET_Z3(s, fmap(z3, [](auto e) { return e.get_sort(); }));
  SET_CVC5(s, fmap(cvc5, [](auto e) { return e.getSort(); }));
  return s;
}

unsigned Expr::bitwidth() const {
  return sort().bitwidth();
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
  optional<uint64_t> res;

#ifdef SOLVER_Z3
  {
    uint64_t tmp;
    if (this->z3 && this->z3->is_numeral_u64(tmp))
      writeOrCheck(res, std::move(tmp));
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (this->cvc5 && this->cvc5->isUInt64Value())
    writeOrCheck(res, this->cvc5->getUInt64Value());
  else if (this->cvc5 && this->cvc5->isBitVectorValue())
    writeOrCheck(res, to_uint64(this->cvc5->getBitVectorValue(10)));

#endif // SOLVER_CVC5

  if (res)
    v = *res;
  return res.has_value();
}

bool Expr::isInt(int64_t &v) const {
  optional<int64_t> res;

#ifdef SOLVER_Z3
  {
    int64_t tmp;
    if (this->z3 && this->z3->is_numeral_i64(tmp))
      writeOrCheck(res, std::move(tmp));
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (this->cvc5 && this->cvc5->isInt64Value())
    writeOrCheck(res, this->cvc5->getInt64Value());
  else if (this->cvc5 && this->cvc5->isBitVectorValue())
    writeOrCheck(res, to_int64(this->cvc5->getBitVectorValue(10)));

#endif // SOLVER_CVC5

  if (res)
    v = *res;
  return res.has_value();
}

optional<uint64_t> Expr::asUInt() const {
  uint64_t i;
  if (isUInt(i))
    return i;
  return {};
}

bool Expr::isNumeral() const {
  bool res = false;
  IF_Z3_ENABLED(res |= z3 && z3->is_numeral());
  IF_CVC5_ENABLED(res |= isConstantCVC5Term());
  return res;
}

bool Expr::isFalse() const {
  bool res = false;
  IF_Z3_ENABLED(res |= z3 && z3->is_false());
  IF_CVC5_ENABLED(
      res |= cvc5 && cvc5->isBooleanValue() && !cvc5->getBooleanValue());
  return res;
}

bool Expr::isTrue() const {
  bool res = false;
  IF_Z3_ENABLED(res |= z3 && z3->is_true());
  IF_CVC5_ENABLED(
      res |= cvc5 && cvc5->isBooleanValue() && cvc5->getBooleanValue());
  return res;
}

bool Expr::isVar() const {
  bool res = false;
  IF_Z3_ENABLED(
    if (z3) res |= z3->is_app() && z3->is_const() && !z3->is_numeral()
  );
  IF_CVC5_ENABLED(if (cvc5) {
    res |= cvc5->getKind() == cvc5::Kind::VARIABLE ||
           cvc5->getKind() == cvc5::Kind::CONSTANT;
  });
  return res;
}

bool Expr::hasQuantifier() const {
#ifdef SOLVER_Z3
  if (z3) {
    auto e = getZ3Expr();
    if (e.is_forall() || e.is_exists()) return true;
    if (!e.is_app()) return false;

    Z3_app a = e;
    for (unsigned i = 0; i < Z3_get_app_num_args(*sctx.z3, a); i++) {
      Expr newe = Expr();
      newe.setZ3(z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, i)));
      if (newe.hasQuantifier())
        return true;
    }
    return false;
  }
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  if(cvc5) {
    auto e = getCVC5Term();
    if (e.getKind() == cvc5::Kind::FORALL || e.getKind() == cvc5::Kind::EXISTS)
      return true;

    for (unsigned i = 0; i < e.getNumChildren(); i++) {
      Expr newe = Expr();
      newe.setCVC5(std::move(e[i]));
      if (newe.hasQuantifier())
        return true;
    }
    return false;
  }
#endif // SOLVER_CVC5
  return false;
}

string Expr::getVarName() const {
  assert(isVar());
  // TODO: CVC5
  string name;
  IF_Z3_ENABLED(name = z3->decl().name().str());
  return name;
}

#define EXPR_BVOP_UINT64(NAME) \
Expr Expr:: NAME (uint64_t arg) const {\
  return NAME(mkBV(arg, sort().bitwidth())); \
}
#define SET_Z3_USEOP(e, rhs, op) \
  SET_Z3(e, fmap(this->z3, [&rhs](auto e2) { \
    return z3::op(e2, *rhs.z3); \
  }))
#define SET_Z3_USEOP_CONST(e, rhs_const, op) \
  SET_Z3(e, fmap(this->z3, [&rhs_const](auto e2) { \
    return z3::op(e2, rhs_const); \
  }))

#define SET_CVC5_USEOP(e, rhs, op) \
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5, [&rhs](auto &solver, auto e2) { \
    return solver.mkTerm(cvc5::Kind::op, {e2, *rhs.cvc5}); \
  }))

#define CHECK_LOCK() assert(!isOpLocked)
#define CHECK_LOCK_OTHER(other) assert(!other.isOpLocked)
#define CHECK_LOCK2(rhs) assert(!isOpLocked && !rhs.isOpLocked)

EXPR_BVOP_UINT64(urem)
Expr Expr::urem(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t rhsval;
  if (rhs.isUInt(rhsval)) {
    if (rhsval == 1)
      return Expr::mkBV(0, rhs);
    else if (rhsval > 0 && (rhsval & (rhsval - 1)) == 0) {
      uint64_t l = log2_ceil(rhsval);
      return Expr::mkBV(0, bitwidth() - l).concat(extract(l - 1, 0));
    }
  }

  uint64_t a, b;
  // If divisor is zero, follow the solver's behavior
  // (see also: rewriter.hi_div0 in Z3)
  if (isUInt(a) && rhs.isUInt(b) && b != 0)
    return mkBV(a % b, rhs.bitwidth());

  Expr e;
  SET_Z3_USEOP(e, rhs, urem);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_UREM);
  return e;
}

EXPR_BVOP_UINT64(udiv)
Expr Expr::udiv(const Expr& rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t rhsval;
  if (rhs.isUInt(rhsval)) {
    if (rhsval == 1)
      return *this;
    else if (rhsval > 0 && (rhsval & (rhsval - 1)) == 0) {
      uint64_t l = log2_ceil(rhsval);
      return Expr::mkBV(0, l).concat(extract(bitwidth() - 1, l));
    }
  }

  uint64_t a, b;
  // If divisor is zero, follow the solver's behavior
  // (see also: rewriter.hi_div0 in Z3)
  if (isUInt(a) && rhs.isUInt(b) && b != 0)
    return mkBV(a / b, rhs.bitwidth());

  Expr e;
  SET_Z3_USEOP(e, rhs, udiv);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_UDIV);
  return e;
}

EXPR_BVOP_UINT64(ult)
Expr Expr::ult(const Expr& rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBool(a < b);

  {
    using namespace matchers;
    optional<Expr> dummy, divisor;
    uint64_t a, b;
    // (bvurem _, d) < d -> true
    if (URem(Any(dummy), Any(divisor)).match(*this)) {
      if (divisor->isUInt(a) && rhs.isUInt(b) && a <= b)
        return mkBool(true);
    }

    optional<Expr> llhs, lrhs, rlhs, rrhs;
    bool lhsConcatMatch = Concat(Any(llhs), Any(lrhs)).match(*this);
    bool rhsConcatMatch = Concat(Any(rlhs), Any(rrhs)).match(rhs);
    // [llhs, lrhs] < [llhs, rrhs] -> lrhs < rrhs
    if (lhsConcatMatch && rhsConcatMatch && llhs->isIdentical(*rlhs))
      return lrhs->ult(*rrhs);
    else if (lhsConcatMatch && llhs->isUInt(a) && rhs.isUInt(b) &&
             (b >> lrhs->bitwidth()) == a)
      return lrhs->ult(b ^ (a << lrhs->bitwidth()));
    else if (rhsConcatMatch && this->isUInt(a) && rlhs->isUInt(b) &&
             (a >> rrhs->bitwidth()) == b)
      return rrhs->ugt(a ^ (b << rrhs->bitwidth()));
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, ult);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_ULT);
  return e;
}

EXPR_BVOP_UINT64(slt)
Expr Expr::slt(const Expr& rhs) const {
  CHECK_LOCK2(rhs);

  int64_t a, b;
  if (isInt(a) && rhs.isInt(b))
    return mkBool(a < b);

  Expr e;
  SET_Z3_USEOP(e, rhs, slt);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_SLT);
  return e;
}

EXPR_BVOP_UINT64(ule)
Expr Expr::ule(const Expr& rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBool(a <= b);
  else if (isUInt(a) && a == 0)
    return Expr::mkBool(true);

  Expr e;
  SET_Z3_USEOP(e, rhs, ule);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_ULE);
  return e;
}

EXPR_BVOP_UINT64(sle)
Expr Expr::sle(const Expr& rhs) const {
  CHECK_LOCK2(rhs);

  int64_t a, b;
  if (isInt(a) && rhs.isInt(b))
    return mkBool(a <= b);

  Expr e;
  SET_Z3_USEOP(e, rhs, sle);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_SLE);
  return e;
}

EXPR_BVOP_UINT64(ugt)
Expr Expr::ugt(const Expr& rhs) const {
  return rhs.ult(*this);
}

EXPR_BVOP_UINT64(sgt)
Expr Expr::sgt(const Expr& rhs) const {
  return rhs.slt(*this);
}

EXPR_BVOP_UINT64(uge)
Expr Expr::uge(const Expr& rhs) const {
  return rhs.ule(*this);
}

EXPR_BVOP_UINT64(sge)
Expr Expr::sge(const Expr& rhs) const {
  return rhs.sle(*this);
}

Expr Expr::isNaN() const {
  Expr e;
  SET_Z3(e, fmap(this->z3, [](auto &z3){
    return z3.mk_is_nan();
  }));
  return e;
}

Expr Expr::select(const Expr &idx) const {
  return select(vector{idx});
}

Expr Expr::select(const vector<Expr> &idxs) const {
  // Check whether the result can be simplified
  optional<Expr> elem;
  using namespace matchers;
  if (ConstSplatArray(Any(elem)).match(*this)) {
    return *elem;
  }

  Expr e;

  SET_Z3(e, fmap(this->z3, [&idxs](auto e) {
    return z3::select(e, toZ3ExprVector(idxs)); 
  }));
#ifdef SOLVER_Z3
  if (hasZ3Expr()) {
    if (Lambda(Any(elem)).match((*this)) && idxs.size() == 1) {
      e.setZ3((elem->substituteDeBruijn({idxs[0]})).getZ3Expr());
    }
  }
#endif // SOLVER_Z3

  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5,
      [&idxs](auto &solver, auto e) {
    if (e.getSort().isArray()) {
      assert(idxs.size() == 1);
      return solver.mkTerm(cvc5::Kind::SELECT, {e, *idxs[0].cvc5});
    } else {
      auto v = toCVC5TermVector(idxs);
      v.insert(v.begin(), e);
      return solver.mkTerm(cvc5::Kind::APPLY_UF, {v});
    }
  }));
#ifdef SOLVER_CVC5
  if (hasCVC5Term()) {
    optional<Expr> idx;
    if (Lambda(Any(elem), Any(idx)).match(*this) && idxs.size() == 1) {
      e.setCVC5(elem->substitute({*idx}, idxs).getCVC5Term());
    }
  }
#endif // SOLVER_CVC5

  return e;
}

#ifdef SOLVER_CVC5
static cvc5::Term mkCVC5Lambda(
    const cvc5::Term &var, const cvc5::Term &body) {
  auto vlist = sctx.cvc5->mkTerm(cvc5::Kind::VARIABLE_LIST, {var});
  return sctx.cvc5->mkTerm(cvc5::Kind::LAMBDA, {vlist, body});
}

// Convert arr to lambda idx, arr idx
static cvc5::Term toCVC5Lambda(const cvc5::Term &arr) {
  auto idx = sctx.cvc5->mkVar(arr.getSort().getArrayIndexSort());
  return mkCVC5Lambda(idx, sctx.cvc5->mkTerm(cvc5::Kind::SELECT, {arr, idx}));
}
#endif // SOLVER_CVC5

Expr Expr::store(const Expr &idx, const Expr &val) const {
  Expr e;
  SET_Z3(e, fmap(this->z3, [&idx, &val](auto e) {
    return z3::store(e, *idx.z3, *val.z3);
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5,
      [&idx, &val](auto &solver, auto e) { // e: array or lambda
    if (e.getSort().isArray())
      return solver.mkTerm(cvc5::Kind::STORE, {e, *idx.cvc5, *val.cvc5});
    else {
      auto dummy_var = solver.mkVar(idx.cvc5->getSort());
      auto oldval = solver.mkTerm(cvc5::Kind::APPLY_UF, {e, dummy_var});
      auto lambda_body =
          solver.mkTerm(cvc5::Kind::ITE,
            {solver.mkTerm(cvc5::Kind::EQUAL, {dummy_var, *idx.cvc5}),
             *val.cvc5, oldval});
      return mkCVC5Lambda(dummy_var, lambda_body);
    }
  }));
  return e;
}

Expr Expr::store(uint64_t idx, const Expr &val) const {
  return store(mkBV(idx, sort().getArrayDomain().bitwidth()), val);
}

Expr Expr::insert(const Expr &elem) const {
  CHECK_LOCK2(elem);

  Expr e;
  // Z3 doesn't support multisets. We encode it using a const array.
  SET_Z3(e, fmap(z3, [&](auto arrayz3) {
    auto idx = *elem.z3;
    return z3::store(arrayz3, idx, z3::select(arrayz3, idx) + 1);
  }));
  SET_CVC5(e, fupdate(sctx.cvc5, [&](auto &solver) {
    auto newBag = solver.mkTerm(cvc5::Kind::BAG_MAKE, {*elem.cvc5, solver.mkInteger(1)});
    return solver.mkTerm(cvc5::Kind::BAG_UNION_DISJOINT, {*cvc5, newBag});
  }));
  return e;
}

Expr Expr::bagUnion(const Expr &other) const {
  CHECK_LOCK2(other);
    Expr e;
  // Z3 doesn't support multisets. We encode it using a const array.
  SET_Z3(e, fupdate2(sctx.z3, z3, [&](auto &ctx, auto &arrayz3) {
    auto domain = arrayz3.get_sort().array_domain();
    auto idx = ctx.constant("idx", domain);
    auto lhs = z3::select(arrayz3, idx);
    auto rhs = z3::select(*other.z3, idx);
    return z3::lambda(idx, lhs + rhs);
  }));
  SET_CVC5(e, fupdate(sctx.cvc5, [&](auto &solver) {
    return solver.mkTerm(cvc5::Kind::BAG_UNION_DISJOINT, {*cvc5, *other.cvc5});
  }));
  return e;
}



Expr Expr::getMSB() const {
  auto bw = sort().bitwidth() - 1;
  return extract(bw, bw);
}

Expr Expr::abs() const {
  CHECK_LOCK();
  if (sort().isBV()) {
    return Expr::mkBV(0, 1).concat(extract(bitwidth() - 2, 0));
  }
  Expr e;
  SET_Z3(e, fmap(this->z3, [&](auto e) { return z3::abs(e); }));
  return e;
}

Expr Expr::extract(unsigned hbit, unsigned lbit) const {
  CHECK_LOCK();

  uint64_t u;
  if (isUInt(u) && hbit < 64) {
    u = u >> lbit;
    if (hbit != 63)
      u = u & ((1ull << (uint64_t)(hbit + 1)) - 1);
    return Expr::mkBV(u, hbit - lbit + 1);
  }

  using namespace matchers;
  optional<Expr> lhs, rhs;
  if (Concat(Any(lhs), Any(rhs)).match(*this)) {
    if (lbit == 0 && hbit == rhs->bitwidth() - 1)
      return *rhs;
    else if (lbit == rhs->bitwidth() && hbit == bitwidth() - 1)
      return *lhs;
  } else if (lbit == 0 && ZeroExt(Any(lhs)).match(*this) &&
             hbit == lhs->bitwidth() - 1) {
    return *lhs;
  }

  Expr e;
  SET_Z3(e, fmap(this->z3, [&hbit, &lbit](auto e) {
    return e.extract(hbit, lbit); 
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5,
      [hbit, lbit](auto &solver, auto e) {
    return solver.mkTerm(
        solver.mkOp(cvc5::Kind::BITVECTOR_EXTRACT, {hbit, lbit}), {e});
  }));
  return e;
}

Expr Expr::concat(const Expr &lowbits) const {
  CHECK_LOCK();

  Expr e;
  SET_Z3_USEOP(e, lowbits, concat);
  SET_CVC5_USEOP(e, lowbits, BITVECTOR_CONCAT);
  return e;
}

Expr Expr::zext(unsigned bits) const {
  CHECK_LOCK();

  uint64_t i;
  if (isUInt(i))
    return mkBV(i, bitwidth() + bits);

  Expr e;
  SET_Z3_USEOP_CONST(e, bits, zext);
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5,
      [&bits](auto &solver, auto e) {
    return solver.mkTerm(
      solver.mkOp(cvc5::Kind::BITVECTOR_ZERO_EXTEND, {bits}), {e});
  }));
  return e;
}

Expr Expr::sext(unsigned bits) const {
  CHECK_LOCK();

  Expr e;
  SET_Z3_USEOP_CONST(e, bits, sext);
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5,
      [&bits](auto &solver, auto e) {
    return solver.mkTerm(
      solver.mkOp(cvc5::Kind::BITVECTOR_SIGN_EXTEND, {bits}), {e});
  }));
  return e;
}

Expr Expr::trunc(unsigned bits) const {
  smart_assert(bits < bitwidth(), "bits must be smaller than bitwidth, but "
      "got " << bits << " >= " << bitwidth());
  return extract(bitwidth() - bits - 1, 0);
}

Expr Expr::implies(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  if (rhs.isTrue())
    return rhs;
  else if (rhs.isFalse())
    return this->operator!();
  else if (this->isTrue())
    return rhs;
  else if (this->isFalse())
    return Expr::mkBool(true);

  Expr e;
  SET_Z3_USEOP(e, rhs, implies);
  SET_CVC5_USEOP(e, rhs, IMPLIES);
  return e;
}

Expr Expr::isZero() const {
  if (sort().isFPASort()) {
    Expr e;
    SET_Z3(e, fmap(this->z3, [](auto &z3){
      return z3.mk_is_zero();
    }));
    return e;
  } else {
    return *this == Expr::mkBV(0, sort().bitwidth());
  }
}

Expr Expr::isNonZero() const {
  return !isZero();
}

Expr Expr::toOneBitBV() const {
  assert(sort().isBool());
  return mkIte(*this, Expr::mkBV(1, 1), Expr::mkBV(0, 1));
}


EXPR_BVOP_UINT64(operator+)
Expr Expr::operator+(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBV(a + b, rhs.bitwidth());
  else if (isUInt(a) && a == 0)
    return rhs;
  else if (rhs.isUInt(b) && b == 0)
    return *this;

  using namespace matchers;
  {
    optional<Expr> a, b, b2, a2, b3;
    // ((a / b) * b) + (a % b) -> a
    if ((Mul(UDiv(Any(a), Any(b)), Any(b2)).match(*this) ||
         Mul(Any(b2), UDiv(Any(a), Any(b))).match(*this)) &&
         b->isIdentical(*b2) && b->isNonZero().isTrue()) {
      // check (a % b)
      if (URem(Any(a2), Any(b3)).match(rhs) &&
          a->isIdentical(*a2) && b->isIdentical(*b3)) {
        return *a;
      }
    }
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, operator+);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_ADD);
  return e;
}

EXPR_BVOP_UINT64(operator-)
Expr Expr::operator-(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBV(a - b, rhs.bitwidth());
  else if (rhs.isUInt(b) && b == 0)
    return *this;

  Expr e;
  SET_Z3_USEOP(e, rhs, operator-);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_SUB);
  return e;
}

EXPR_BVOP_UINT64(operator*)
Expr Expr::operator*(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBV(a * b, rhs.bitwidth());
  else if (rhs.isUInt(b)) {
    if (b == 1)
      return *this;
    else if (b == 0)
      return rhs;
  } else if (isUInt(a)) {
    if (a == 1)
      return rhs;
    else if (a == 0)
      return *this;
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, operator*);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_MULT);
  return e;
}

Expr Expr::operator/(const Expr &rhs) const {
  CHECK_LOCK2(rhs);
  Expr e;
  SET_Z3_USEOP(e, rhs, operator/);
  return e;
}

EXPR_BVOP_UINT64(operator^)
Expr Expr::operator^(const Expr &rhs) const {
  Expr e;
  SET_Z3_USEOP(e, rhs, operator^);
  if (sort().isBV()) {
    SET_CVC5_USEOP(e, rhs, BITVECTOR_XOR);
  } else {
    SET_CVC5_USEOP(e, rhs, XOR);
  }
  return e;
}

Expr Expr::operator&(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  if (rhs.isFalse() || isTrue())
    return rhs;
  else if (rhs.isTrue() || isFalse())
    return *this;

  Expr e;
  SET_Z3_USEOP(e, rhs, operator&);
  if (sort().isBV()) {
    SET_CVC5_USEOP(e, rhs, BITVECTOR_AND);
  } else {
    SET_CVC5_USEOP(e, rhs, AND);
  }
  return e;
}

Expr Expr::operator&(bool rhs) const {
  CHECK_LOCK();

  if (!rhs)
    return mkBool(false);
  return *this;
}

Expr Expr::operator|(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  if (rhs.isTrue() || isFalse())
    return rhs;
  else if (rhs.isFalse() || isTrue())
    return *this;

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBV(a | b, sort().bitwidth());

  Expr e;
  SET_Z3_USEOP(e, rhs, operator|);
  if (sort().isBV()) {
    SET_CVC5_USEOP(e, rhs, BITVECTOR_OR);
  } else {
    SET_CVC5_USEOP(e, rhs, OR);
  }
  return e;
}

Expr Expr::operator|(bool rhs) const {
  CHECK_LOCK();

  if (rhs)
    return mkBool(true);
  return *this;
}

EXPR_BVOP_UINT64(operator==)
Expr Expr::operator==(const Expr &rhs) const {
  if (isIdentical(rhs, true))
    return mkBool(true);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b))
    return mkBool(a == b);

  {
    using namespace matchers;
    optional<Expr> lhsh, lhsl;
    if (Concat(Any(lhsh), Any(lhsl)).match(*this)) {
      optional<Expr> rhsh, rhsl;
      if (Concat(Any(rhsh), Any(rhsl)).match(rhs) &&
          lhsl->bitwidth() == rhsl->bitwidth()) {
        uint64_t lhsl_const, rhsl_const;
        // [lhsh, lhsl_const] == [rhsh, rhsl_const]
        if (lhsl->isUInt(lhsl_const) && rhsl->isUInt(rhsl_const)) {
          if (lhsl_const != rhsl_const)
            return Expr::mkBool(false);
          else
            return *lhsh == *rhsh;
        }

        uint64_t lhsh_const, rhsh_const;
        // [lhsh_const, lhsl] == [rhsh_const, rhsl]
        if (lhsh->isUInt(lhsh_const) && rhsh->isUInt(rhsh_const)) {
          if (lhsh_const != rhsh_const)
            return Expr::mkBool(false);
          else
            return *lhsl == *rhsl;
        }
      }

      uint64_t lhsh_const, rhs_const;
      // [lhsh_const, lshl] == rhs_const
      if (lhsh->isUInt(lhsh_const) && rhs.isUInt(rhs_const)) {
        if ((rhs_const >> lhsl->bitwidth()) != lhsh_const)
          return Expr::mkBool(false);
        else
          return *lhsl == (rhs_const ^ (lhsh_const << lhsl->bitwidth()));
      }
    }
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, operator==);
  SET_CVC5_USEOP(e, rhs, EQUAL);
  return e;
}

Expr Expr::operator!() const {
  CHECK_LOCK();

  if (isTrue())
    return mkBool(false);
  else if (isFalse())
    return mkBool(true);

  Expr e;
  SET_Z3(e, fmap(this->z3, [&](auto e) { return !e; }));
  SET_CVC5(e, fmap(this->cvc5, [&](auto e) { return e.notTerm(); }));
  return e;
}

Expr Expr::operator~() const {
  CHECK_LOCK();

  if (isTrue())
    return mkBool(false);
  else if (isFalse())
    return mkBool(true);

  Expr e;
  SET_Z3(e, fmap(this->z3, [&](auto e) { return ~e; }));
  SET_CVC5(e, fupdate2(sctx.cvc5, this->cvc5, [&](auto &solver, auto e2) { \
    return solver.mkTerm(cvc5::Kind::BITVECTOR_NOT, {e2}); \
  }));
  return e;
}

Expr Expr::operator-() const {
  CHECK_LOCK();
  Expr e;
  SET_Z3(e, fmap(this->z3, [&](auto e) { return -e; }));
  return e;
}

Expr &Expr::operator&=(const Expr &rhs) {
  CHECK_LOCK2(rhs);

  Expr e = *this & rhs;
  SET_Z3(*this, std::move(e.z3));
  SET_CVC5(*this, std::move(e.cvc5));
  return *this;
}

Expr &Expr::operator|=(const Expr &rhs) {
  CHECK_LOCK2(rhs);

  Expr e = *this | rhs;
  SET_Z3(*this, std::move(e.z3));
  SET_CVC5(*this, std::move(e.cvc5));
  return *this;
}

EXPR_BVOP_UINT64(shl)
Expr Expr::shl(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b) && b < 64)
    return mkBV(a << b, rhs.bitwidth());
  else if (rhs.isUInt(b)) {
    if (b == 0)
      return *this;
  } else if (isUInt(a)) {
    if (a == 0)
      return *this;
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, shl);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_SHL);
  return e;
}

EXPR_BVOP_UINT64(ashr)
Expr Expr::ashr(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  // The value of a >> b for unsigned a is the integer part of a/(2^b)
  // which is equivalent to lshr.
  // Therefore we cannot apply uint optimization for ashr.
  // Even if we use signed int, the value of >> on a signed int
  // is implementation-defined until c++17 (it is ashr since c++20),
  // so it is hard to implement a robust optimization.
  if (rhs.isUInt(b)) {
    if (b == 0)
      return *this;
  } else if (isUInt(a)) {
    if (a == 0)
      return *this;
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, ashr);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_ASHR);
  return e;
}

EXPR_BVOP_UINT64(lshr)
Expr Expr::lshr(const Expr &rhs) const {
  CHECK_LOCK2(rhs);

  uint64_t a, b;
  if (isUInt(a) && rhs.isUInt(b) && b < 64)
    return mkBV(a >> b, rhs.bitwidth());
  else if (rhs.isUInt(b)) {
    if (b == 0)
      return *this;
  } else if (isUInt(a)) {
    if (a == 0)
      return *this;
  }

  Expr e;
  SET_Z3_USEOP(e, rhs, lshr);
  SET_CVC5_USEOP(e, rhs, BITVECTOR_LSHR);
  return e;
}

Expr Expr::substitute(
    const std::vector<Expr> &vars,
    const std::vector<Expr> &values) const {
  Expr e;
#ifdef SOLVER_Z3
  e.setZ3(fmap(this->z3, [&vars, &values](auto e) {
    return e.substitute(toZ3ExprVector(vars), toZ3ExprVector(values));
  }));
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
  e.setCVC5(fmap(this->cvc5, [&vars, &values](auto e) {
    return e.substitute(toCVC5TermVector(vars), toCVC5TermVector(values));
  }));
#endif // SOLVER_CVC5

  return e;
}

#ifdef SOLVER_Z3
Expr Expr::substituteDeBruijn(const std::vector<Expr> &values) const {
  Expr e;
  e.setZ3(fmap(this->z3, [&values](z3::expr e) {
    return e.substitute(toZ3ExprVector(values));
  }));

  // CVC5 doesn't support de bruijn indexing
  return e;
}
#endif // SOLVER_Z3

bool Expr::isIdentical(const Expr &e2, bool is_or) const {
  bool res = !is_or;
  bool temp;

  IF_Z3_ENABLED(
    if (z3) {
      temp = (Z3_ast)*z3 == (Z3_ast)*e2.z3;
      res = is_or ? (res || temp) : (res && temp);
    })
  
  IF_CVC5_ENABLED(
    if (cvc5) {
      temp = cvc5->getId() == e2.cvc5->getId();
      res = is_or ? (res || temp) : (res && temp);
    })
  return res;
}

Expr Expr::mkFreshVar(const Sort &s, const std::string &prefix) {
  Expr e;
  SET_Z3(e, fupdate2(sctx.z3, s.z3, [&prefix](auto &ctx, auto &z3sort){
    return z3::expr(ctx, Z3_mk_fresh_const(ctx, prefix.data(), z3sort));
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, s.cvc5, [&prefix](auto &ctx, auto &cvc5sort){
    return ctx.mkConst(cvc5sort, sctx.getFreshName(prefix));
  }));
  return e;
}

Expr Expr::mkFreshVar(const Expr &sort_of, const std::string &prefix) {
  return mkFreshVar(sort_of.sort(), prefix);
}

Expr Expr::mkVar(const Sort &s, const std::string &name, bool boundVar) {
  Expr e;
  SET_Z3(e, fupdate2(sctx.z3, s.z3, [&name](auto &ctx, auto &sortz3){
    return ctx.constant(name.data(), sortz3);
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, s.cvc5,
      [&name, &boundVar](auto &ctx, auto &cvc5sort){
    if (!sctx.getNamedTerm(name).has_value()) {
      cvc5::Term new_var;
      if (boundVar)
        new_var = ctx.mkVar(cvc5sort, name);
      else
        new_var = ctx.mkConst(cvc5sort, name);
      sctx.addNamedTerm(name, std::move(new_var));
    }

    const auto term = *sctx.getNamedTerm(name);
    smart_assert((boundVar && term.getKind() == cvc5::Kind::VARIABLE) ||
                 (!boundVar && term.getKind() == cvc5::Kind::CONSTANT),
                 "Boundness does not match");
    assert(cvc5sort == term.getSort() &&
           "Term(s) of duplicate names are not allowed");
    return term;
  }));
  return e;
}

Expr Expr::mkVar(const Expr &sort_of, const std::string &name, bool boundVar) {
  return mkVar(sort_of.sort(), name, boundVar);
}

Expr Expr::mkBV(const uint64_t val, const size_t sz) {
  Expr e;
  SET_Z3(e, fupdate(sctx.z3, [val, sz](auto &ctx){
    return ctx.bv_val(val, sz); 
  }));
  SET_CVC5(e, fupdate(sctx.cvc5, [val, sz](auto &ctx){
    return ctx.mkBitVector(sz, val);
  }));
  return e;
}

Expr Expr::mkBV(const uint64_t val, const Expr &sort_of) {
  return mkBV(val, sort_of.bitwidth());
}

Expr Expr::mkBool(const bool val) {
  Expr e;
  SET_Z3(e, fupdate(sctx.z3, [val](auto &ctx){
    return ctx.bool_val(val); 
  }));
  SET_CVC5(e, fupdate(sctx.cvc5, [val](auto &ctx){
    return ctx.mkBoolean(val);
  }));
  return e;
}

Expr Expr::mkFpaVal(const float val) {
  Expr e;
  SET_Z3(e, fupdate(sctx.z3, [val](auto &ctx){
    return ctx.fpa_val(val);
  }));
  return e;
}

Expr Expr::mkFpaVal(const double val) {
  Expr e;
  SET_Z3(e, fupdate(sctx.z3, [val](auto &ctx){
    return ctx.fpa_val(val);
  }));
  return e;
}

Expr Expr::mkForall(const vector<Expr> &vars, const Expr &body) {
  for (auto &v: vars) {
    smart_assert(v.isVar(), "Not a variable: " << v);
  }

  uint64_t v;
  if (body.isUInt(v) || body.isTrue() || body.isFalse()) {
    // forall idx, constant == constant (because we don't have 'False' sort)
    return body;
  }

  Expr e;
  SET_Z3(e, fmap(body.z3, [&](auto &z3body){
    return z3::forall(toZ3ExprVector(vars), z3body);
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, body.cvc5, [&](auto &solver, auto cvc5body){
    auto cvc5vars = toCVC5TermVector(vars);
    auto vlist = solver.mkTerm(cvc5::Kind::VARIABLE_LIST, {cvc5vars});
    return solver.mkTerm(cvc5::Kind::FORALL, {vlist, cvc5body});
  }));
  return e;
}

Expr Expr::mkExists(const vector<Expr> &vars, const Expr &body) {
  Expr e;
  SET_Z3(e, fmap(body.z3, [&](auto &z3body){
    return z3::exists(toZ3ExprVector(vars), z3body);
  }));
  return e;
}

Expr Expr::mkLambda(const Expr &var, const Expr &body) {
  return mkLambda(vector{var}, body);
}

Expr Expr::mkLambda(const vector<Expr> &vars, const Expr &body) {
  Expr e;
  SET_Z3(e, fmap(body.z3, [&](auto &z3body){
    return z3::lambda(toZ3ExprVector(vars), z3body);
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, body.cvc5, [&](auto &solver, auto cvc5body){
    auto cvc5vars = toCVC5TermVector(vars);
    auto vlist = solver.mkTerm(cvc5::Kind::VARIABLE_LIST, cvc5vars);
    return solver.mkTerm(cvc5::Kind::LAMBDA, {vlist, cvc5body});
  }));
  return e;
}

Expr Expr::mkSplatArray(const Sort &domain, const Expr &splatElem) {
  Expr e;
  SET_Z3(e, fmap(splatElem.z3, [&](auto &e){
    return z3::const_array(*domain.z3, e);
  }));

#ifdef SOLVER_CVC5
  if (splatElem.cvc5 && sctx.cvc5) {
    auto &solver = *sctx.cvc5;
    auto &elem = *splatElem.cvc5;
    // TOOD: How to avoid this constant-ness check?
    if (elem.isIntegerValue() || elem.isFloatingPointValue() ||
        elem.isBooleanValue()) {
      e.setCVC5(solver.mkConstArray(
        solver.mkArraySort(*domain.cvc5, elem.getSort()), elem));
    } else {
      auto dummy_var = solver.mkVar(*domain.cvc5);
      e.setCVC5(mkCVC5Lambda(dummy_var, elem));
    }
  }
#endif // SOLVER_CVC5
  return e;
}

Expr Expr::mkIte(const Expr &cond, const Expr &then, const Expr &els) {
  if (cond.isTrue())
    return then;
  else if (cond.isFalse())
    return els;
  else if (then.isTrue())
    return cond | els;
  else if (els.isFalse())
    return cond & then;

  optional<Expr> lhs, rhs;
  using namespace matchers;
  if (Equals(Any(lhs), Any(rhs)).match(cond)) {
    if ((lhs->isIdentical(then) && rhs->isIdentical(els)) ||
        (lhs->isIdentical(els) && rhs->isIdentical(then)))
      // ite(x == y, x, y) -> y
      return els;
  }

  Expr e;
  SET_Z3(e, fmap(cond.z3, [&](auto &condz3){
    return z3::ite(condz3, *then.z3, *els.z3);
  }));
  SET_CVC5(e, fupdate2(sctx.cvc5, cond.cvc5, [&](auto &solver, auto condcvc) {
    auto thenSort = then.cvc5->getSort();
    auto elsSort = els.cvc5->getSort();
    auto thenval = *then.cvc5, elsval = *els.cvc5;
    if (thenSort.isFunction() && elsSort.isArray())
      elsval = toCVC5Lambda(elsval);
    else if (thenSort.isArray() && elsSort.isFunction())
      thenval = toCVC5Lambda(thenval);
    return solver.mkTerm(cvc5::Kind::ITE, {condcvc, thenval, elsval});
  }));
  return e;
}

Expr Expr::mkEmptyBag(const Sort &domain) {
  Expr e;
  // Z3 doesn't support multisets. We encode it using a const array.
  SET_Z3(e, Expr::mkSplatArray(domain, Index::zero()).z3);
  SET_CVC5(e, fupdate2(sctx.cvc5, domain.cvc5, [&](auto &solver, auto domcvc) {
    auto bag = solver.mkBagSort(domcvc);
    return solver.mkEmptyBag(bag);
  }));
  return e;
}

Expr Expr::mkAddNoOverflow(const Expr &a, const Expr &b, bool is_signed) {
  CHECK_LOCK_OTHER(a);
  CHECK_LOCK_OTHER(b);

  return is_signed ?
      ((a + b).sext(1) == a.sext(1) + b.sext(1)) :
      ((a.zext(1) + b.zext(1)).getMSB() == 0);
}

// ------- Sort -------

IF_Z3_ENABLED(z3::sort Sort::getZ3Sort() const { return *z3; })

IF_CVC5_ENABLED(cvc5::Sort Sort::getCVC5Sort() const { return *cvc5; })

Sort Sort::getArrayDomain() const {
  Sort s;
  SET_Z3(s, fmap(z3, [&](const z3::sort &sz3) {
    return sz3.array_domain();
  }));
  SET_CVC5(s, fupdate2(sctx.cvc5, cvc5, [&](auto &solver, auto cvc5sort) {
    if (cvc5sort.isFunction()) {
      auto dom = cvc5sort.getFunctionDomainSorts();
      assert(dom.size() == 1);
      return dom[0];
    }
    return cvc5sort.getArrayIndexSort();
  }));
  return s;
}

bool Sort::isFPASort() const {
  optional<bool> res;
  IF_Z3_ENABLED(if(z3) writeOrCheck(res, z3->is_fpa()));
  return *res;
}

bool Sort::isBV() const {
  optional<bool> res;
  IF_Z3_ENABLED(if(z3) writeOrCheck(res, z3->is_bv()));
  IF_CVC5_ENABLED(if(cvc5) writeOrCheck(res, cvc5->isBitVector()));
  return res && *res;
}

Sort Sort::toFnSort() const {
  Sort s;
  SET_Z3(s, fmap(z3, [](const auto &sz) { return sz; }));
  SET_CVC5(s, fupdate2(sctx.cvc5, cvc5,
      [](auto &solver, auto s) {
    if (s.isArray()) {
      return solver.mkFunctionSort(
          {s.getArrayIndexSort()}, s.getArrayElementSort());
    }
    return s;
  }));
  return s;
}

unsigned Sort::bitwidth() const {
  optional<unsigned> res;
  IF_Z3_ENABLED(if(z3) writeOrCheck(res, z3->bv_size()));
  IF_CVC5_ENABLED(if(cvc5) writeOrCheck(res, cvc5->getBitVectorSize()));
  return *res;
}

bool Sort::isArray() const {
  optional<bool> res;
  IF_Z3_ENABLED(if(z3) writeOrCheck(res, z3->is_array()));
  IF_CVC5_ENABLED(if(cvc5) writeOrCheck(res, cvc5->isArray()));
  return res && *res;
}

bool Sort::isBool() const {
  optional<bool> res;
  IF_Z3_ENABLED(if(z3) writeOrCheck(res, z3->is_bool()));
  IF_CVC5_ENABLED(if(cvc5) writeOrCheck(res, cvc5->isBoolean()));
  return res && *res;
}

Sort Sort::bvSort(size_t bw) {
  Sort s;
  SET_Z3(s, fupdate(sctx.z3, [bw](auto &ctx){ return ctx.bv_sort(bw); }));
  SET_CVC5(s, fupdate(sctx.cvc5, [bw](auto &ctx){
      return ctx.mkBitVectorSort(bw); }));
  return s;
}

Sort Sort::boolSort() {
  Sort s;
  SET_Z3(s, fupdate(sctx.z3, [](auto &ctx){ return ctx.bool_sort(); }));
  SET_CVC5(s, fupdate(sctx.cvc5, [](auto &c){ return c.getBooleanSort(); }));
  return s;
}

Sort Sort::arraySort(const Sort &domain, const Sort &range) {
  Sort s;
  SET_Z3(s, fupdate2(sctx.z3, domain.z3, [&range](auto &ctx, auto domz3){
    return ctx.array_sort(domz3, *range.z3);
  }));
  SET_CVC5(s, fupdate2(sctx.cvc5, domain.cvc5,
      [&range](auto &ctx, auto domcvc5){
        return ctx.mkArraySort(domcvc5, *range.cvc5);
    }
  ));
  return s;
}

Sort Sort::fp32IEEE754Sort() {
  Sort s;
  SET_Z3(s, fupdate(sctx.z3, [](auto &ctx){ return ctx.fpa_sort(8, 24); })); // f32
  return s;
}

Sort Sort::fp64IEEE754Sort() {
  Sort s;
  SET_Z3(s, fupdate(sctx.z3, [](auto &ctx){ return ctx.fpa_sort(11, 53); })); // f64
  return s;
}

// ------- CheckResult -------

bool CheckResult::isUnknown() const {
  return !hasSat() && !hasUnsat();
}

bool CheckResult::hasSat() const {
  bool res = false;
  IF_Z3_ENABLED(res |= z3 && (*z3 == z3::check_result::sat));
  IF_CVC5_ENABLED(res |= cvc5 && cvc5->isSat());
  return res;
}

bool CheckResult::hasUnsat() const {
  bool res = false;
  IF_Z3_ENABLED(res |= z3 && (*z3 == z3::check_result::unsat));
  IF_CVC5_ENABLED(res |= cvc5 && cvc5->isUnsat());
  return res;
}

bool CheckResult::isInconsistent() const {
  return hasSat() && hasUnsat();
}

// ------- Model -------

Expr Model::eval(const Expr &e, bool modelCompletion) const {
  Expr newe;
  SET_Z3(newe, fmap(z3, [modelCompletion, &e](auto &z3model){
    return z3model.eval(e.getZ3Expr(), modelCompletion);
  }));
  SET_CVC5(newe, fupdate2(sctx.cvc5, e.cvc5, [](auto &solver, auto ec){
    // getValue() creates a new BV, so the model gets invalidated
    // re-running checkSat() is very expensive, but this is so far
    // the only way to retrieve the values
    solver.checkSat();
    return solver.getValue(ec);
  }));

  return newe;
}

vector<Expr> Model::eval(const vector<Expr> &exprs, bool modelCompletion) const {
  vector<Expr> values;
  values.reserve(exprs.size());

#ifdef SOLVER_CVC5
  auto cvc5_values = fupdate(sctx.cvc5, [exprs](auto &solver) {
    // see the comment at Expr Model::eval(const Expr &e, bool modelCompletion)
    solver.checkSat();
    auto cvc5_exprs = toCVC5TermVector(exprs);
    return solver.getValue(cvc5_exprs);
  });
#endif // SOLVER_CVC5

  for (size_t i = 0; i < exprs.size(); ++i) {
#ifdef SOLVER_Z3
    auto &e = exprs[i];
    auto z3_value = fmap(z3, [modelCompletion, &e](auto &z3model) {
      return z3model.eval(e.getZ3Expr(), modelCompletion);
    });
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
    optional<cvc5::Term> cvc5_value;
    if (cvc5_values) {
      cvc5_value = std::move((*cvc5_values)[i]);
    }
#endif // SOLVER_CVC5

    Expr value;
    SET_Z3(value, std::move(z3_value));
    SET_CVC5(value, std::move(cvc5_value));
    values.push_back(std::move(value));
  }

  return values;
}

Model Model::empty() {
  // FIXME
  Model m;
  SET_Z3(m, {*sctx.z3});
  return m;
}

// ------- Solver -------

Solver::Solver(const char *logic) {
#ifdef SOLVER_Z3
  z3 = fupdate(sctx.z3, [logic](auto &ctx){
    return z3::solver(ctx, logic);
  });
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  // We can't create new solver since it won't be compatible with
  // the variables created by previous solver
  if (sctx.cvc5)
    sctx.cvc5->push();
#endif // SOLVER_CVC5
}

Solver::~Solver() {
#ifdef SOLVER_CVC5
  // We can't destroy solver since it will invalidate every variable
  // it has created
  if (sctx.cvc5) {
    sctx.clearCachedTerms();
    sctx.cvc5->pop();
  }
#endif // SOLVER_CVC5
}

void Solver::add(const Expr &e) {
  IF_Z3_ENABLED(fupdate(z3, [&e](auto &solver) {
    solver.add(*e.z3);
    return 0;
  }));
  IF_CVC5_ENABLED(fupdate(sctx.cvc5, [&e](auto &solver) {
    solver.assertFormula(*e.cvc5);
    return 0;
  }));
}

void Solver::reset() {
  IF_Z3_ENABLED(fupdate(z3, [](auto &solver) { solver.reset(); return 0; }));
  IF_CVC5_ENABLED(fupdate(sctx.cvc5, [](auto &solver) {
    solver.resetAssertions();
    return 0;
  }));
}

CheckResult Solver::check() {
  // TODO: concurrent run with solvers and return the fastest one?
  CheckResult cr;
  SET_Z3(cr, fupdate(z3, [](auto &solver) { return solver.check(); }));
  SET_CVC5(cr, fupdate(sctx.cvc5,
      [](auto &solver) { return solver.checkSat(); }));
  return cr;
}

Model Solver::getModel() const {
  Model m;
  SET_Z3(m, fmap(z3, [](auto &solver) { return solver.get_model(); }));
  return m;
}


void useZ3() { IF_Z3_ENABLED(sctx.useZ3()); }
void useCVC5() { IF_CVC5_ENABLED(sctx.useCVC5()); }
uint64_t getTimeout() { return sctx.timeout_ms; }
void setTimeout(const uint64_t ms) { sctx.timeout_ms = ms; }



namespace matchers {
#ifdef SOLVER_Z3
void Matcher::setZ3(Expr &e, optional<z3::expr> &&opt) const {
  e.setZ3(std::move(opt));
}

bool Matcher::matchBinaryOp(const Expr &expr, Z3_decl_kind z3Kind,
    function<bool(const Expr&)> lhsMatcher,
    function<bool(const Expr&)> rhsMatcher) const {
  if (expr.hasZ3Expr()) {
    auto e = expr.getZ3Expr();
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
    if (Z3_get_decl_kind(*sctx.z3, decl) != z3Kind)
      return false;

    Expr lhs = newExpr(), rhs = newExpr();
    setZ3(lhs, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0)));
    setZ3(rhs, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 1)));
    return lhsMatcher(lhs) && rhsMatcher(rhs);
  } else {
    return false;
  }
}
#endif // SOLVER_Z3

#ifdef SOLVER_CVC5
void Matcher::setCVC5(Expr &e, optional<cvc5::Term> &&opt) const {
  e.setCVC5(std::move(opt));
}

bool Matcher::matchBinaryOp(const Expr &expr, cvc5::Kind opKind,
    function<bool(const Expr&)> lhsMatcher,
    function<bool(const Expr&)> rhsMatcher) const {
  if (expr.hasCVC5Term()) {
    auto term = expr.getCVC5Term();
    if (term.getKind() != opKind || term.getNumChildren() != 2)
      return false;

    Expr lhs = newExpr(), rhs = newExpr();
    setCVC5(lhs, std::move(term[0]));
    setCVC5(rhs, std::move(term[1]));
    return lhsMatcher(lhs) && rhsMatcher(rhs);
  } else {
    return false;
  }
}

#endif // SOLVER_CVC5


bool ConstBool::operator()(const Expr &expr) const {
  return (val && expr.isTrue()) || (!val && expr.isFalse());
}

bool ConstSplatArray::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr()) {
    auto e = expr.getZ3Expr();
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
    if (Z3_get_decl_kind(*sctx.z3, decl) != Z3_OP_CONST_ARRAY)
      return false;

    Expr newe = newExpr();
    setZ3(newe, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0)));
    return subMatcher(newe);
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term()) {
    auto e = expr.getCVC5Term();

    if (!e.isConstArray())
      return false;

    Expr newe = newExpr();
    setCVC5(newe, e.getConstArrayBase());
    return subMatcher(newe);
  }
#endif // SOLVER_CVC5

  return false;
}

bool Lambda::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr()) {
    auto e = expr.getZ3Expr();

    if (!e.is_lambda())
      return false;

    Z3_ast body = Z3_get_quantifier_body(*sctx.z3, (Z3_ast)e);

    Expr newe = newExpr();
    setZ3(newe, z3::expr(*sctx.z3, body));

    // Z3 matches body only (because Z3 supports de bruijn indexing)
    return bodyMatcher(newe);
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term()) {
    auto e = expr.getCVC5Term();
    if (e.getKind() != cvc5::Kind::LAMBDA || e.getNumChildren() != 2)
      return false;

    auto vlist = e[0];
    if (vlist.getKind() != cvc5::Kind::VARIABLE_LIST)
      return false;

    Expr newidx = newExpr(), newe = newExpr();
    setCVC5(newidx, std::move(vlist[0]));
    setCVC5(newe, std::move(e[1]));

    return bodyMatcher(newe) && idxMatcher(newidx);
  }
#endif // SOLVER_CVC5

  return false;
}

bool Store::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr()) {
    auto e = expr.getZ3Expr();
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
    if (Z3_get_decl_kind(*sctx.z3, decl) != Z3_OP_STORE)
      return false;

    Expr arr = newExpr(), idx = newExpr(), val = newExpr();
    setZ3(arr, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0)));
    setZ3(idx, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 1)));
    setZ3(val, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 2)));

    return arrMatcher(arr) && idxMatcher(idx) && valMatcher(val);
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term()) {
    auto term = expr.getCVC5Term();

    if (term.getKind() != cvc5::Kind::STORE || term.getNumChildren() != 3)
      return false;
    
    Expr arr = newExpr(), idx = newExpr(), val = newExpr();
    setCVC5(arr, std::move(term[0]));
    // Note idx is cvc5::BOUND_VAR_LIST
    // must unwrap BOUND_VAR_LIST when using concrete index
    setCVC5(idx, std::move(term[1])); 
    setCVC5(val, std::move(term[2]));
    
    return arrMatcher(arr) && idxMatcher(idx) && valMatcher(val);
  }
#endif // SOLVER_CVC5
  return false;
}

bool Concat::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr())
    return matchBinaryOp(expr, Z3_OP_CONCAT, lhsMatcher, rhsMatcher);
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term())
    return matchBinaryOp(expr, cvc5::Kind::BITVECTOR_CONCAT, lhsMatcher,
                         rhsMatcher);
#endif // SOLVER_CVC5

  return false;
}

bool URem::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr())
    return matchBinaryOp(expr, Z3_OP_BUREM, lhsMatcher, rhsMatcher);
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term())
    return matchBinaryOp(expr, cvc5::Kind::BITVECTOR_UREM, lhsMatcher,
                         rhsMatcher);
#endif // SOLVER_CVC5

  return false;
}

bool UDiv::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr())
    return matchBinaryOp(expr, Z3_OP_BUDIV, lhsMatcher, rhsMatcher);
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term())
    return matchBinaryOp(expr, cvc5::Kind::BITVECTOR_UDIV, lhsMatcher,
                         rhsMatcher);
#endif // SOLVER_CVC5

  return false;
}

bool Mul::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr())
    return matchBinaryOp(expr, Z3_OP_BMUL, lhsMatcher, rhsMatcher);
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term())
    return matchBinaryOp(expr, cvc5::Kind::BITVECTOR_MULT, lhsMatcher,
                         rhsMatcher);
#endif // SOLVER_CVC5

  return false;
}

bool ZeroExt::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr()) {
    auto e = expr.getZ3Expr();
    if (!e.is_app())
      return false;

    Z3_app a = e;
    Z3_func_decl decl = Z3_get_app_decl(*sctx.z3, a);
    if (Z3_get_decl_kind(*sctx.z3, decl) != Z3_OP_ZERO_EXT)
      return false;

    Expr subexpr = newExpr();
    setZ3(subexpr, z3::expr(*sctx.z3, Z3_get_app_arg(*sctx.z3, a, 0)));
    return matcher(subexpr);
  }
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term()) {
    auto op = expr.getCVC5Term();
    if (op.getKind() != cvc5::Kind::BITVECTOR_ZERO_EXTEND)
      return false;
    if (op.getNumChildren() != 1)
      return false;

    Expr subexpr = newExpr();
    setCVC5(subexpr, op[0]);
    return matcher(subexpr);
  }
#endif // SOLVER_CVC5
  return false;
}

bool Equals::operator()(const Expr &expr) const {
#ifdef SOLVER_Z3
  if (expr.hasZ3Expr())
    return matchBinaryOp(expr, Z3_OP_EQ, lhsMatcher, rhsMatcher);
#endif // SOLVER_Z3
#ifdef SOLVER_CVC5
  if (expr.hasCVC5Term())
    return matchBinaryOp(expr, cvc5::Kind::EQUAL, lhsMatcher, rhsMatcher);
#endif // SOLVER_CVC5

  return false;
}

}
} // namespace smt

namespace {
#ifdef SOLVER_Z3
z3::expr_vector toZ3ExprVector(const vector<smt::Expr> &vec) {
  z3::expr_vector ev(*smt::sctx.z3);
  for (auto &e: vec)
    ev.push_back(e.getZ3Expr());
  return ev;
}

z3::sort_vector toZ3SortVector(const vector<smt::Sort> &vec) {
  z3::sort_vector ev(*smt::sctx.z3);
  for (auto &e: vec)
    ev.push_back(e.getZ3Sort());
  return ev;
}
#endif // end SOLVER_Z3

#ifdef SOLVER_CVC5
vector<cvc5::Term> toCVC5TermVector(const vector<smt::Expr> &vec) {
  vector<cvc5::Term> cvc5_terms;
  cvc5_terms.reserve(vec.size());
  for (const auto e : vec) {
    cvc5_terms.push_back(e.getCVC5Term());
  }
  return cvc5_terms;
}

vector<cvc5::Sort> toCVC5SortVector(const vector<smt::Sort> &vec) {
  vector<cvc5::Sort> cvc5_sorts;
  cvc5_sorts.reserve(vec.size());
  for (const auto e : vec) {
    cvc5_sorts.push_back(e.getCVC5Sort());
  }
  return cvc5_sorts;
}
#endif // SOLVER_CVC5
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const smt::Expr &e) {
  stringstream ss;
  ss << e;
  os << ss.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const smt::Expr &e) {
  IF_Z3_ENABLED(if (e.hasZ3Expr()) os << e.getZ3Expr());
  IF_CVC5_ENABLED(if (e.hasCVC5Term()) os << e.getCVC5Term());
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
