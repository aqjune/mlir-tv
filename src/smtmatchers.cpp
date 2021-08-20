#include "smtmatchers.h"

using namespace std;

namespace smt {
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
}