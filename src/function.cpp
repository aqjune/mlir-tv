#include "function.h"
#include "utils.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <vector>

using namespace std;
using namespace smt;

namespace {
map<string, DeclaredFunction, std::less<>> calleeMap;

vector<uint64_t> getShapeDimVector(const mlir::ShapedType shapedTy) {
  smart_assert(shapedTy.hasStaticShape(), "Not having static shape: "
                                          << shapedTy);
  const auto dims = shapedTy.getShape();
  return vector<uint64_t>(dims.begin(), dims.end());
}
} // namespace

DeclaredFunction::DeclaredFunction(vector<mlir::Type> &&domain,
                                   mlir::Type &&range, FnDecl &&decl)
    : domain(move(domain)), range(move(range)), decl(move(decl)) {}

DeclaredFunction DeclaredFunction::declare(std::vector<mlir::Type> &&domain,
                                           mlir::Type &&range,
                                           const std::string_view name) {
  auto typeToSort = [](mlir::Type t) {
    if (auto tty = t.dyn_cast<mlir::TensorType>()) {
      if (!tty.hasRank()) {
        throw UnsupportedException("A call with an unranked tensor as operand "
                                   "or return value is not supported");
      } else if (!tty.hasStaticShape()) {
        throw UnsupportedException("A call with a dynamically sized tensor "
                                   "as operand or return value is not"
                                   " supported");
      } else if (!Tensor::isTypeSupported(tty)) {
        throw UnsupportedException("Unsupported tensor type: " +
                                   to_string(tty));
      }

      return Tensor::getSort(tty.getElementType());
    } else if (t.isa<mlir::MemRefType>()) {
      throw UnsupportedException(
          "Function call with memref operand(s) or return type is unsupported");
    } else if (t.isIntOrIndexOrFloat()) {
      auto s = convertPrimitiveTypeToSort(t);
      if (!s)
        throw UnsupportedException("Unsupported scalar type: " + to_string(t));
      return *s;
    }
    throw UnsupportedException("Unsupported type: " + to_string(t));
  };

  vector<Sort> smtDomain;
  smtDomain.reserve(domain.size());
  transform(domain.cbegin(), domain.cend(), back_inserter(smtDomain),
            typeToSort);
  FnDecl decl(smtDomain, typeToSort(range), string(name) + "_tvfn");
  return DeclaredFunction(move(domain), move(range), move(decl));
}

ValueTy DeclaredFunction::apply(const std::vector<ValueTy> &operands) const {
  vector<Expr> operandExprs;
  operandExprs.reserve(operands.size());

  transform(operands.cbegin(), operands.cend(), back_inserter(operandExprs),
            getExpr);
  if (range.isIntOrIndexOrFloat()) {
    auto fn_output = fromExpr(decl.apply(operandExprs), range);
    smart_assert(fn_output, "Cannot create ValueTy from the call's result"
                            " because its MLIR type is "
                                << range);
    return *fn_output;
  } else if (auto tensorRange = range.dyn_cast<mlir::TensorType>()) {
    smart_assert(tensorRange.hasStaticShape(), "The range is a dynamically"
                 " sized tensor type; UnsupportedException must have been "
                 "thrown");
    const auto dims = getShapeDimVector(tensorRange);
    auto fn_output =
        Tensor(tensorRange.getElementType(), decl.apply(operandExprs), dims);
    return fn_output;
  } else {
    smart_assert(false, "Cannot create ValueTy from the call's result"
                        " because its MLIR type is " << range);
  }
}

optional<DeclaredFunction> getDeclaredFunction(const std::string_view name) {
  auto fn_itr = calleeMap.find(name);
  if (fn_itr != calleeMap.end()) {
    return fn_itr->second;
  } else {
    return nullopt;
  }
}

bool declareFunction(vector<mlir::Type> &&domain, mlir::Type &&range,
                     const string_view name) {
  if (getDeclaredFunction(name)) {
    // no-op if there already exists a function of the same name
    return false;
  } else {
    llvm::outs() << "WARNING: Function \"" << name << "\" is assumed to be "
                 << "stateless and does not read or write global memory\n";

    calleeMap.insert({string(name), DeclaredFunction::declare(
                                        move(domain), move(range), name)});
    return true;
  }
}
