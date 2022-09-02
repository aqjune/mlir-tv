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
  bool hasTensor = false, hasMemRef = false;
  for (const auto &operand_ty : domain) {
    if (operand_ty.isa<mlir::TensorType>()) {
      hasTensor = true;
    } else if (operand_ty.isa<mlir::MemRefType>()) {
      hasMemRef = true;
    } else if (!operand_ty.isIntOrIndexOrFloat()) {
      throw UnsupportedException("Invalid operand type");
    }
  }

  if (hasTensor) {
    throw UnsupportedException(
        "Function call with tensor operand(s) is unsupported");
  }

  if (hasMemRef) {
    throw UnsupportedException(
        "Function call with memref operand(s) is unsupported");
  }

  auto getScalarSort = [](mlir::Type ty) {
    auto s = convertPrimitiveTypeToSort(ty);
    smart_assert(s, "A primitive type must be given, but got " << ty);
    return *s;
  };

  vector<Sort> smtDomain;
  smtDomain.reserve(domain.size());
  transform(domain.cbegin(), domain.cend(), back_inserter(smtDomain),
            getScalarSort);

  if (range.isIntOrIndexOrFloat()) {
    const auto smtRange = getScalarSort(range);
    FnDecl decl(smtDomain, smtRange, string(name) + "_tvfn");
    return DeclaredFunction(move(domain), move(range), move(decl));
  } else if (range.isa<mlir::TensorType>()) {
    const auto tensorRange = range.dyn_cast<mlir::TensorType>();
    const auto tensorElementType = tensorRange.getElementType();
    const auto dims = getShapeDimVector(tensorRange);
    const auto smtRange =
        static_cast<Expr>(Tensor::var(tensorElementType,
                                      string(name) + "_tv_ret_tensor", dims))
            .sort();

    FnDecl decl(smtDomain, smtRange, string(name) + "_tvfn");
    return DeclaredFunction(move(domain), move(range), move(decl));
  } else if (range.isa<mlir::MemRefType>()) {
    throw UnsupportedException(
        "Function that returns memref is not supported yet");
  } else {
    throw UnsupportedException("Invalid return type");
  }
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
  } else if (range.isa<mlir::TensorType>()) {
    const auto tensorRange = range.dyn_cast<mlir::TensorType>();
    const auto dims = getShapeDimVector(tensorRange);
    auto fn_output =
        Tensor(tensorRange.getElementType(), decl.apply(operandExprs), dims);
    return fn_output;
  } else {
    smart_assert(false, "Cannot create ValueTy from the call's result"
                        " because its MLIR type is "
                            << range);
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
