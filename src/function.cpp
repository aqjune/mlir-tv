#include "function.h"
#include "utils.h"

#include <algorithm>
#include <map>
#include <vector>

using namespace std;
using namespace smt;

namespace {
map<string, DeclaredFunction, std::less<>> calleeMap;
}

DeclaredFunction::DeclaredFunction(vector<mlir::Type> &&domain,
                                   mlir::Type &&range, FnDecl &&decl)
    : domain(move(domain)), range(move(range)), decl(move(decl)) {}

DeclaredFunction DeclaredFunction::declare(std::vector<mlir::Type> &&domain,
                                           mlir::Type &&range,
                                           const std::string_view name) {
  // analyze verification complexity
  auto verifComplexity = Complexity::SCALAR;
  for (const auto &operand_ty : domain) {
    if (operand_ty.isa<mlir::TensorType>()) {
      verifComplexity = Complexity::TENSOR;
    } else if (operand_ty.isa<mlir::MemRefType>()) {
      verifComplexity = Complexity::MEMREF;
    } else if (!operand_ty.isIntOrIndexOrFloat()) {
      throw UnsupportedException("Invalid operand type");
    }
  }

  auto getScalarSort = [](mlir::Type ty) {
    auto s = convertPrimitiveTypeToSort(ty);
    smart_assert(s, "A primitive type must be given, but got " << ty);
    return *s;
  };

  switch (verifComplexity) {
  case Complexity::SCALAR: {
    vector<Sort> smtDomain;
    smtDomain.reserve(domain.size());
    transform(domain.cbegin(), domain.cend(), back_inserter(smtDomain),
              getScalarSort);

    if (range.isa<mlir::TensorType>()) {
      throw UnsupportedException(
          "Function that returns tensor is not supported yet");
    }
    if (range.isa<mlir::MemRefType>()) {
      throw UnsupportedException(
          "Function that returns memref is not supported yet");
    }

    const auto smtRange = getScalarSort(range);
    FnDecl decl(smtDomain, smtRange, string(name) + "_tvfn");

    return DeclaredFunction(move(domain), move(range), move(decl));
  }
  case Complexity::TENSOR: {
    throw UnsupportedException(
        "Function call with tensor operand(s) is unsupported");
  }
  case Complexity::MEMREF: {
    throw UnsupportedException(
        "Function call with memref operand(s) is unsupported");
  }
  }

  llvm_unreachable("Invalid verification complexity");
}

ValueTy DeclaredFunction::apply(const std::vector<ValueTy> &operands) const {
  vector<Expr> operandExprs;
  operandExprs.reserve(operands.size());

  transform(operands.cbegin(), operands.cend(), back_inserter(operandExprs),
            getExpr);
  auto fn_output = fromExpr(decl.apply(operandExprs), range);
  smart_assert(fn_output, "Cannot create ValueTy from the call's result"
      " because its MLIR type is " << range);
  return *fn_output;
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
    calleeMap.insert({string(name), DeclaredFunction::declare(
                                        move(domain), move(range), name)});
    return true;
  }
}
