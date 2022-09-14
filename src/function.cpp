#include "function.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <variant>

using namespace std;
using namespace smt;

namespace {
map<string, DeclaredFunction, std::less<>> calleeMap;
} // namespace

DeclaredFunction::DeclaredFunction(vector<mlir::Type> &&domain,
                                   mlir::Type &&range, FnDecl &&decl,
                                   vector<FnDecl> &&dims,
                                   optional<int64_t> &&rangeDimRefIdx)
    : domain(move(domain)), range(move(range)), decl(move(decl)),
      dims(move(dims)), rangeDimRefIdx(move(rangeDimRefIdx)) {}

DeclaredFunction DeclaredFunction::declare(std::vector<mlir::Type> &&domain,
                                           mlir::Type &&range,
                                           const std::string_view name,
                                           optional<int64_t> &&rangeDimRefIdx) {
  auto typeToSort = [](mlir::Type t) {
    if (auto tty = t.dyn_cast<mlir::TensorType>()) {
      if (!tty.hasRank()) {
        throw UnsupportedException("A call with an unranked tensor as operand "
                                   "or return value is not supported");
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

  if (rangeDimRefIdx) {
    const auto dimRefIdx = *rangeDimRefIdx;
    smart_assert(dimRefIdx < domain.size(),
                 "Tried to refer to the argument of an invalid index");

    const auto shapedDimRef = domain[dimRefIdx].dyn_cast<mlir::ShapedType>();
    const auto shapedRange = range.dyn_cast<mlir::ShapedType>();
    smart_assert(
        shapedDimRef && shapedRange,
        "Both the specified domain and the range must be shaped types");
    smart_assert(shapedDimRef.getRank() == shapedRange.getRank(),
                 "The specified domain and the range must have the same ranks");

    for (size_t r = 0; r < shapedDimRef.getRank(); r++) {
      if (shapedDimRef.isDynamicDim(r) != shapedRange.isDynamicDim(r)) {
        throw UnsupportedException("The dimensions of the specified domain and "
                                   "the range are incompatible");
      } else if (!shapedDimRef.isDynamicDim(r) && shapedRange.isDynamicDim(r) &&
                 shapedDimRef.getDimSize(r) != shapedRange.getDimSize(r)) {
        throw UnsupportedException("The dimensions of the specified domain and "
                                   "the range are incompatible");
      }
    }
  }

  vector<Sort> smtDomain;
  for (const auto operandTy : domain) {
    smtDomain.push_back(typeToSort(operandTy));
    if (auto tensorOperandTy = operandTy.dyn_cast<mlir::TensorType>()) {
      const auto rank = tensorOperandTy.getRank();
      for (size_t i = 0; i < rank; i++) {
        smtDomain.push_back(Index::sort());
      }
    }
  }
  FnDecl decl(smtDomain, typeToSort(range), string(name) + "_tvfn");

  vector<FnDecl> dims;
  if (auto sty = range.dyn_cast<mlir::ShapedType>()) {
    const auto rank = sty.getRank();
    dims.reserve(rank);
    const auto dimPrefix = string(name) + "_tvfn_dim_";
    for (size_t i = 0; i < rank; i++) {
      auto dim = FnDecl(smtDomain, Index::sort(), dimPrefix + to_string(i));
      dims.push_back(move(dim));
    }
  }
  return DeclaredFunction(move(domain), move(range), move(decl), move(dims),
                          move(rangeDimRefIdx));
}

ValueTy DeclaredFunction::apply(const std::vector<ValueTy> &operands) const {
  vector<Expr> operandExprs;

  auto getZeroGuardedExpr = [](const ValueTy &val) {
    if (holds_alternative<Tensor>(val)) {
      const auto tensorVal = get<Tensor>(val);
      Expr numElements = tensorVal.get1DSize();

      const Expr arr = tensorVal;
      const auto i = static_cast<Expr>(Index::var("idx", VarType::BOUND));

      const auto elemType = tensorVal.getElemType();
      auto zero = getZero(elemType);
      if (!zero) {
        smart_assert(false, "Invalid tensor element type " << elemType);
      }
      return Expr::mkLambda(
          i, Expr::mkIte(i.ult(numElements), arr.select(i), *zero));
    } else {
      return getExpr(val);
    }
  };

  for (const auto &operandVal : operands) {
    operandExprs.push_back(getZeroGuardedExpr(operandVal));
    if (holds_alternative<Tensor>(operandVal)) {
      const auto dims = get<Tensor>(operandVal).getDims();
      operandExprs.insert(end(operandExprs), dims.cbegin(), dims.cend());
    }
  }

  if (range.isIntOrIndexOrFloat()) {
    auto fn_output = fromExpr(decl.apply(operandExprs), range);
    smart_assert(fn_output, "Cannot create ValueTy from the call's result"
                            " because its MLIR type is "
                                << range);
    return *fn_output;
  } else if (auto tensorRange = range.dyn_cast<mlir::TensorType>()) {
    vector<Expr> dims;
    const auto rank = tensorRange.getRank();
    dims.reserve(rank);

    if (rangeDimRefIdx) {
      // dim reference argument is given
      const auto dimRefVal = operands[*rangeDimRefIdx];
      if (holds_alternative<Tensor>(dimRefVal)) {
        const auto refDims = get<Tensor>(dimRefVal).getDims();
        dims.insert(end(dims), refDims.cbegin(), refDims.cend());
      }
    } else {
      for (size_t i = 0; i < rank; i++) {
        if (tensorRange.isDynamicDim(i)) {
          dims.push_back(this->dims[i].apply(operandExprs));
        } else {
          // static dimension does not need to be obtained via UF
          dims.push_back(Index(tensorRange.getDimSize(i)));
        }
      }
    }

    auto fn_output = Tensor::fromArray(tensorRange.getElementType(),
                                       decl.apply(operandExprs), move(dims));
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
                     const string_view name,
                     optional<int64_t> &&dimsReferenceIdx) {
  if (getDeclaredFunction(name)) {
    // no-op if there already exists a function of the same name
    return false;
  } else {
    llvm::outs() << "WARNING: Function \"" << name << "\" is assumed to be "
                 << "stateless and does not read or write global memory\n";

    calleeMap.insert({string(name),
                      DeclaredFunction::declare(move(domain), move(range), name,
                                                move(dimsReferenceIdx))});
    return true;
  }
}
