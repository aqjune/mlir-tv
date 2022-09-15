#pragma once

#include "smt.h"
#include "value.h"

#include "mlir/IR/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class DeclaredFunction {
private:
  std::vector<mlir::Type> domain;
  mlir::Type range;
  smt::FnDecl decl;
  std::vector<smt::FnDecl> dims;
  std::optional<int64_t> rangeDimRefIdx;

  DeclaredFunction(std::vector<mlir::Type> &&domain, mlir::Type &&range,
                   smt::FnDecl &&decl, std::vector<smt::FnDecl> &&dims,
                   std::optional<int64_t> &&rangeDimRefIdx);

public:
  static DeclaredFunction declare(std::vector<mlir::Type> &&domain,
                                  mlir::Type &&range,
                                  const std::string_view name,
                                  std::optional<int64_t> &&rangeDimRefIdx);

  ValueTy apply(const std::vector<ValueTy> &operands) const;
};

std::optional<DeclaredFunction>
getDeclaredFunction(const std::string_view name);
bool declareFunction(std::vector<mlir::Type> &&domain, mlir::Type &&range,
                     const std::string_view name,
                     std::optional<int64_t> &&dimsReferenceIdx);
