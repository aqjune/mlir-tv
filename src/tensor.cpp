#include "tensor.h"
#include "smt.h"

using namespace mlir;


Tensor::Tensor(): arr(ctx) {}

Tensor Tensor::newVar(TensorType tensorTy, const std::string &name) {
  Tensor t;

  uint64_t rank = tensorTy.getRank();
  for (auto i = 0; i < rank; ++i) {
    t.dims.emplace_back(ctx.bv_val(tensorTy.getDimSize(i), BITS_INDEX));
  }
  t.arr = ctx.constant(name.c_str(),
        ctx.array_sort(ctx.bv_sort(BITS_INDEX), ctx.bv_sort(BITS_FLOAT)));

  return t;
}

z3::expr Tensor::newIdxConst(uint64_t idx) {
  return ctx.bv_val(idx, BITS_INDEX);
}

z3::expr Tensor::newIdxVar(const std::string &name) {
  return ctx.bv_const(name.c_str(), BITS_INDEX);
}

std::vector<z3::expr>
Tensor::newIdxVars(const std::vector<std::string> &names) {
  std::vector<z3::expr> v;
  for (auto &n: names)
    v.emplace_back(newIdxVar(n));
  return v;
}


llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Tensor &t) {
  os << t.arr << "(dim :" << t.dims[0];
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << t.dims[i];
  os << ")";
  return os;
};

